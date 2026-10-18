from typing import Dict, List, Any
import numpy as np

from persona.registry import Registry
from persona.strategy import FisherMatrix
from persona.parameter import PersonaParam


class Agent:
    """
    PERSONA Agent implementing online PPO with Natural Gradient (Fisher Information Matrix),
    separated Stance/Swing phase baselines, noise integration, and dual standard deviation
    (controlling variance for exploration/exploitation and parametric variance for biological heterogeneity).
    """
    def __init__(self,
                 registry: Registry,
                 strategy: FisherMatrix | None = None,
                 lr_mu: float = 0.02,
                 lr_sigma_ctrl: float = 0.005,
                 lr_sigma_param: float = 0.001,
                 window_size: int = 50,
                 smooth_kernel_size: int = 5):
        self.registry = registry
        self.params: Dict[str, PersonaParam] = self.registry.get_all_optimizable_params()

        param_names = list(self.params.keys())
        if strategy is not None:
            self.strategy = strategy
        else:
            self.strategy = FisherMatrix(
                param_names=param_names,
                lr_mu=lr_mu,
                lr_sigma=lr_sigma_ctrl,
                window_size=window_size
            )

        self.lr_sigma_ctrl = lr_sigma_ctrl
        self.lr_sigma_param = lr_sigma_param
        self.smooth_kernel_size = smooth_kernel_size

        # Output smoothing filter (convolution matrix / moving average)
        if self.smooth_kernel_size > 1:
            self.kernel = np.ones(self.smooth_kernel_size) / self.smooth_kernel_size
        else:
            self.kernel = np.array([1.0])

        self.output_buffer: List[np.ndarray] = []
        self.target_buffer: List[np.ndarray] = []

        # Current step state
        self.current_actions: Dict[str, float] = {}
        self.current_step = 0

        # Historical logs for validation
        self.history = {
            "steps": [],
            "error_vector": [],
            "loss": [],
            "advantage_vector": [],
            "phase": [],
            "influence": [],
            "means": {k: [] for k in param_names},
            "sigmas_ctrl": {k: [] for k in param_names},
            "sigmas_param": {k: [] for k in param_names},
        }

    def sample_actions(self) -> Dict[str, float]:
        """
        Samples an exploratory action for each parameter using its controlling variance.
        """
        self.current_actions = {}
        for name, param in self.params.items():
            self.current_actions[name] = param.sample_exploration()
        return dict(self.current_actions)

    def _smooth(self, signal: np.ndarray, buffer: List[np.ndarray]) -> np.ndarray:
        buf_arr = np.asarray(signal, dtype=np.float64)
        buffer.append(buf_arr)
        if len(buffer) > self.smooth_kernel_size:
            buffer.pop(0)
        return np.mean(np.array(buffer), axis=0)

    def preprocess_signal(self, raw_signal: np.ndarray, zero_shift: float = 0.0) -> np.ndarray:
        """
        Preprocesses the model output signal:
        1. Zero-point shift
        2. Moving average over the last `smooth_kernel_size` steps
        """
        sig = np.asarray(raw_signal, dtype=np.float64) - zero_shift
        return self._smooth(sig, self.output_buffer)

    def preprocess_target(self, raw_target: np.ndarray) -> np.ndarray:
        """
        Applies the same moving average to the target so output and target
        share one filter (no differential lag between the two signals).
        """
        return self._smooth(raw_target, self.target_buffer)

    def compute_vector_error(self, output: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Vector error: E_v = |target - output|
        """
        out = np.asarray(output, dtype=np.float64)
        tgt = np.asarray(target, dtype=np.float64)
        return np.abs(tgt - out)

    def step(self,
             output: np.ndarray,
             target: np.ndarray,
             phase: str = "stance",
             zero_shift: float = 0.0) -> Dict[str, Any]:
        """
        Runs one optimization step:
        1. Preprocesses output signal.
        2. Computes vector error and vector reward.
        3. Separates stance/swing baseline and computes component-wise advantage.
        4. Gathers noise producer deltas.
        5. Computes natural gradient / Fisher step.
        6. Updates parameter means, controlling stddevs, and parametric stddevs.
        7. Enforces homeostasis (clipping).
        """
        self.current_step += 1
        smoothed_out = self.preprocess_signal(output, zero_shift=zero_shift)
        smoothed_tgt = self.preprocess_target(target)

        # Vector error and scalarized loss
        # TODO(advantage): component-wise vector error/advantage (spec §5.1);
        # currently the loss is scalar (mean of the error vector).
        error_vec = self.compute_vector_error(smoothed_out, smoothed_tgt)
        loss = float(np.mean(error_vec))

        # Normalize phase: str broadcasts to one key per output component;
        # a sequence is keyed per component (per-leg phase).
        if isinstance(phase, str):
            phase_key = phase
        else:
            phase_key = tuple(phase)

        # Component-wise advantage with phase-separated moving-average baselines
        advantage_vec = self.strategy.compute_advantage(error_vec, phase=phase_key)

        # Collect perturbations: parameter delta = action - mean
        perturbations: Dict[str, float] = {}
        for name, param in self.params.items():
            a = self.current_actions.get(name, param.mean)
            perturbations[name] = a - param.mean

        # Collect noise producer fluctuations
        for nr_name, nr in self.registry.noise_recorders.items():
            noise_deltas = nr.get_latest_step_deltas()
            for tag, d_val in noise_deltas.items():
                perturbations[tag] = d_val

        # Record step in Fisher Matrix strategy (legacy perturbation history, kept
        # for the Fisher heat-map figure)
        self.strategy.record_step(
            perturbation_dict=perturbations,
            loss_vector=error_vec,
            phase=phase
        )

        # Natural gradient: pair the exploration perturbation with the reward
        # advantage A = baseline - error (per leg).
        dtheta = np.zeros(self.strategy.n_total, dtype=np.float64)
        for i, name in enumerate(self.strategy.all_names):
            dtheta[i] = perturbations.get(name, 0.0)
        self.strategy.record_response(dtheta, advantage_vec)

        # Compute natural gradient update
        mean_dict = {k: p.mean for k, p in self.params.items()}
        sigma_ctrl_dict = {k: p.sigma_ctrl for k, p in self.params.items()}

        delta_means = self.strategy.compute_natural_gradient_step(
            action_dict=self.current_actions,
            mean_dict=mean_dict,
            sigma_ctrl_dict=sigma_ctrl_dict,
            advantage_vector=advantage_vec
        )

        # Update parameter means and variances (homeostasis enforced inside PersonaParam)
        adv_scalar = float(np.mean(advantage_vec))
        for name, param in self.params.items():
            if name in delta_means:
                param.update_mean(delta_means[name])

            # Controlling variance update
            action_val = self.current_actions.get(name, param.mean)
            param.update_controlling_variance(
                action=action_val,
                advantage=adv_scalar,
                lr_sigma=self.lr_sigma_ctrl
            )

            # TODO(sigma_param): sigma_param (biological dispersion) is held fixed
            # for the MVP; registry applies the scalar explored value (`inert`).
            # Promote log(sigma_param) to a feature in theta in a later revision.

        influence = self.strategy.influence()

        # Logging
        self.history["steps"].append(self.current_step)
        self.history["error_vector"].append(error_vec.copy())
        self.history["loss"].append(loss)
        self.history["advantage_vector"].append(advantage_vec.copy())
        self.history["phase"].append(phase_key)
        self.history["influence"].append(influence.copy())
        for k, p in self.params.items():
            self.history["means"][k].append(p.mean)
            self.history["sigmas_ctrl"][k].append(p.sigma_ctrl)
            self.history["sigmas_param"][k].append(p.sigma_param)

        return {
            "step": self.current_step,
            "error_vector": error_vec,
            "loss": loss,
            "advantage_vector": advantage_vec,
            "phase": phase_key,
            "influence": influence,
            "delta_means": delta_means
        }