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

        # Current step state
        self.current_actions: Dict[str, float] = {}
        self.current_step = 0

        # Historical logs for validation
        self.history = {
            "steps": [],
            "error_vector": [],
            "reward_vector": [],
            "advantage_vector": [],
            "phase": [],
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

    def preprocess_signal(self, raw_signal: np.ndarray, zero_shift: float = 0.0) -> np.ndarray:
        """
        Preprocesses output or reference signal:
        1. Zero-point shift
        2. Moving average based on convolution matrix
        """
        sig = np.asarray(raw_signal, dtype=np.float64) - zero_shift
        self.output_buffer.append(sig)
        if len(self.output_buffer) > self.smooth_kernel_size:
            self.output_buffer.pop(0)

        # Apply moving average across the buffer
        buf_arr = np.array(self.output_buffer)
        smoothed = np.mean(buf_arr, axis=0)
        return smoothed

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
        tgt = np.asarray(target, dtype=np.float64)

        # Vector error and reward
        error_vec = self.compute_vector_error(smoothed_out, tgt)
        reward_vec = -error_vec

        # Component-wise advantage with phase-separated moving average baselines
        advantage_vec = self.strategy.compute_advantage(reward_vec, phase=phase)

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

        # Record step in Fisher Matrix strategy
        self.strategy.record_step(
            perturbation_dict=perturbations,
            advantage_vector=advantage_vec,
            reward_vector=reward_vec,
            phase=phase
        )

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

            # Parametric variance fine adjustment (population dispersion)
            # Parametric stddev shifts slightly towards optimal homogeneity/heterogeneity
            param.update_parametric_variance(
                delta=self.lr_sigma_param * np.sign(adv_scalar) * 0.01
            )

        # Logging
        self.history["steps"].append(self.current_step)
        self.history["error_vector"].append(error_vec.copy())
        self.history["reward_vector"].append(reward_vec.copy())
        self.history["advantage_vector"].append(advantage_vec.copy())
        self.history["phase"].append(phase)
        for k, p in self.params.items():
            self.history["means"][k].append(p.mean)
            self.history["sigmas_ctrl"][k].append(p.sigma_ctrl)
            self.history["sigmas_param"][k].append(p.sigma_param)

        return {
            "step": self.current_step,
            "error_vector": error_vec,
            "reward_vector": reward_vec,
            "advantage_vector": advantage_vec,
            "phase": phase,
            "delta_means": delta_means
        }