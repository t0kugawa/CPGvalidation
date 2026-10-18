from typing import Any, Tuple
import nest
import numpy as np

from persona.parameter import PersonaParam
from persona.population import Population
from persona.connection import Connection
from persona.noise_recorder import NoiseRecorder
from persona.registry import Registry
from t2xcpg.utils import (
    get_jitter_delay, create_spike_recorder, create_stdp_synapse, STDP_POSTFIX,
    stdp_param_spec,
)


class Leg:
    """
    CPG Leg network matching Diagram 2.
    Implements:
    - Rhythm generators (RGE, RGF) with reciprocal inhibition (InE, InF).
    - Motor neurons (M-E, M-F) and Muscle outputs (mE, mF).
    - Renshaw cells (R-E, R-F) mediating recurrent and Ia-inhibitory feedback.
    - Ia afferent stretch pathways (Ia-E, Ia-F) exciting homonymous MN/RG and reciprocal interneurons.
    - Ib afferent tension pathways inhibiting homonymous MNs against excessive strain.
    - Cutaneous (CUT) afferents reinforcing stance phase.
    All neuronal and synaptic parameters are registered via PERSONA Registry.
    """
    def __init__(self,
                 leg_name: str,
                 registry: Registry,
                 n_rg: int = 20,
                 n_mn: int = 20,
                 n_int: int = 15,
                 n_aff: int = 20,
                 p_scale: float = 1.0,
                 w_scale: float = 1.0,
                 ref_hz: float = 100.0):
        self.leg = leg_name
        self.registry = registry
        self.n_rg = n_rg
        self.n_mn = n_mn
        self.n_int = n_int
        self.n_aff = n_aff
        self.p_scale = float(p_scale)
        self.w_scale = float(w_scale)
        self.ref_hz = float(ref_hz)
        self.plastic_projections: list[str] = []

        # Create Noise Recorder for this leg
        self.noise_recorder = NoiseRecorder(f"noise_{leg_name}")
        self.registry.register_noise_recorder(self.noise_recorder)

        # 1. Build Neuronal Populations
        self._build_populations()

        # 2. Build Synaptic Connections
        self._build_connections()

        # 3. Readout Recorders
        self.rec_muse = create_spike_recorder(self.mus_e)
        self.rec_musf = create_spike_recorder(self.mus_f)
        self.rec_rge = create_spike_recorder(self.rg_e)
        self.rec_rgf = create_spike_recorder(self.rg_f)

        self.registry.finalize_connections()

    def _param(self, name: str, mean: float, sigma_param: float = 0.05,
               sigma_ctrl: float = 0.1, min_val: float = -100.0, max_val: float = 100.0) -> PersonaParam:
        return PersonaParam(
            mean=mean,
            sigma_param=sigma_param,
            sigma_ctrl=sigma_ctrl,
            min=min_val,
            max=max_val
        )

    _STDP_RANGES = {
        "tau_plus": (1.0, 200.0),
        "lambda": (1e-5, 0.1),
        "alpha": (0.0, 1.0),
        "mu_plus": (0.0, 1.0),
        "mu_minus": (0.0, 1.0),
        "Wmax": (1.0, 2000.0),
    }

    def _stdp_params(self, syn_type: str, weight: float) -> dict:
        """
        Optimizable STDP hyper-parameters for one plastic projection. The
        synaptic weight is owned by STDP and is only seeded (never optimized);
        Wmax is kept above the seed weight so the initial synapse is valid.
        """
        out = {}
        for name, mean in stdp_param_spec(syn_type).items():
            mean = float(mean)
            lo, hi = self._STDP_RANGES.get(name, (-1e6, 1e6))
            if name == "Wmax":
                mean = max(mean, abs(weight) * 1.01)
                lo = max(lo, abs(weight))
            out[name] = self._param(
                name, mean,
                sigma_param=max(abs(mean) * 0.05, 1e-6),
                sigma_ctrl=max(abs(mean) * 0.1, 1e-6),
                min_val=lo, max_val=hi)
        return out

    def _izhi(self, a, b, c, d):
        return {
            "a": self._param("a", a, 0.002, 0.005, 0.001, 0.1),
            "b": self._param("b", b, 0.02, 0.04, 0.05, 0.5),
            "c": self._param("c", c, 2.0, 5.0, -80.0, -50.0),
            "d": self._param("d", d, 1.0, 2.0, 0.0, 20.0),
        }

    def _build_populations(self):
        # Izhikevich parameters from cpg_2legs_fast: regular-spiking excitatory,
        # fast-spiking inhibitory, and intrinsically-bursting flexor RG.
        def exc():
            return self._izhi(0.02, 0.2, -65.0, 8.0)

        def inh():
            return self._izhi(0.1, 0.2, -65.0, 2.0)

        def rgf():
            return self._izhi(0.02, 0.2, -55.0, 4.0)

        # Rhythm Generators
        self.rg_e = self.registry.create_population(
            Population(model="izhikevich", n=self.n_rg, params=exc(), name=f"rg_e_{self.leg}")
        )
        self.rg_f = self.registry.create_population(
            Population(model="izhikevich", n=self.n_rg, params=rgf(), name=f"rg_f_{self.leg}")
        )

        # Motor Neurons
        self.m_e = self.registry.create_population(
            Population(model="izhikevich", n=self.n_mn, params=exc(), name=f"m_e_{self.leg}")
        )
        self.m_f = self.registry.create_population(
            Population(model="izhikevich", n=self.n_mn, params=exc(), name=f"m_f_{self.leg}")
        )

        # Renshaw Cells (R-E, R-F)
        self.r_e = self.registry.create_population(
            Population(model="izhikevich", n=self.n_int, params=inh(), name=f"r_e_{self.leg}")
        )
        self.r_f = self.registry.create_population(
            Population(model="izhikevich", n=self.n_int, params=inh(), name=f"r_f_{self.leg}")
        )

        # Ia inhibitory interneurons (Ia-E, Ia-F): speed/stretch reciprocal inhibition
        self.ia_e = self.registry.create_population(
            Population(model="izhikevich", n=self.n_int, params=inh(), name=f"ia_e_{self.leg}")
        )
        self.ia_f = self.registry.create_population(
            Population(model="izhikevich", n=self.n_int, params=inh(), name=f"ia_f_{self.leg}")
        )

        # Ib inhibitory interneurons (InIb-E, InIb-F): force/load autogenic inhibition
        self.in_ib_e = self.registry.create_population(
            Population(model="izhikevich", n=self.n_int, params=inh(), name=f"in_ib_e_{self.leg}")
        )
        self.in_ib_f = self.registry.create_population(
            Population(model="izhikevich", n=self.n_int, params=inh(), name=f"in_ib_f_{self.leg}")
        )

        # Excitatory relay interneurons (InE, InF)
        self.in_e = self.registry.create_population(
            Population(model="izhikevich", n=self.n_int, params=exc(), name=f"in_e_{self.leg}")
        )
        self.in_f = self.registry.create_population(
            Population(model="izhikevich", n=self.n_int, params=exc(), name=f"in_f_{self.leg}")
        )

        # Muscles (Parrot readout)
        self.mus_e = nest.Create("parrot_neuron", self.n_mn)
        self.mus_f = nest.Create("parrot_neuron", self.n_mn)
        self.registry.population_nodes[f"mus_e_{self.leg}"] = self.mus_e
        self.registry.population_nodes[f"mus_f_{self.leg}"] = self.mus_f

        # Sensory Feedback Generators (Poisson -> Parrot)
        # 1. CUT (cutaneous)
        self.cut_pg = nest.Create("poisson_generator", self.n_aff, params={"rate": 0.0})
        self.cut_in = nest.Create("parrot_neuron", self.n_aff)
        nest.Connect(self.cut_pg, self.cut_in, conn_spec={"rule": "one_to_one"})
        self.registry.population_nodes[f"cut_in_{self.leg}"] = self.cut_in

        # 2. Ia Afferents (stretch length & velocity)
        self.ia_e_pg = nest.Create("poisson_generator", self.n_aff, params={"rate": 10.0})
        self.ia_e_in = nest.Create("parrot_neuron", self.n_aff)
        nest.Connect(self.ia_e_pg, self.ia_e_in, conn_spec={"rule": "one_to_one"})
        self.registry.population_nodes[f"ia_e_in_{self.leg}"] = self.ia_e_in

        self.ia_f_pg = nest.Create("poisson_generator", self.n_aff, params={"rate": 10.0})
        self.ia_f_in = nest.Create("parrot_neuron", self.n_aff)
        nest.Connect(self.ia_f_pg, self.ia_f_in, conn_spec={"rule": "one_to_one"})
        self.registry.population_nodes[f"ia_f_in_{self.leg}"] = self.ia_f_in

        # 3. Ib Afferents (strain force & activation)
        self.ib_e_pg = nest.Create("poisson_generator", self.n_aff, params={"rate": 5.0})
        self.ib_e_in = nest.Create("parrot_neuron", self.n_aff)
        nest.Connect(self.ib_e_pg, self.ib_e_in, conn_spec={"rule": "one_to_one"})
        self.registry.population_nodes[f"ib_e_in_{self.leg}"] = self.ib_e_in

        self.ib_f_pg = nest.Create("poisson_generator", self.n_aff, params={"rate": 5.0})
        self.ib_f_in = nest.Create("parrot_neuron", self.n_aff)
        nest.Connect(self.ib_f_pg, self.ib_f_in, conn_spec={"rule": "one_to_one"})
        self.registry.population_nodes[f"ib_f_in_{self.leg}"] = self.ib_f_in

    def _connect(self, src: str, tgt: str, weight: float, p: float = 0.5,
                 is_stdp: bool = False, delay: float | None = None,
                 key: str | None = None) -> Any:
        d = get_jitter_delay() if delay is None else delay
        p = p * self.p_scale
        weight = weight * self.w_scale

        model = "static_synapse"
        wr = None
        if is_stdp:
            syn_base = key.rsplit("_", 1)[0] if key else "default"
            syn_type = STDP_POSTFIX.get(syn_base, "default")
            model, wr = create_stdp_synapse(syn_type, f"{src}_{tgt}")
            # STDP owns the synaptic weight; PERSONA optimizes only the STDP
            # hyper-parameters (tau_plus/lambda/alpha/mu/Wmax). The seed weight
            # is written once via the syn_spec and never overwritten afterwards.
            parameters = self._stdp_params(syn_type, weight)
            extra = {"delay": float(d), "weight": float(weight)}
        else:
            parameters = {"weight": self._param(
                "weight", weight, sigma_param=abs(weight) * 0.05,
                sigma_ctrl=abs(weight) * 0.1,
                min_val=-150.0 if weight < 0 else 0.0,
                max_val=0.0 if weight < 0 else 150.0)}
            extra = {"delay": float(d)}

        cfg = Connection(
            source=src,
            target=tgt,
            model=model,
            parameters=parameters,
            rule="pairwise_bernoulli",
            p=float(p),
            noise_recorder_id=self.noise_recorder.name,
            extra_syn_spec=extra,
            key=key,
            weight_recorder=wr
        )
        if key is not None and key not in self.plastic_projections:
            self.plastic_projections.append(key)
        return self.registry.create_connection(cfg)

    def update_plastic_weights(self, t_before: float, t_now: float):
        """
        Samples STDP weight changes from the weight recorders and records them
        as ordinary noise features in this leg's NoiseRecorder.
        """
        deltas = self.registry.sample_plastic_weight_deltas(t_before, t_now)
        for key in self.plastic_projections:
            self.noise_recorder.record_direct(key, t_now, "weight", deltas.get(key, 0.0))

    def _build_connections(self):
        leg = self.leg

        # --- Rhythm-generator half-centre THROUGH the InE/InF relays
        #     (cpg_2legs_fast): RGE->InE (exc), InE-|RGF, RGF->InF (exc), InF-|RGE.
        self._connect(f"rg_e_{leg}", f"in_e_{leg}", weight=12.0, p=0.15)   # W_RG2INE, P_RG_RECIP_E
        self._connect(f"in_e_{leg}", f"rg_f_{leg}", weight=-8.0, p=0.30)   # W_INE2RGF
        self._connect(f"rg_f_{leg}", f"in_f_{leg}", weight=18.0, p=0.30)   # W_RG2INF, P_RG_RECIP_F
        self._connect(f"in_f_{leg}", f"rg_e_{leg}", weight=-48.0, p=0.30)  # W_INF2RGE

        # --- RG self-recurrence ---
        self._connect(f"rg_e_{leg}", f"rg_e_{leg}", weight=4.0, p=0.12)    # W_RG_REC_E, P_RG_REC
        self._connect(f"rg_f_{leg}", f"rg_f_{leg}", weight=5.5, p=0.12)    # W_RG_REC_F

        # --- Motor output: RG -> MN direct; MN -> muscle ---
        self._connect(f"rg_e_{leg}", f"m_e_{leg}", weight=2.75, p=0.50)    # W_RG2MN
        self._connect(f"rg_f_{leg}", f"m_f_{leg}", weight=2.75, p=0.50)
        self._connect(f"m_e_{leg}", f"mus_e_{leg}", weight=1.0, p=0.80)    # W_M2MUS, P_M2MUS
        self._connect(f"m_f_{leg}", f"mus_f_{leg}", weight=1.0, p=0.80)

        # --- Ia (spindle): monosynaptic to homonymous MN/RG + Ia inhibitory IN ---
        self._connect(f"ia_e_in_{leg}", f"m_e_{leg}", weight=12.0, p=0.40)  # Ia(top) -> M-E (exc)
        self._connect(f"ia_f_in_{leg}", f"m_f_{leg}", weight=12.0, p=0.40)  # Ia(bottom) -> M-F (exc)
        # MOD_IA_RG_STDP: homonymous Ia->RG is plastic (spec D12) — Ia-E potentiates
        # onto RG-E, Ia-F onto RG-F; muscle afferents become a learning drive.
        self._connect(f"ia_e_in_{leg}", f"rg_e_{leg}", weight=12.0, p=0.40,
                      is_stdp=True, key=f"ia_rge_{leg}")
        self._connect(f"ia_f_in_{leg}", f"rg_f_{leg}", weight=12.0, p=0.40,
                      is_stdp=True, key=f"ia_rgf_{leg}")
        self._connect(f"ia_e_in_{leg}", f"ia_e_{leg}", weight=6.0, p=0.40)   # W_IA_IN2INT
        self._connect(f"ia_f_in_{leg}", f"ia_f_{leg}", weight=6.0, p=0.40)
        self._connect(f"ia_e_{leg}", f"m_f_{leg}", weight=-10.0, p=0.40)     # W_IA_INT2ANT
        self._connect(f"ia_f_{leg}", f"m_e_{leg}", weight=-10.0, p=0.40)
        self._connect(f"ia_e_in_{leg}", f"in_e_{leg}", weight=6.0, p=0.25)   # W_IA2IN, P_IA2IN
        self._connect(f"ia_f_in_{leg}", f"in_f_{leg}", weight=6.0, p=0.25)

        # --- Recurrent (Renshaw) inhibition (project values) ---
        self._connect(f"m_e_{leg}", f"r_e_{leg}", weight=15.0, p=0.50)     # M-E -> R-E
        self._connect(f"r_e_{leg}", f"m_e_{leg}", weight=-20.0, p=0.50)    # R-E -| M-E
        self._connect(f"r_e_{leg}", f"ia_e_{leg}", weight=-12.0, p=0.40)   # R-E -| Ia-E
        self._connect(f"m_f_{leg}", f"r_f_{leg}", weight=15.0, p=0.50)     # M-F -> R-F
        self._connect(f"r_f_{leg}", f"m_f_{leg}", weight=-20.0, p=0.50)    # R-F -| M-F
        self._connect(f"r_f_{leg}", f"ia_f_{leg}", weight=-12.0, p=0.40)   # R-F -| Ia-F

        # --- Ib Golgi tendon organ, disynaptic via Ib interneurons (project values) ---
        # Extensor Ib: GTO -> InIb-E -> {M-E inh, M-F exc, RGF exc}
        self._connect(f"ib_e_in_{leg}", f"in_ib_e_{leg}", weight=16.0, p=0.40)   # GTO -> InIb-E
        self._connect(f"in_ib_e_{leg}", f"m_e_{leg}", weight=-18.0, p=0.40)      # autogenic inhibition
        self._connect(f"in_ib_e_{leg}", f"m_f_{leg}", weight=12.0, p=0.35)       # disinhibit flexor
        self._connect(f"in_ib_e_{leg}", f"rg_f_{leg}", weight=10.0, p=0.35)      # phase switch -> flexion
        # Flexor Ib: GTO -> InIb-F -> {M-F inh, M-E exc, RGE exc}
        self._connect(f"ib_f_in_{leg}", f"in_ib_f_{leg}", weight=16.0, p=0.40)   # GTO -> InIb-F
        self._connect(f"in_ib_f_{leg}", f"m_f_{leg}", weight=-18.0, p=0.40)      # autogenic inhibition
        self._connect(f"in_ib_f_{leg}", f"m_e_{leg}", weight=12.0, p=0.35)       # disinhibit extensor
        self._connect(f"in_ib_f_{leg}", f"rg_e_{leg}", weight=10.0, p=0.35)      # phase switch -> extension

        # --- Ib interneuron convergence (spindle, CPG) ---
        self._connect(f"ia_e_in_{leg}", f"in_ib_e_{leg}", weight=8.0, p=0.30)    # Ia -> InIb-E
        self._connect(f"ia_f_in_{leg}", f"in_ib_f_{leg}", weight=8.0, p=0.30)    # Ia -> InIb-F
        self._connect(f"rg_e_{leg}", f"in_ib_e_{leg}", weight=8.0, p=0.30)       # RGE -> InIb-E
        self._connect(f"rg_f_{leg}", f"in_ib_f_{leg}", weight=8.0, p=0.30)       # RGF -> InIb-F

        # --- Cutaneous (CUT) stance reinforcement ---
        self._connect(f"cut_in_{leg}", f"rg_e_{leg}", weight=14.0, p=0.35,
                      is_stdp=True, key=f"cut_rge_{leg}")   # W_CUT2RGE, P_CUT2RGE_STATIC
        self._connect(f"cut_in_{leg}", f"in_e_{leg}", weight=6.0, p=0.30)   # W_CUT2INE, P_CUT2INE

    def update_sensory_feedback(self, cut_hz: float,
                                ia_e_hz: float, ia_f_hz: float,
                                ib_e_hz: float, ib_f_hz: float,
                                current_time_ms: float = 0.0):
        """
        Updates Poisson generator rates based on live MuJoCo sensory inputs.
        Records noise fluctuations in the NoiseRecorder.
        """
        self.cut_pg.set({"rate": float(max(0.0, cut_hz))})
        self.ia_e_pg.set({"rate": float(max(0.0, ia_e_hz))})
        self.ia_f_pg.set({"rate": float(max(0.0, ia_f_hz))})
        self.ib_e_pg.set({"rate": float(max(0.0, ib_e_hz))})
        self.ib_f_pg.set({"rate": float(max(0.0, ib_f_hz))})

        self.noise_recorder.observe("cut", current_time_ms, "rate", float(cut_hz))
        self.noise_recorder.observe("ia_e", current_time_ms, "rate", float(ia_e_hz))
        self.noise_recorder.observe("ia_f", current_time_ms, "rate", float(ia_f_hz))
        self.noise_recorder.observe("ib_e", current_time_ms, "rate", float(ib_e_hz))
        self.noise_recorder.observe("ib_f", current_time_ms, "rate", float(ib_f_hz))

    def get_muscle_drives(self, dt_ms: float, current_time_ms: float = 0.0) -> Tuple[float, float]:
        """
        Normalized muscle activation in [0, 1] from spikes during the step.
        A pool at `ref_hz` per neuron (default 100 Hz) maps to full drive, so
        the drive tracks population rate rather than saturating the instant the
        pool fires. The per-step activation change is recorded as a noise
        feature (inner muscle dynamics) for PERSONA.
        """
        ev_e = self.rec_muse.get("events")["times"]
        ev_f = self.rec_musf.get("events")["times"]
        n_e = len(ev_e)
        n_f = len(ev_f)
        self.rec_muse.set({"n_events": 0})
        self.rec_musf.set({"n_events": 0})

        drive_e = self._drive(n_e, dt_ms)
        drive_f = self._drive(n_f, dt_ms)
        self.noise_recorder.observe("mus_e", current_time_ms, "act", drive_e)
        self.noise_recorder.observe("mus_f", current_time_ms, "act", drive_f)
        return drive_e, drive_f

    def _drive(self, spikes: int, dt_ms: float) -> float:
        ref_hz = getattr(self, "ref_hz", 100.0)
        max_spikes = max(1.0, self.n_mn * ref_hz * (dt_ms / 1000.0))
        return float(np.clip(spikes / max_spikes, 0.0, 1.0))
