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
    - Commissural relay interneurons (V2a, V0v, V0d, V3-F, In1).
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

    def _param(self, name: str, mean: float, sigma_param: float = 0.05,
               sigma_ctrl: float = 0.1, min_val: float = -100.0, max_val: float = 100.0) -> PersonaParam:
        return PersonaParam(
            mean=mean,
            sigma_param=sigma_param,
            sigma_ctrl=sigma_ctrl,
            min=min_val,
            max=max_val
        )

    def _build_populations(self):
        # Izhikevich parameters default: regular bursting/spiking
        izhi_p = {
            "a": self._param("a", 0.02, 0.002, 0.005, 0.001, 0.1),
            "b": self._param("b", 0.20, 0.02, 0.04, 0.05, 0.5),
            "c": self._param("c", -65.0, 2.0, 5.0, -80.0, -50.0),
            "d": self._param("d", 8.0, 1.0, 2.0, 0.0, 20.0),
        }

        # Rhythm Generators
        self.rg_e = self.registry.create_population(
            Population(model="izhikevich", n=self.n_rg, params=izhi_p, name=f"rg_e_{self.leg}")
        )
        self.rg_f = self.registry.create_population(
            Population(model="izhikevich", n=self.n_rg, params=izhi_p, name=f"rg_f_{self.leg}")
        )

        # Motor Neurons
        self.m_e = self.registry.create_population(
            Population(model="izhikevich", n=self.n_mn, params=izhi_p, name=f"m_e_{self.leg}")
        )
        self.m_f = self.registry.create_population(
            Population(model="izhikevich", n=self.n_mn, params=izhi_p, name=f"m_f_{self.leg}")
        )

        # Renshaw Cells (R-E, R-F)
        self.r_e = self.registry.create_population(
            Population(model="izhikevich", n=self.n_int, params=izhi_p, name=f"r_e_{self.leg}")
        )
        self.r_f = self.registry.create_population(
            Population(model="izhikevich", n=self.n_int, params=izhi_p, name=f"r_f_{self.leg}")
        )

        # Ia Inhibitory Interneurons (Ia-E, Ia-F)
        self.ia_e = self.registry.create_population(
            Population(model="izhikevich", n=self.n_int, params=izhi_p, name=f"ia_e_{self.leg}")
        )
        self.ia_f = self.registry.create_population(
            Population(model="izhikevich", n=self.n_int, params=izhi_p, name=f"ia_f_{self.leg}")
        )

        # Reciprocal Interneurons (InE, InF, In1)
        self.in_e = self.registry.create_population(
            Population(model="izhikevich", n=self.n_int, params=izhi_p, name=f"in_e_{self.leg}")
        )
        self.in_f = self.registry.create_population(
            Population(model="izhikevich", n=self.n_int, params=izhi_p, name=f"in_f_{self.leg}")
        )
        self.in1 = self.registry.create_population(
            Population(model="izhikevich", n=self.n_int, params=izhi_p, name=f"in1_{self.leg}")
        )

        # Commissural Nuclei (V2a, V0v, V0d, V3-F)
        self.v2a = self.registry.create_population(
            Population(model="izhikevich", n=self.n_int, params=izhi_p, name=f"v2a_{self.leg}")
        )
        self.v0v = self.registry.create_population(
            Population(model="izhikevich", n=self.n_int, params=izhi_p, name=f"v0v_{self.leg}")
        )
        self.v0d = self.registry.create_population(
            Population(model="izhikevich", n=self.n_int, params=izhi_p, name=f"v0d_{self.leg}")
        )
        self.v3_f = self.registry.create_population(
            Population(model="izhikevich", n=self.n_int, params=izhi_p, name=f"v3_f_{self.leg}")
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
        w_param = self._param("weight", weight, sigma_param=abs(weight) * 0.05,
                              sigma_ctrl=abs(weight) * 0.1,
                              min_val=-150.0 if weight < 0 else 0.0,
                              max_val=0.0 if weight < 0 else 150.0)

        p_param = self._param("p", p, sigma_param=0.02, sigma_ctrl=0.05, min_val=0.01, max_val=1.0)

        extra = {"delay": float(d)}
        model = "static_synapse"
        wr = None
        if is_stdp:
            syn_type = STDP_POSTFIX.get(key, "default") if key else "default"
            model, wr = create_stdp_synapse(syn_type, f"{src}_{tgt}")

        cfg = Connection(
            source=src,
            target=tgt,
            model=model,
            parameters={"weight": w_param},
            rule="pairwise_bernoulli",
            p=p_param,
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
            self.noise_recorder.observe_weight(key, t_now, deltas.get(key, 0.0))

    def _build_connections(self):
        leg = self.leg

        # --- RG Reciprocal Inhibition (RG-E <-> RG-F via InE, InF) ---
        self._connect(f"rg_e_{leg}", f"in_e_{leg}", weight=12.0, p=0.40)
        self._connect(f"in_e_{leg}", f"rg_f_{leg}", weight=-24.0, p=0.40)
        self._connect(f"rg_f_{leg}", f"in_f_{leg}", weight=18.0, p=0.50)
        self._connect(f"in_f_{leg}", f"rg_e_{leg}", weight=-45.0, p=0.50)

        # --- RG Self-recurrence ---
        self._connect(f"rg_e_{leg}", f"rg_e_{leg}", weight=4.5, p=0.15)
        self._connect(f"rg_f_{leg}", f"rg_f_{leg}", weight=5.5, p=0.15)

        # --- RG to Motor Neurons ---
        self._connect(f"rg_e_{leg}", f"m_e_{leg}", weight=28.0, p=0.50)
        self._connect(f"rg_f_{leg}", f"m_f_{leg}", weight=28.0, p=0.50)

        # --- Motor Neurons to Muscles ---
        self._connect(f"m_e_{leg}", f"mus_e_{leg}", weight=1.0, p=0.80)
        self._connect(f"m_f_{leg}", f"mus_f_{leg}", weight=1.0, p=0.80)

        # --- Renshaw Recurrent Inhibition (M -> R -> M and R -> Ia) ---
        # Extensor Renshaw
        self._connect(f"m_e_{leg}", f"r_e_{leg}", weight=15.0, p=0.50)
        self._connect(f"r_e_{leg}", f"m_e_{leg}", weight=-20.0, p=0.50)
        self._connect(f"r_e_{leg}", f"ia_e_{leg}", weight=-12.0, p=0.40)

        # Flexor Renshaw
        self._connect(f"m_f_{leg}", f"r_f_{leg}", weight=15.0, p=0.50)
        self._connect(f"r_f_{leg}", f"m_f_{leg}", weight=-20.0, p=0.50)
        self._connect(f"r_f_{leg}", f"ia_f_{leg}", weight=-12.0, p=0.40)

        # --- Ia Reciprocal Inhibition to Antagonist Motor Neurons ---
        self._connect(f"ia_e_{leg}", f"m_f_{leg}", weight=-25.0, p=0.40)
        self._connect(f"ia_f_{leg}", f"m_e_{leg}", weight=-25.0, p=0.40)

        # --- Ia Spindle Afferent Feedback (Stretch reflex & RG driving) ---
        # Extensor Ia
        self._connect(f"ia_e_in_{leg}", f"m_e_{leg}", weight=18.0, p=0.40)
        self._connect(f"ia_e_in_{leg}", f"ia_e_{leg}", weight=12.0, p=0.40)
        self._connect(f"ia_e_in_{leg}", f"rg_e_{leg}", weight=5.0, p=0.35,
                      is_stdp=True, key=f"ia_rg_e_{leg}")

        # Flexor Ia
        self._connect(f"ia_f_in_{leg}", f"m_f_{leg}", weight=18.0, p=0.40)
        self._connect(f"ia_f_in_{leg}", f"ia_f_{leg}", weight=12.0, p=0.40)
        self._connect(f"ia_f_in_{leg}", f"rg_f_{leg}", weight=5.0, p=0.35,
                      is_stdp=True, key=f"ia_rg_f_{leg}")

        # --- Ib Golgi Tendon Organ Feedback (Tension protection / autogenic inhibition) ---
        self._connect(f"ib_e_in_{leg}", f"m_e_{leg}", weight=-18.0, p=0.40)
        self._connect(f"ib_f_in_{leg}", f"m_f_{leg}", weight=-18.0, p=0.40)

        # --- Cutaneous (CUT) Feedback (Stance reinforcement) ---
        self._connect(f"cut_in_{leg}", f"rg_e_{leg}", weight=22.0, p=0.45, is_stdp=True, key=f"cut_rge_{leg}")
        self._connect(f"cut_in_{leg}", f"in_e_{leg}", weight=8.0, p=0.30)

        # --- Commissural Projections from RG-F ---
        self._connect(f"rg_f_{leg}", f"v2a_{leg}", weight=15.0, p=0.45)
        self._connect(f"v2a_{leg}", f"v0v_{leg}", weight=18.0, p=0.45)
        self._connect(f"rg_f_{leg}", f"v0d_{leg}", weight=12.0, p=0.40)
        self._connect(f"rg_f_{leg}", f"v3_f_{leg}", weight=15.0, p=0.40)

        # In1 inhibits ipsilateral RG-F
        self._connect(f"in1_{leg}", f"rg_f_{leg}", weight=-20.0, p=0.40)

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

    def get_muscle_drives(self, dt_ms: float) -> Tuple[float, float]:
        """
        Normalized muscle activation in [0, 1] from spikes during the step.
        A pool at `ref_hz` per neuron (default 100 Hz) maps to full drive, so
        the drive tracks population rate rather than saturating the instant the
        pool fires.
        """
        ev_e = self.rec_muse.get("events")["times"]
        ev_f = self.rec_musf.get("events")["times"]
        n_e = len(ev_e)
        n_f = len(ev_f)
        self.rec_muse.set({"n_events": 0})
        self.rec_musf.set({"n_events": 0})

        return self._drive(n_e, dt_ms), self._drive(n_f, dt_ms)

    def _drive(self, spikes: int, dt_ms: float) -> float:
        ref_hz = getattr(self, "ref_hz", 100.0)
        max_spikes = max(1.0, self.n_mn * ref_hz * (dt_ms / 1000.0))
        return float(np.clip(spikes / max_spikes, 0.0, 1.0))
