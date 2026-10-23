import os
from typing import Any, Tuple

import numpy as np

from persona.parameter import PersonaParam
from persona.population import Population
from persona.connection import Connection
from persona.registry import Registry
from t2xcpg.cfg import POPULATIONS_FINAL
from t2xcpg.utils import (
    get_jitter_delay, create_stdp_synapse, STDP_POSTFIX, stdp_param_spec,
)


# Ib -> antagonist rhythm generator. Excitatory: the detailed scheme draws the
# InIb-E/F -> RGF/RGE projection with an excitatory arrow (the "CPG
# phase-switch signal"), and the 2026-10-15 spec calls it "contralateral-phase
# RG exc". The earlier -18.0 came from W_IN2RG, a cpg_2legs_fast constant that
# is defined but never connected; the sign was the error, the magnitude is kept.
W_IB2RG = 18.0

# Renshaw recurrent inhibition is a structural stabiliser (scheme only, no
# reference). PERSONA otherwise drove r_f -> m_f from -4 to about -20 and
# silenced the flexor motor pool, so the native registry freeze pins each
# Renshaw weight's learned mean at its project value. Bounds and exploration
# are unchanged. The env toggle is for A/B runs: CPG_RENSHAW_CLAMP=0 reproduces
# the unconstrained baseline.
RENSHAW_PROJECTIONS = (
    ("m_e", "r_e", 3.0, 0.50),    # M-E -> R-E
    ("r_e", "m_e", -4.0, 0.50),   # R-E -| M-E
    ("r_e", "ia_e", -2.4, 0.40),  # R-E -| Ia-E
    ("m_f", "r_f", 3.0, 0.50),    # M-F -> R-F
    ("r_f", "m_f", -4.0, 0.50),   # R-F -| M-F
    ("r_f", "ia_f", -2.4, 0.40),  # R-F -| Ia-F
)

# The M-F Izhikevich parameters are frozen at the reference excitability. The
# PERSONA homeostatic update otherwise drove a/b/d away from reference and
# silenced the flexor pool (L musF 0.111 -> 0.008 over 20 s).
M_F_IZH_REF = {"a": 0.02, "b": 0.2, "c": -65.0, "d": 8.0}

# C1: the flexor motor pool's excitatory drive is seeded to compensate the
# reduced source populations. The port reference is W_RG2MN=2.75 and the
# homonymous Ia_F->MN weight 12.0. When n_rg/n_aff are below FINAL, the same
# per-synapse seed delivers proportionally less drive: the RG-F rate is a
# fraction of FINAL, PERSONA suppresses the R side further (rg_f->m_f
# 2.75 -> 1.09, Ia_F->m_f 12 -> 1.29), and musF_R collapses to 0. The seed is
# scaled by min(4, n_rg_FINAL/n_rg, n_aff_FINAL/n_aff), capped at 4x, and left
# optimizable. At FINAL the scale is 1.0 (the reference values). Measured 20 s
# at n_rg=10/n_aff=6 (4x): musF_L 0.093 (11.3% zero), musF_R 0.053 (19.1%
# zero); the pre-fix run was 0.166 / 0.002 (93.9% zero).
M_F_DRIVE_REF = {"rg_f": 2.75, "ia_f_in": 12.0}
M_F_DRIVE_MAX_SCALE = 4.0


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
                 ref_hz: float = 100.0,
                 cut_multisite: bool = False,
                 n_cut_sites: int = 8):
        self.leg = leg_name
        self.registry = registry
        self.n_rg = n_rg
        self.n_mn = n_mn
        self.n_int = n_int
        self.n_aff = n_aff
        # C1 size compensation: keep the flexor M-F seed at the reference
        # values at FINAL, and raise it (max 4x) when the RG-F/Ia_F source
        # populations are reduced. The two source sizes enter symmetrically.
        self.mf_drive_scale = min(
            M_F_DRIVE_MAX_SCALE,
            POPULATIONS_FINAL["n_rg"] / max(1, self.n_rg),
            POPULATIONS_FINAL["n_aff"] / max(1, self.n_aff),
        )
        self.mf_drive_seed = {
            key: value * self.mf_drive_scale
            for key, value in M_F_DRIVE_REF.items()
        }
        self.p_scale = float(p_scale)
        self.w_scale = float(w_scale)
        self.ref_hz = float(ref_hz)
        self.cut_multisite = bool(cut_multisite)
        self.n_cut_sites = int(n_cut_sites)
        if self.cut_multisite and self.n_cut_sites < 1:
            raise ValueError("n_cut_sites must be >= 1 for multi-site CUT")
        self.n_aff_site = max(1, self.n_aff // self.n_cut_sites)
        self.cut_site_pg: list[str] = []
        self.cut_site_in: list[str] = []
        self.plastic_projections: list[str] = []
        self.plastic_counts: dict[str, int] = {}

        # 1. Build Neuronal Populations
        self._build_populations()

        # 2. Build Synaptic Connections
        self._build_connections()

        # 3. Readout Recorders
        # Muscle drives are read from the MOTOR pool: the muscle parrots are a
        # convergent relay (each motor spike lands on ~n_mn*P_M2MUS of them), and
        # NEST collapses coincident relay input into one spike per neuron per ms,
        # so the relay count saturates at n_mn as soon as the pool fires - a
        # fan-in correction cannot recover a rate from it. Driving the plants from
        # that count produced a bang-bang command (measured: drive in {0.8, 1.0}
        # at n_mn=100) instead of the motor rate.
        self.rec_mne = self.registry.create_spike_recorder(self.m_e)
        self.rec_mnf = self.registry.create_spike_recorder(self.m_f)
        self.rec_muse = self.registry.create_spike_recorder(self.mus_e)
        self.rec_musf = self.registry.create_spike_recorder(self.mus_f)
        self.rec_rge = self.registry.create_spike_recorder(self.rg_e)
        self.rec_rgf = self.registry.create_spike_recorder(self.rg_f)

        self.registry.finalize_connections()

        # C2: the structural freezes go through the native PERSONA registry
        # (mean/sigma pins), not the old min=max/sigma=0 clamp. Freezing here,
        # after finalize_connections, anchors to the effective values captured
        # at the end of construction (including the M-F mutation above).
        self._freeze_structural_params()

    def _freeze_structural_params(self):
        """Pin the structural CPG sites through the native registry freeze."""
        if os.environ.get("CPG_RENSHAW_CLAMP", "1") != "0":
            keys = [f"conn_{src}_{self.leg}_{tgt}_{self.leg}_weight"
                    for src, tgt, _, _ in RENSHAW_PROJECTIONS]
            self.registry.freeze(keys, attrs=("mean",))
        if os.environ.get("CPG_MF_IZH_FREEZE", "1") != "0":
            keys = [f"pop_m_f_{self.leg}_{name}" for name in M_F_IZH_REF]
            self.registry.freeze(keys, attrs=("mean", "sigma"))

    def _param(self, name: str, mean: float, sigma: float = 0.1,
               min_val: float = -100.0, max_val: float = 100.0) -> PersonaParam:
        return PersonaParam(
            mean=mean,
            sigma=sigma,
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
                sigma=max(abs(mean) * 0.1, 1e-6),
                min_val=lo, max_val=hi)
        return out

    def _izhi(self, a, b, c, d):
        return {
            "a": self._param("a", a, 0.005, 0.001, 0.1),
            "b": self._param("b", b, 0.04, 0.05, 0.5),
            "c": self._param("c", c, 5.0, -80.0, -50.0),
            "d": self._param("d", d, 2.0, 0.0, 20.0),
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
        if os.environ.get("CPG_MF_IZH_FREEZE", "1") != "0":
            # Post-creation M-F mutation: seeded at the reference excitability
            # with no dispersion. `finalize_connections` recaptures these as
            # the freeze anchor; the native freeze then holds mean and sigma.
            mf_cfg = self.registry.population_cfgs[f"m_f_{self.leg}"]
            for name, value in M_F_IZH_REF.items():
                p = mf_cfg.params[name]
                p.mean = float(value)
                p.sigma = 0.0

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

        # Half-centre inhibitory interneurons (InE, InF): fast-spiking
        # (izh_inh_params in cpg_2legs_fast). With exc() the two half-centres
        # are co-active instead of alternating.
        self.in_e = self.registry.create_population(
            Population(model="izhikevich", n=self.n_int, params=inh(), name=f"in_e_{self.leg}")
        )
        self.in_f = self.registry.create_population(
            Population(model="izhikevich", n=self.n_int, params=inh(), name=f"in_f_{self.leg}")
        )

        # Commissural premotor interneurons (NEURON `memcpg/leg.py`): V2a/V0v/V3F
        # are excitatory relays, V0d and In1 are inhibitory. They replace the
        # direct RG<->RG commissural projection that `cpg_2legs_fast` abstracts
        # as "V0v/V0d/V2a/In1 classes".
        self.v2a = self.registry.create_population(
            Population(model="izhikevich", n=self.n_int, params=exc(), name=f"v2a_{self.leg}")
        )
        self.v0v = self.registry.create_population(
            Population(model="izhikevich", n=self.n_int, params=exc(), name=f"v0v_{self.leg}")
        )
        self.v3f = self.registry.create_population(
            Population(model="izhikevich", n=self.n_int, params=exc(), name=f"v3f_{self.leg}")
        )
        self.v0d = self.registry.create_population(
            Population(model="izhikevich", n=self.n_int, params=inh(), name=f"v0d_{self.leg}")
        )
        self.in1 = self.registry.create_population(
            Population(model="izhikevich", n=self.n_int, params=inh(), name=f"in1_{self.leg}")
        )

        # Muscles (Parrot readout)
        self.mus_e = self.registry.create_parrot_nodes(f"mus_e_{self.leg}", self.n_mn)
        self.mus_f = self.registry.create_parrot_nodes(f"mus_f_{self.leg}", self.n_mn)

        # Sensory Feedback Generators (Poisson -> Parrot) as registered relays;
        # each group's per-step rate delta goes to its producer recorder.
        # 1. CUT (cutaneous)
        # Rest rate is 0, so its current value is not a usable scale: declare
        # 10 Hz explicitly — the same full-scale rate as the Ia groups (spec §5.3).
        # With Option B the single relay is a one-neuron noise tap only: the
        # afferent spike trains are the per-site relays below, and the tap keeps
        # the `cut_L/R` producer feature (the aggregate rate) without doubling
        # the CUT Poisson population.
        cut_relay_n = 1 if self.cut_multisite else self.n_aff
        self.cut_pg, self.cut_in = self.registry.create_relay(
            f"cut_pg_{self.leg}", f"cut_in_{self.leg}", cut_relay_n,
            rate=0.0, noise_name=f"cut_{self.leg}", noise_scale=10.0)
        if self.cut_multisite:
            for k in range(self.n_cut_sites):
                pg_key = f"cut_pg_{self.leg}_s{k}"
                in_key = f"cut_in_{self.leg}_s{k}"
                self.registry.create_relay(
                    pg_key, in_key, self.n_aff_site,
                    rate=0.0, noise_name=None)
                self.cut_site_pg.append(pg_key)
                self.cut_site_in.append(in_key)

        # 2. Ia Afferents (stretch length & velocity)
        self.ia_e_pg, self.ia_e_in = self.registry.create_relay(
            f"ia_e_pg_{self.leg}", f"ia_e_in_{self.leg}", self.n_aff,
            rate=10.0, noise_name=f"ia_e_{self.leg}")
        self.ia_f_pg, self.ia_f_in = self.registry.create_relay(
            f"ia_f_pg_{self.leg}", f"ia_f_in_{self.leg}", self.n_aff,
            rate=10.0, noise_name=f"ia_f_{self.leg}")

        # 3. Ib Afferents (strain force & activation)
        self.ib_e_pg, self.ib_e_in = self.registry.create_relay(
            f"ib_e_pg_{self.leg}", f"ib_e_in_{self.leg}", self.n_aff,
            rate=5.0, noise_name=f"ib_e_{self.leg}")
        self.ib_f_pg, self.ib_f_in = self.registry.create_relay(
            f"ib_f_pg_{self.leg}", f"ib_f_in_{self.leg}", self.n_aff,
            rate=5.0, noise_name=f"ib_f_{self.leg}")

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
            model, wr = create_stdp_synapse(self.registry, syn_type, f"{src}_{tgt}")
            # STDP owns the synaptic weight; PERSONA optimizes only the STDP
            # hyper-parameters (tau_plus/lambda/alpha/mu/Wmax). The seed weight
            # is written once via the syn_spec and never overwritten afterwards.
            parameters = self._stdp_params(syn_type, weight)
            extra = {"delay": float(d), "weight": float(weight)}
        else:
            parameters = {"weight": self._param(
                "weight", weight, abs(weight) * 0.1,
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
            extra_syn_spec=extra,
            key=key,
            weight_recorder=wr
        )
        if key is not None and key not in self.plastic_projections:
            self.plastic_projections.append(key)
        return self.registry.create_connection(cfg)

    def update_plastic_weights(self, t_before: float, t_now: float):
        """
        Samples STDP weight changes from the weight recorders (the registry
        computes each projection's per-synapse mean increment and records it in
        the matching producer recorder). Returns the per-projection deltas and
        stores the per-projection update counts for logging.
        """
        deltas = self.registry.sample_plastic_weight_deltas(t_before, t_now)
        counts = self.registry.last_plastic_counts()
        self.plastic_counts = {}
        for key in self.plastic_projections:
            self.plastic_counts[key] = int(counts.get(key, 0))
        return deltas

    def _build_connections(self):
        leg = self.leg

        # --- Rhythm-generator half-centre THROUGH the InE/InF relays
        #     (cpg_2legs_fast): RGE->InE (exc), InE-|RGF, RGF->InF (exc), InF-|RGE.
        self._connect(f"rg_e_{leg}", f"in_e_{leg}", weight=12.0, p=0.15)   # W_RG2INE, P_RG_RECIP_E
        self._connect(f"in_e_{leg}", f"rg_f_{leg}", weight=-8.0, p=0.15)   # W_INE2RGF, P_RG_RECIP_E
        self._connect(f"rg_f_{leg}", f"in_f_{leg}", weight=18.0, p=0.30)   # W_RG2INF, P_RG_RECIP_F
        self._connect(f"in_f_{leg}", f"rg_e_{leg}", weight=-48.0, p=0.30)  # W_INF2RGE

        # --- RG self-recurrence ---
        self._connect(f"rg_e_{leg}", f"rg_e_{leg}", weight=4.0, p=0.12)    # W_RG_REC_E, P_RG_REC
        self._connect(f"rg_f_{leg}", f"rg_f_{leg}", weight=5.5, p=0.12)    # W_RG_REC_F

        # --- Motor output: RG -> MN direct; MN -> muscle ---
        self._connect(f"rg_e_{leg}", f"m_e_{leg}", weight=2.75, p=0.50)    # W_RG2MN
        self._connect(f"rg_f_{leg}", f"m_f_{leg}",
                      weight=self.mf_drive_seed["rg_f"], p=0.50)  # C1 seed
        self._connect(f"m_e_{leg}", f"mus_e_{leg}", weight=1.0, p=0.80)    # W_M2MUS, P_M2MUS
        self._connect(f"m_f_{leg}", f"mus_f_{leg}", weight=1.0, p=0.80)

        # --- Ia (spindle): monosynaptic to homonymous MN/RG + Ia inhibitory IN ---
        self._connect(f"ia_e_in_{leg}", f"m_e_{leg}", weight=12.0, p=0.40)  # Ia(top) -> M-E (exc)
        self._connect(f"ia_f_in_{leg}", f"m_f_{leg}",
                      weight=self.mf_drive_seed["ia_f_in"], p=0.40)  # Ia(bottom) -> M-F (exc), C1 seed
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
        # Renshaw recurrent inhibition, attenuated to 0.2x. This loop is net
        # *positive* feedback here: M excites R, R inhibits the motor pool, but
        # R also inhibits the Ia interneuron which inhibits the motor pool, so R
        # disinhibits M. At full gain it held the motor pools at the 0.2 ms step
        # cap (2418 Hz while RG-F ran at 79 Hz); at 0.2x the motor pools track RG
        # (`test_results/renshaw_fix.py`). PERSONA must not drive the recurrent
        # limb to -20: that silences M-F, so the native freeze pins the learned
        # mean at the project value. CPG_RENSHAW_CLAMP=0 is the A/B baseline.
        for src, tgt, weight, p in RENSHAW_PROJECTIONS:
            self._connect(f"{src}_{leg}", f"{tgt}_{leg}", weight=weight, p=p)

        # --- Ib Golgi tendon organ, disynaptic via Ib interneurons (project values) ---
        # Extensor Ib: GTO -> InIb-E -> {M-E inh, M-F exc, RGF inh}
        self._connect(f"ib_e_in_{leg}", f"in_ib_e_{leg}", weight=16.0, p=0.40)   # GTO -> InIb-E
        self._connect(f"in_ib_e_{leg}", f"m_e_{leg}", weight=-18.0, p=0.40)      # autogenic inhibition
        self._connect(f"in_ib_e_{leg}", f"m_f_{leg}", weight=12.0, p=0.35)       # disinhibit flexor
        self._connect(f"in_ib_e_{leg}", f"rg_f_{leg}", weight=W_IB2RG, p=0.35)  # Ib-E excites RG-F (phase switch)
        # Flexor Ib: GTO -> InIb-F -> {M-F inh, M-E exc, RGE inh}
        self._connect(f"ib_f_in_{leg}", f"in_ib_f_{leg}", weight=16.0, p=0.40)   # GTO -> InIb-F
        self._connect(f"in_ib_f_{leg}", f"m_f_{leg}", weight=-18.0, p=0.40)      # autogenic inhibition
        self._connect(f"in_ib_f_{leg}", f"m_e_{leg}", weight=12.0, p=0.35)       # disinhibit extensor
        self._connect(f"in_ib_f_{leg}", f"rg_e_{leg}", weight=W_IB2RG, p=0.35)  # Ib-F excites RG-E (phase switch)

        # --- Ib interneuron convergence (spindle, CPG) ---
        self._connect(f"ia_e_in_{leg}", f"in_ib_e_{leg}", weight=8.0, p=0.30)    # Ia -> InIb-E
        self._connect(f"ia_f_in_{leg}", f"in_ib_f_{leg}", weight=8.0, p=0.30)    # Ia -> InIb-F

        # --- Cutaneous (CUT) stance reinforcement ---
        if self.cut_multisite:
            # Option B: each plantar site has its own Poisson relay and its own
            # STDP projection. The per-site weights keep the same total CUT
            # convergence as the single aggregate projection: n_aff_site * 8
            # is ~n_aff, so p and the seed weights are unchanged.
            for k in range(self.n_cut_sites):
                src = f"cut_in_{leg}_s{k}"
                self._connect(src, f"rg_e_{leg}", weight=14.0, p=0.35,
                              is_stdp=True, key=f"cut_rge_{leg}_s{k}")
                self._connect(src, f"in_e_{leg}", weight=6.0, p=0.30)
        else:
            self._connect(f"cut_in_{leg}", f"rg_e_{leg}", weight=14.0, p=0.35,
                          is_stdp=True, key=f"cut_rge_{leg}")   # W_CUT2RGE, P_CUT2RGE_STATIC
            self._connect(f"cut_in_{leg}", f"in_e_{leg}", weight=6.0, p=0.30)   # W_CUT2INE, P_CUT2INE

        # --- Commissural V-cell relay, ipsilateral half (NEURON memcpg/leg.py) ---
        # RG_F drives the three V-cell classes; V2a feeds V0v, whose
        # contralateral projection is built in L2CPG. In1 is the inhibitory
        # relay onto the ipsilateral RG_F. Weights and delays are memcpg's; p is
        # the port's int->int density.
        self._connect(f"rg_f_{leg}", f"v2a_{leg}", weight=0.75, p=0.50, delay=3.0)
        self._connect(f"rg_f_{leg}", f"v0d_{leg}", weight=0.75, p=0.50, delay=3.0)
        self._connect(f"rg_f_{leg}", f"v3f_{leg}", weight=1.5, p=0.50, delay=3.0)
        self._connect(f"v2a_{leg}", f"v0v_{leg}", weight=1.2, p=0.50, delay=3.0)
        self._connect(f"in1_{leg}", f"rg_f_{leg}", weight=-0.5, p=0.50, delay=1.0)

    def update_sensory_feedback(self, cut_hz: float,
                                ia_e_hz: float, ia_f_hz: float,
                                ib_e_hz: float, ib_f_hz: float,
                                current_time_ms: float = 0.0,
                                cut_sites=None, cut_active=None):
        """
        Updates Poisson generator rates based on live MuJoCo sensory inputs.
        Goes through the registry, which records each group's per-step rate
        delta into its producer recorder (read-and-clear upstream).

        With `cut_multisite`, `cut_sites` and `cut_active` carry one value per
        plantar site. Empty tuples (the initial `CPGInput()` before the first
        MuJoCo step) mean all sites are silent. An inactive site's generator is
        set to 0 Hz, so it cannot emit spikes or drive its STDP projection; the
        aggregate `cut_hz` still feeds the `cut_L/R` noise tap.
        """
        side = self.leg
        self.registry.set_relay_rate(f"cut_pg_{side}", cut_hz)
        self.registry.set_relay_rate(f"ia_e_pg_{side}", ia_e_hz)
        self.registry.set_relay_rate(f"ia_f_pg_{side}", ia_f_hz)
        self.registry.set_relay_rate(f"ib_e_pg_{side}", ib_e_hz)
        self.registry.set_relay_rate(f"ib_f_pg_{side}", ib_f_hz)

        if self.cut_multisite:
            n_sites = len(self.cut_site_pg)
            cut_sites = () if cut_sites is None else cut_sites
            cut_active = () if cut_active is None else cut_active
            if len(cut_sites) == 0 and len(cut_active) == 0:
                cut_sites = (0.0,) * n_sites
                cut_active = (False,) * n_sites
            elif len(cut_sites) != n_sites or len(cut_active) != n_sites:
                raise ValueError(
                    f"cut_multisite expects {n_sites} per-site rates and "
                    f"gates, got {len(cut_sites)} and {len(cut_active)}")
            for pg, rate, active in zip(self.cut_site_pg, cut_sites,
                                        cut_active):
                self.registry.set_relay_rate(pg, float(rate) if active else 0.0)

    def get_muscle_drives(self, dt_ms: float, current_time_ms: float = 0.0) -> Tuple[float, float]:
        """
        Normalized muscle activation in [0, 1] from the motor pool's spikes this
        step. A pool at `ref_hz` per neuron (default 100 Hz) maps to full drive, so
        the drive tracks population rate and stays graded over its whole range.
        Muscle drives are readouts/logs, not noise channels.
        """
        n_e = len(self.rec_mne.get("events")["times"])
        n_f = len(self.rec_mnf.get("events")["times"])
        self.rec_mne.set({"n_events": 0})
        self.rec_mnf.set({"n_events": 0})

        drive_e = self._drive(n_e, dt_ms)
        drive_f = self._drive(n_f, dt_ms)
        return drive_e, drive_f

    def _drive(self, spikes: int, dt_ms: float, n: int | None = None) -> float:
        ref_hz = getattr(self, "ref_hz", 100.0)
        count = self.n_mn if n is None else int(n)
        max_spikes = max(1.0, count * ref_hz * (dt_ms / 1000.0))
        return float(np.clip(spikes / max_spikes, 0.0, 1.0))

    def get_rg_rates(self, dt_ms: float,
                     current_time_ms: float = 0.0) -> Tuple[float, float]:
        """
        Normalized RGE/RGF population activity in [0, 1] from the spikes the
        rhythm-generator recorders saw this step. The recorders are read then
        cleared (`cpg_2legs_fast`'s constant-cost pattern), so both the cost and
        the memory stay bounded over a long run.
        """
        n_e = len(self.rec_rge.get("events")["times"])
        n_f = len(self.rec_rgf.get("events")["times"])
        self.rec_rge.set({"n_events": 0})
        self.rec_rgf.set({"n_events": 0})
        return (self._drive(n_e, dt_ms, self.n_rg),
                self._drive(n_f, dt_ms, self.n_rg))
