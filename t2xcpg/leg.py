import os
from typing import Any, Tuple

import numpy as np

from persona.parameter import PersonaParam
from persona.population import Population
from persona.connection import Connection
from persona.registry import Registry
from t2xcpg.cfg import POPULATIONS_FINAL
from t2xcpg.params import PARAMS
from t2xcpg.utils import (
    delay_param, get_jitter_delay, create_stdp_synapse, STDP_POSTFIX,
    stdp_param_spec,
)


# Ib -> antagonist rhythm generator (initial value in params.py:
# w_con_in_ib_e_rg_f / w_con_in_ib_f_rg_e). Excitatory: the detailed scheme
# draws the InIb-E/F -> RGF/RGE projection with an excitatory arrow (the
# "CPG phase-switch signal"), and the 2026-10-15 spec calls it
# "contralateral-phase RG exc". The earlier -18.0 came from W_IN2RG, a
# cpg_2legs_fast constant that is defined but never connected; the sign was
# the error, the magnitude is kept.

# Temporary CPG-only tuning scales (defaults preserve the current model).
SCALE_IA = float(os.environ.get("CPG_SCALE_IA", str(PARAMS["scale_ia"])))
SCALE_IB = float(os.environ.get("CPG_SCALE_IB", str(PARAMS["scale_ib"])))
SCALE_CUT = float(os.environ.get("CPG_SCALE_CUT", str(PARAMS["scale_cut"])))
SCALE_INHIB = float(os.environ.get("CPG_SCALE_INHIB", str(PARAMS["scale_inhib"])))

# Renshaw recurrent inhibition is a structural stabiliser (scheme only, no
# reference). PERSONA otherwise drove r_f -> m_f from -4 to about -20 and
# silenced the flexor motor pool, so the native registry freeze pins each
# Renshaw weight's learned mean at its project value. Bounds and exploration
# are unchanged. The env toggle is for A/B runs: CPG_RENSHAW_CLAMP=0 reproduces
# the unconstrained baseline.
_RENSHAW_PAIRS = (
    ("m_e", "r_e"),   # M-E -> R-E
    ("r_e", "m_e"),   # R-E -l M-E
    ("r_e", "ia_e"),  # R-E -l Ia-E
    ("m_f", "r_f"),   # M-F -> R-F
    ("r_f", "m_f"),   # R-F -l M-F
    ("r_f", "ia_f"),  # R-F -l Ia-F
)
RENSHAW_PROJECTIONS = tuple(
    (src, tgt, PARAMS[f"w_con_{src}_{tgt}"], PARAMS[f"p_con_{src}_{tgt}"])
    for src, tgt in _RENSHAW_PAIRS
)

# The M-F Izhikevich parameters are frozen at the reference excitability. The
# PERSONA homeostatic update otherwise drove a/b/d away from reference and
# silenced the flexor pool (L musF 0.111 -> 0.008 over 20 s).
M_F_IZH_REF = {name: PARAMS[f"izh_mf_{name}"] for name in ("a", "b", "c", "d")}

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
M_F_DRIVE_REF = {
    "rg_f": PARAMS["w_con_rg_f_m_f"],
    "ia_f_in": PARAMS["w_con_ia_f_in_m_f"],
}
# CUT parameters from cpg_2legs_fast (single-site CUT model).
W_CUT2RGE = PARAMS["w_stdp_cut_rge"]
P_CUT2RGE = PARAMS["p_con_cut_rge"]
W_CUT2INE = PARAMS["w_con_cut_in_e"]
P_CUT2INE = PARAMS["p_con_cut_in_e"]
M_F_DRIVE_MAX_SCALE = PARAMS["mf_drive_max_scale"]


class Leg:
    """One hindlimb CPG: rhythm generators, half-centre interneurons, motor
    pools, Renshaw cells and Ia/Ib/CUT afferent pathways.

    Every neuronal and synaptic parameter is registered with the PERSONA
    Registry so the agent can personalize it.
    """
    def __init__(self,
                 leg_name: str,
                 registry: Registry,
                 n_rg: int = PARAMS["n_rg_default"],
                 n_mn: int = PARAMS["n_mn_default"],
                 n_int: int = PARAMS["n_int_default"],
                 n_aff: int = PARAMS["n_aff_default"],
                 p_scale: float = PARAMS["p_scale_default"],
                 w_scale: float = PARAMS["w_scale_default"],
                 ref_hz: float = PARAMS["ref_hz_default"],
                 cut_multisite: bool = PARAMS["cut_multisite_default"],
                 n_cut_sites: int = PARAMS["n_cut_sites_default"]):
        self.leg = leg_name
        self.registry = registry
        self.registry.ensure_delay_range(
            PARAMS["delay_min_ms"], PARAMS["delay_max_ms"])
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
        self.cut_site_nodes: list[Any] = []
        self.plastic_projections: list[str] = []
        self.plastic_counts: dict[str, int] = {}

        # 1. Build Neuronal Populations
        self._build_populations()

        # 2. Build Synaptic Connections
        self._build_connections()

        # 3. Spike recorders for every population this leg owns.
        self.recorders = {}
        self.rec_ids = {}
        pools = (
            ("rg_e", self.rg_e), ("rg_f", self.rg_f),
            ("m_e", self.m_e), ("m_f", self.m_f),
            ("r_e", self.r_e), ("r_f", self.r_f),
            ("ia_e", self.ia_e), ("ia_f", self.ia_f),
            ("in_ib_e", self.in_ib_e), ("in_ib_f", self.in_ib_f),
            ("in_e", self.in_e), ("in_f", self.in_f),
            ("v2a", self.v2a), ("v0v", self.v0v), ("v3f", self.v3f),
            ("v0d", self.v0d), ("in1", self.in1),
            ("mus_e", self.mus_e), ("mus_f", self.mus_f),
            ("cut_in", self.cut_in),
            ("ia_e_in", self.ia_e_in), ("ia_f_in", self.ia_f_in),
            ("ib_e_in", self.ib_e_in), ("ib_f_in", self.ib_f_in),
        )
        for base, nodes in pools:
            name = f"{base}_{self.leg}"
            self.recorders[name] = self.registry.create_spike_recorder(nodes)
            self.rec_ids[name] = np.asarray(nodes.tolist(), dtype=np.int64)
        for k, nodes in enumerate(self.cut_site_nodes):
            name = f"cut_in_{self.leg}_s{k}"
            self.recorders[name] = self.registry.create_spike_recorder(nodes)
            self.rec_ids[name] = np.asarray(nodes.tolist(), dtype=np.int64)
        self.rec_mne = self.recorders[f"m_e_{self.leg}"]
        self.rec_mnf = self.recorders[f"m_f_{self.leg}"]
        self.rec_muse = self.recorders[f"mus_e_{self.leg}"]
        self.rec_musf = self.recorders[f"mus_f_{self.leg}"]
        self.rec_rge = self.recorders[f"rg_e_{self.leg}"]
        self.rec_rgf = self.recorders[f"rg_f_{self.leg}"]
        self._last_counts = {}
        self._last_rates = {}
        self._last_events = {}

        self.registry.finalize_connections()

        # C2: structural freezes use the native PERSONA registry pins. Freezing
        # after finalize_connections anchors the effective construction values
        # (including the M-F mutation above).
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

    def _param(self, name: str, mean: float,
               sigma: float = PARAMS["syn_weight_sigma_fraction"],
               min_val: float = PARAMS["syn_weight_min"],
               max_val: float = PARAMS["syn_weight_max"]) -> PersonaParam:
        return PersonaParam(
            mean=mean,
            sigma=sigma,
            min=min_val,
            max=max_val
        )

    _STDP_RANGES = {
        "tau_plus": (PARAMS["stdp_range_tau_plus_min"],
                     PARAMS["stdp_range_tau_plus_max"]),
        "lambda": (PARAMS["stdp_range_lambda_min"],
                   PARAMS["stdp_range_lambda_max"]),
        "alpha": (PARAMS["stdp_range_alpha_min"],
                  PARAMS["stdp_range_alpha_max"]),
        "mu_plus": (PARAMS["stdp_range_mu_plus_min"],
                    PARAMS["stdp_range_mu_plus_max"]),
        "mu_minus": (PARAMS["stdp_range_mu_minus_min"],
                     PARAMS["stdp_range_mu_minus_max"]),
        "Wmax": (PARAMS["stdp_range_wmax_min"],
                 PARAMS["stdp_range_wmax_max"]),
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
                mean = max(mean, abs(weight) * PARAMS["stdp_wmax_seed_factor"])
                lo = max(lo, abs(weight))
            out[name] = self._param(
                name, mean,
                sigma=max(abs(mean) * PARAMS["stdp_sigma_fraction"],
                          PARAMS["stdp_sigma_floor"]),
                min_val=lo, max_val=hi)
        return out

    def _izhi(self, a, b, c, d):
        return {
            "a": self._param("a", a, PARAMS["izh_sigma_a"],
                             PARAMS["izh_min_a"], PARAMS["izh_max_a"]),
            "b": self._param("b", b, PARAMS["izh_sigma_b"],
                             PARAMS["izh_min_b"], PARAMS["izh_max_b"]),
            "c": self._param("c", c, PARAMS["izh_sigma_c"],
                             PARAMS["izh_min_c"], PARAMS["izh_max_c"]),
            "d": self._param("d", d, PARAMS["izh_sigma_d"],
                             PARAMS["izh_min_d"], PARAMS["izh_max_d"]),
        }

    def _build_populations(self):
        # Izhikevich parameters from cpg_2legs_fast: regular-spiking excitatory,
        # fast-spiking inhibitory, and intrinsically-bursting flexor RG.
        def exc():
            return self._izhi(PARAMS["izh_exc_a"], PARAMS["izh_exc_b"],
                              PARAMS["izh_exc_c"], PARAMS["izh_exc_d"])

        def inh():
            return self._izhi(PARAMS["izh_inh_a"], PARAMS["izh_inh_b"],
                              PARAMS["izh_inh_c"], PARAMS["izh_inh_d"])

        def rgf():
            return self._izhi(PARAMS["izh_rgf_a"], PARAMS["izh_rgf_b"],
                              PARAMS["izh_rgf_c"], PARAMS["izh_rgf_d"])

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
        cut_relay_n = (PARAMS["n_cut_noise_tap"] if self.cut_multisite
                       else self.n_aff)
        self.cut_pg, self.cut_in = self.registry.create_relay(
            f"cut_pg_{self.leg}", f"cut_in_{self.leg}", cut_relay_n,
            rate=PARAMS["hz_cut_rest"], noise_name=f"cut_{self.leg}",
            noise_scale=PARAMS["hz_cut_noise_scale"])
        if self.cut_multisite:
            for k in range(self.n_cut_sites):
                pg_key = f"cut_pg_{self.leg}_s{k}"
                in_key = f"cut_in_{self.leg}_s{k}"
                _pg, in_nodes = self.registry.create_relay(
                    pg_key, in_key, self.n_aff_site,
                    rate=PARAMS["hz_cut_site_rest"], noise_name=None)
                self.cut_site_pg.append(pg_key)
                self.cut_site_in.append(in_key)
                self.cut_site_nodes.append(in_nodes)

        # 2. Ia Afferents (stretch length & velocity)
        self.ia_e_pg, self.ia_e_in = self.registry.create_relay(
            f"ia_e_pg_{self.leg}", f"ia_e_in_{self.leg}", self.n_aff,
            rate=PARAMS["hz_ia"], noise_name=f"ia_e_{self.leg}")
        self.ia_f_pg, self.ia_f_in = self.registry.create_relay(
            f"ia_f_pg_{self.leg}", f"ia_f_in_{self.leg}", self.n_aff,
            rate=PARAMS["hz_ia"], noise_name=f"ia_f_{self.leg}")

        # 3. Ib Afferents (strain force & activation)
        self.ib_e_pg, self.ib_e_in = self.registry.create_relay(
            f"ib_e_pg_{self.leg}", f"ib_e_in_{self.leg}", self.n_aff,
            rate=PARAMS["hz_ib"], noise_name=f"ib_e_{self.leg}")
        self.ib_f_pg, self.ib_f_in = self.registry.create_relay(
            f"ib_f_pg_{self.leg}", f"ib_f_in_{self.leg}", self.n_aff,
            rate=PARAMS["hz_ib"], noise_name=f"ib_f_{self.leg}")

    def _connect(self, src: str, tgt: str, weight: float,
                 p: float = PARAMS["leg_p_default"],
                 is_stdp: bool = False, delay: float | None = None,
                 key: str | None = None) -> Any:
        d = get_jitter_delay() if delay is None else delay
        p = p * self.p_scale
        weight = weight * self.w_scale
        delay_parameter = {"delay": delay_param(float(d))}

        model = "static_synapse"
        wr = None
        if is_stdp:
            syn_base = key.rsplit("_", 1)[0] if key else "default"
            syn_type = STDP_POSTFIX.get(syn_base, "default")
            model, wr = create_stdp_synapse(self.registry, syn_type, f"{src}_{tgt}")
            parameters = {**self._stdp_params(syn_type, weight), **delay_parameter}
            extra = {"weight": float(weight)}
        else:
            parameters = {"weight": self._param(
                "weight", weight,
                abs(weight) * PARAMS["syn_weight_sigma_fraction"],
                min_val=PARAMS["syn_weight_min"] if weight < 0 else 0.0,
                max_val=0.0 if weight < 0 else PARAMS["syn_weight_max"]),
                **delay_parameter}
            extra = {}

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
        self._connect(f"rg_e_{leg}", f"in_e_{leg}",
                      weight=PARAMS["w_con_rg_e_in_e"],
                      p=PARAMS["p_con_rg_e_in_e"])
        self._connect(f"in_e_{leg}", f"rg_f_{leg}",
                      weight=PARAMS["w_con_in_e_rg_f"] * SCALE_INHIB,
                      p=PARAMS["p_con_in_e_rg_f"])
        self._connect(f"rg_f_{leg}", f"in_f_{leg}",
                      weight=PARAMS["w_con_rg_f_in_f"],
                      p=PARAMS["p_con_rg_f_in_f"])
        self._connect(f"in_f_{leg}", f"rg_e_{leg}",
                      weight=PARAMS["w_con_in_f_rg_e"] * SCALE_INHIB,
                      p=PARAMS["p_con_in_f_rg_e"])

        # --- RG self-recurrence ---
        self._connect(f"rg_e_{leg}", f"rg_e_{leg}",
                      weight=PARAMS["w_con_rg_e_rg_e"],
                      p=PARAMS["p_con_rg_e_rg_e"])
        self._connect(f"rg_f_{leg}", f"rg_f_{leg}",
                      weight=PARAMS["w_con_rg_f_rg_f"],
                      p=PARAMS["p_con_rg_f_rg_f"])

        # --- Motor output: RG -> MN direct; MN -> muscle ---
        self._connect(f"rg_e_{leg}", f"m_e_{leg}",
                      weight=PARAMS["w_con_rg_e_m_e"],
                      p=PARAMS["p_con_rg_e_m_e"])
        self._connect(f"rg_f_{leg}", f"m_f_{leg}",
                      weight=self.mf_drive_seed["rg_f"],
                      p=PARAMS["p_con_rg_f_m_f"])  # C1 seed
        self._connect(f"m_e_{leg}", f"mus_e_{leg}",
                      weight=PARAMS["w_con_m_e_mus_e"],
                      p=PARAMS["p_con_m_e_mus_e"])
        self._connect(f"m_f_{leg}", f"mus_f_{leg}",
                      weight=PARAMS["w_con_m_f_mus_f"],
                      p=PARAMS["p_con_m_f_mus_f"])

        # --- Ia (spindle): monosynaptic to homonymous MN/RG + Ia inhibitory IN ---
        self._connect(f"ia_e_in_{leg}", f"m_e_{leg}",
                      weight=PARAMS["w_con_ia_e_in_m_e"] * SCALE_IA,
                      p=PARAMS["p_con_ia_e_in_m_e"])  # Ia(top) -> M-E (exc)
        self._connect(f"ia_f_in_{leg}", f"m_f_{leg}",
                      weight=self.mf_drive_seed["ia_f_in"] * SCALE_IA,
                      p=PARAMS["p_con_ia_f_in_m_f"])  # Ia(bottom) -> M-F (exc), C1 seed
        # MOD_IA_RG_STDP: homonymous Ia->RG is plastic (spec D12) — Ia-E potentiates
        # onto RG-E, Ia-F onto RG-F; muscle afferents become a learning drive.
        self._connect(f"ia_e_in_{leg}", f"rg_e_{leg}",
                      weight=PARAMS["w_stdp_ia_e_in_rg_e"] * SCALE_IA,
                      p=PARAMS["p_con_ia_e_in_rg_e"],
                      is_stdp=True, key=f"ia_rge_{leg}")
        self._connect(f"ia_f_in_{leg}", f"rg_f_{leg}",
                      weight=PARAMS["w_stdp_ia_f_in_rg_f"] * SCALE_IA,
                      p=PARAMS["p_con_ia_f_in_rg_f"],
                      is_stdp=True, key=f"ia_rgf_{leg}")
        self._connect(f"ia_e_in_{leg}", f"ia_e_{leg}",
                      weight=PARAMS["w_con_ia_e_in_ia_e"],
                      p=PARAMS["p_con_ia_e_in_ia_e"])  # W_IA_IN2INT
        self._connect(f"ia_f_in_{leg}", f"ia_f_{leg}",
                      weight=PARAMS["w_con_ia_f_in_ia_f"],
                      p=PARAMS["p_con_ia_f_in_ia_f"])
        self._connect(f"ia_e_{leg}", f"m_f_{leg}",
                      weight=PARAMS["w_con_ia_e_m_f"],
                      p=PARAMS["p_con_ia_e_m_f"])  # W_IA_INT2ANT
        self._connect(f"ia_f_{leg}", f"m_e_{leg}",
                      weight=PARAMS["w_con_ia_f_m_e"],
                      p=PARAMS["p_con_ia_f_m_e"])

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
        self._connect(f"ib_e_in_{leg}", f"in_ib_e_{leg}",
                      weight=PARAMS["w_con_ib_e_in_in_ib_e"],
                      p=PARAMS["p_con_ib_e_in_in_ib_e"])  # GTO -> InIb-E
        self._connect(f"in_ib_e_{leg}", f"m_e_{leg}",
                      weight=PARAMS["w_con_in_ib_e_m_e"],
                      p=PARAMS["p_con_in_ib_e_m_e"])  # autogenic inhibition
        self._connect(f"in_ib_e_{leg}", f"m_f_{leg}",
                      weight=PARAMS["w_con_in_ib_e_m_f"] * SCALE_IB,
                      p=PARAMS["p_con_in_ib_e_m_f"])  # disinhibit flexor
        self._connect(f"in_ib_e_{leg}", f"rg_f_{leg}",
                      weight=PARAMS["w_con_in_ib_e_rg_f"] * SCALE_IB,
                      p=PARAMS["p_con_in_ib_e_rg_f"])  # Ib-E excites RG-F (phase switch)
        # Flexor Ib: GTO -> InIb-F -> {M-F inh, M-E exc, RGE inh}
        self._connect(f"ib_f_in_{leg}", f"in_ib_f_{leg}",
                      weight=PARAMS["w_con_ib_f_in_in_ib_f"],
                      p=PARAMS["p_con_ib_f_in_in_ib_f"])  # GTO -> InIb-F
        self._connect(f"in_ib_f_{leg}", f"m_f_{leg}",
                      weight=PARAMS["w_con_in_ib_f_m_f"],
                      p=PARAMS["p_con_in_ib_f_m_f"])  # autogenic inhibition
        self._connect(f"in_ib_f_{leg}", f"m_e_{leg}",
                      weight=PARAMS["w_con_in_ib_f_m_e"] * SCALE_IB,
                      p=PARAMS["p_con_in_ib_f_m_e"])  # disinhibit extensor
        self._connect(f"in_ib_f_{leg}", f"rg_e_{leg}",
                      weight=PARAMS["w_con_in_ib_f_rg_e"] * SCALE_IB,
                      p=PARAMS["p_con_in_ib_f_rg_e"])  # Ib-F excites RG-E (phase switch)

        # --- Cutaneous (CUT) stance reinforcement ---
        if self.cut_multisite:
            # Option B: each plantar site has its own Poisson relay and its own
            # STDP projection. The per-site weights keep the same total CUT
            # convergence as the single aggregate projection: n_aff_site * 8
            # is ~n_aff, so p and the seed weights are unchanged.
            for k in range(self.n_cut_sites):
                src = f"cut_in_{leg}_s{k}"
                self._connect(src, f"rg_e_{leg}", weight=W_CUT2RGE * SCALE_CUT,
                              p=P_CUT2RGE,
                              is_stdp=True, key=f"cut_rge_{leg}_s{k}")
                self._connect(src, f"in_e_{leg}", weight=W_CUT2INE, p=P_CUT2INE)
        else:
            self._connect(f"cut_in_{leg}", f"rg_e_{leg}", weight=W_CUT2RGE * SCALE_CUT, p=P_CUT2RGE,
                          is_stdp=True, key=f"cut_rge_{leg}")   # W_CUT2RGE, P_CUT2RGE_STATIC
            self._connect(f"cut_in_{leg}", f"in_e_{leg}", weight=W_CUT2INE, p=P_CUT2INE)   # W_CUT2INE, P_CUT2INE

        # --- Commissural V-cell relay, ipsilateral half (NEURON memcpg/leg.py) ---
        # RG_F drives the three V-cell classes; V2a feeds V0v, whose
        # contralateral projection is built in L2CPG. In1 is the inhibitory
        # relay onto the ipsilateral RG_F. Weights and delays are memcpg's; p is
        # the port's int->int density.
        self._connect(f"rg_f_{leg}", f"v2a_{leg}",
                      weight=PARAMS["w_con_rg_f_v2a"],
                      p=PARAMS["p_con_rg_f_v2a"], delay=PARAMS["delay_vcell"])
        self._connect(f"rg_f_{leg}", f"v0d_{leg}",
                      weight=PARAMS["w_con_rg_f_v0d"],
                      p=PARAMS["p_con_rg_f_v0d"], delay=PARAMS["delay_vcell"])
        self._connect(f"rg_f_{leg}", f"v3f_{leg}",
                      weight=PARAMS["w_con_rg_f_v3f"],
                      p=PARAMS["p_con_rg_f_v3f"], delay=PARAMS["delay_vcell"])
        self._connect(f"v2a_{leg}", f"v0v_{leg}",
                      weight=PARAMS["w_con_v2a_v0v"],
                      p=PARAMS["p_con_v2a_v0v"], delay=PARAMS["delay_vcell"])
        self._connect(f"in1_{leg}", f"rg_f_{leg}",
                      weight=PARAMS["w_con_in1_rg_f"],
                      p=PARAMS["p_con_in1_rg_f"],
                      delay=PARAMS["delay_in1_rg_f"])

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

    def drain_activity(self, dt_ms: float):
        """Reads and clears every recorder, returning per-step rates and spikes."""
        rates, counts, events = {}, {}, {}
        for name, rec in self.recorders.items():
            ev = rec.get("events")
            times = np.asarray(ev.get("times", []), dtype=np.float64)
            senders = np.asarray(ev.get("senders", []), dtype=np.int64)
            n = int(times.size)
            ids = self.rec_ids[name]
            counts[name] = n
            rates[name] = (1000.0 * n / max(1, len(ids)) / dt_ms
                           if dt_ms > 0.0 else 0.0)
            if n:
                events[name] = {
                    "times": times,
                    "senders": np.searchsorted(ids, senders),
                }
            rec.set({"n_events": 0})
        self._last_rates = rates
        self._last_counts = counts
        self._last_events = events
        return rates, counts, events

    def get_muscle_drives(self, dt_ms: float,
                          current_time_ms: float = 0.0) -> Tuple[float, float]:
        """Normalized motor-pool drive in [0, 1] from this step's spikes."""
        if not self._last_counts:
            self.drain_activity(dt_ms)
        n_e = self._last_counts.get(f"m_e_{self.leg}", 0)
        n_f = self._last_counts.get(f"m_f_{self.leg}", 0)
        return self._drive(n_e, dt_ms), self._drive(n_f, dt_ms)

    def _drive(self, spikes: int, dt_ms: float, n: int | None = None) -> float:
        ref_hz = getattr(self, "ref_hz", PARAMS["ref_hz_default"])
        count = self.n_mn if n is None else int(n)
        max_spikes = max(1.0, count * ref_hz * (dt_ms / 1000.0))
        return float(np.clip(spikes / max_spikes, 0.0, 1.0))

    def get_rg_rates(self, dt_ms: float,
                     current_time_ms: float = 0.0) -> Tuple[float, float]:
        """Normalized RG population activity in [0, 1] from this step's spikes."""
        if not self._last_counts:
            self.drain_activity(dt_ms)
        n_e = self._last_counts.get(f"rg_e_{self.leg}", 0)
        n_f = self._last_counts.get(f"rg_f_{self.leg}", 0)
        return (self._drive(n_e, dt_ms, self.n_rg),
                self._drive(n_f, dt_ms, self.n_rg))
