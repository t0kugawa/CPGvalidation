"""Single source of truth for the CPG's initial parameter values.

Every value that used to be a literal in ``t2xcpg/*.py`` lives in one flat
``PARAMS`` dict.  Modules read their initial values from here; nothing else in
the package should define a model constant.

Row comments use a value-type shorthand:

    [ms]          simulation / delay unit
    [n_neurons]   population size
    [hz]          Poisson relay rate
    [mps]         treadmill speed reference
    [w_con]       static connection weight
    [p_con]       connection probability
    [w_stdp]      STDP seed weight (STDP owns the weight afterwards)
    [delay]       synaptic delay
    [izh]         Izhikevich neuron parameter
    [stdp]        STDP hyper-parameter
    [sigma]       per-subject / per-synapse dispersion
    [bound]       optimizable bound
    [scale]       tuning scale (env-overridable where a CPG_* var exists)
    [gain]        multiplicative drive gain
    [fraction]    fraction of another value
    [count]       structural count
    [choice]      non-numeric initial option

Connection rows use arrows for sign:

    SRC -> TGT    excitatory
    SRC -l TGT    inhibitory

Keys are lower-case code names.  ``rg_e`` etc. are the NEST populations; the
comments use the scheme's upper-case labels (RG_E, InE, ...).
"""

PARAMS = {
    # ── Simulation ──────────────────────────────────────────────────────────
    "dt_ms": 10.0,                              # [ms] control / NEST step
    "syn_ms": 1.0,                              # [ms] base synaptic delay
    "length_m": 0.005,                          # [m] conduction length
    "velocity_mps": 5.45,                       # [mps] conduction velocity
    "delay_scale": 1.0,                          # [scale] global delay scale
    "delay_jitter_ms": 0.2,                     # [sigma] delay jitter sd

    # ── Population sizes ────────────────────────────────────────────────────
    "n_rg_final": 100,                          # [n_neurons] RG per leg, FINAL
    "n_mn_final": 100,                          # [n_neurons] MN per leg, FINAL
    "n_int_final": 50,                          # [n_neurons] interneurons, FINAL
    "n_aff_final": 100,                         # [n_neurons] afferents, FINAL
    "n_bs_final": 100,                          # [n_neurons] BS per drive, FINAL
    "n_rg_default": 20,                         # [n_neurons] RG, Leg() default
    "n_mn_default": 20,                         # [n_neurons] MN, Leg() default
    "n_int_default": 15,                        # [n_neurons] IN, Leg() default
    "n_aff_default": 20,                        # [n_neurons] AFF, Leg() default
    "n_bs_default": 50,                         # [n_neurons] BS, BrainStem default
    "p_scale_default": 1.0,                     # [scale] Leg() p multiplier
    "w_scale_default": 1.0,                     # [scale] Leg() w multiplier
    "ref_hz_default": 100.0,                    # [hz] full-drive reference rate
    "cut_multisite_default": False,             # [choice] per-site CUT relays
    "n_cut_sites_default": 8,                   # [count] plantar CUT sites
    "n_cut_noise_tap": 1,                       # [n_neurons] aggregate CUT tap

    # ── Izhikevich presets (cpg_2legs_fast) ─────────────────────────────────
    "izh_init_a": 0.02,                         # [izh] default recovery time
    "izh_init_b": 0.2,                          # [izh] default recovery coupling
    "izh_init_c": -65.0,                        # [izh] default reset voltage
    "izh_init_d": 8.0,                          # [izh] default after-spike jump
    "izh_init_v_th": 30.0,                      # [izh] default spike threshold
    "izh_init_v_min": -120.0,                   # [izh] default floor voltage
    "izh_exc_a": 0.02,                          # [izh] excitatory a
    "izh_exc_b": 0.2,                           # [izh] excitatory b
    "izh_exc_c": -65.0,                         # [izh] excitatory c
    "izh_exc_d": 8.0,                           # [izh] excitatory d
    "izh_inh_a": 0.1,                           # [izh] inhibitory a (fast-spiking)
    "izh_inh_b": 0.2,                           # [izh] inhibitory b
    "izh_inh_c": -65.0,                         # [izh] inhibitory c
    "izh_inh_d": 2.0,                           # [izh] inhibitory d
    "izh_rgf_a": 0.02,                          # [izh] flexor RG a (bursting)
    "izh_rgf_b": 0.2,                           # [izh] flexor RG b
    "izh_rgf_c": -55.0,                         # [izh] flexor RG c
    "izh_rgf_d": 4.0,                           # [izh] flexor RG d
    "izh_mf_a": 0.02,                           # [izh] flexor MN a (frozen ref)
    "izh_mf_b": 0.2,                            # [izh] flexor MN b (frozen ref)
    "izh_mf_c": -65.0,                          # [izh] flexor MN c (frozen ref)
    "izh_mf_d": 8.0,                            # [izh] flexor MN d (frozen ref)

    # ── Izhikevich parameter shape ──────────────────────────────────────────
    "izh_sigma_a": 0.005,                       # [sigma] a dispersion
    "izh_min_a": 0.001,                         # [bound] a lower
    "izh_max_a": 0.1,                           # [bound] a upper
    "izh_sigma_b": 0.04,                        # [sigma] b dispersion
    "izh_min_b": 0.05,                          # [bound] b lower
    "izh_max_b": 0.5,                           # [bound] b upper
    "izh_sigma_c": 5.0,                         # [sigma] c dispersion
    "izh_min_c": -80.0,                         # [bound] c lower
    "izh_max_c": -50.0,                         # [bound] c upper
    "izh_sigma_d": 2.0,                         # [sigma] d dispersion
    "izh_min_d": 0.0,                           # [bound] d lower
    "izh_max_d": 20.0,                          # [bound] d upper

    # ── Sensory relays ──────────────────────────────────────────────────────
    "hz_ia": 10.0,                              # [hz] Ia relay rest rate
    "hz_ib": 5.0,                               # [hz] Ib relay rest rate
    "hz_cut_rest": 0.0,                         # [hz] CUT aggregate rest rate
    "hz_cut_site_rest": 0.0,                    # [hz] CUT per-site rest rate
    "hz_cut_noise_scale": 10.0,                 # [hz] CUT producer full scale

    # ── Synapse defaults and bounds ─────────────────────────────────────────
    "leg_p_default": 0.5,                       # [p_con] Leg._connect default
    "l2_p_default": 0.4,                        # [p_con] L2CPG._connect default
    "l2_delay_default": 3.0,                    # [delay] L2CPG._connect default
    "syn_weight_sigma_fraction": 0.1,           # [fraction] Leg weight sigma
    "syn_weight_min": -150.0,                   # [bound] Leg weight lower
    "syn_weight_max": 150.0,                    # [bound] Leg weight upper
    "l2_syn_weight_sigma_fraction": 0.1,        # [fraction] L2 weight sigma
    "l2_syn_weight_min": -100.0,                # [bound] L2 weight lower
    "l2_syn_weight_max": 100.0,                 # [bound] L2 weight upper
    "stdp_sigma_fraction": 0.1,                 # [fraction] STDP hyper sigma
    "stdp_sigma_floor": 1e-6,                   # [sigma] min STDP hyper sigma
    "stdp_wmax_seed_factor": 1.01,              # [fraction] Wmax above seed

    # ── STDP presets ────────────────────────────────────────────────────────
    "stdp_default_tau_plus": 20.0,              # [stdp] default tau_plus
    "stdp_default_lambda": 0.001,               # [stdp] default lambda
    "stdp_default_alpha": 0.95,                 # [stdp] default alpha
    "stdp_default_mu_plus": 0.4,                # [stdp] default mu_plus
    "stdp_default_mu_minus": 0.4,               # [stdp] default mu_minus
    "stdp_default_wmax": 120.0,                 # [stdp] default Wmax
    "stdp_ia_tau_plus": 20.0,                   # [stdp] Ia tau_plus
    "stdp_ia_lambda": 0.001,                    # [stdp] Ia lambda
    "stdp_ia_alpha": 0.95,                      # [stdp] Ia alpha
    "stdp_ia_mu_plus": 0.4,                     # [stdp] Ia mu_plus
    "stdp_ia_mu_minus": 0.4,                    # [stdp] Ia mu_minus
    "stdp_ia_wmax": 10.0,                       # [stdp] Ia Wmax
    "stdp_bs_tau_plus": 20.0,                   # [stdp] BS tau_plus
    "stdp_bs_lambda": 0.001,                    # [stdp] BS lambda
    "stdp_bs_alpha": 0.95,                      # [stdp] BS alpha
    "stdp_bs_mu_plus": 0.4,                     # [stdp] BS mu_plus
    "stdp_bs_mu_minus": 0.4,                    # [stdp] BS mu_minus
    "stdp_bs_wmax": 30.0,                       # [stdp] BS Wmax

    # ── STDP parameter ranges ───────────────────────────────────────────────
    "stdp_range_tau_plus_min": 1.0,             # [bound] tau_plus lower
    "stdp_range_tau_plus_max": 200.0,           # [bound] tau_plus upper
    "stdp_range_lambda_min": 1e-5,              # [bound] lambda lower
    "stdp_range_lambda_max": 0.1,               # [bound] lambda upper
    "stdp_range_alpha_min": 0.0,                # [bound] alpha lower
    "stdp_range_alpha_max": 1.0,                # [bound] alpha upper
    "stdp_range_mu_plus_min": 0.0,              # [bound] mu_plus lower
    "stdp_range_mu_plus_max": 1.0,              # [bound] mu_plus upper
    "stdp_range_mu_minus_min": 0.0,             # [bound] mu_minus lower
    "stdp_range_mu_minus_max": 1.0,             # [bound] mu_minus upper
    "stdp_range_wmax_min": 1.0,                 # [bound] Wmax lower
    "stdp_range_wmax_max": 2000.0,              # [bound] Wmax upper

    # ── Brainstem drive ─────────────────────────────────────────────────────
    "bs_e_hz": 60.0,                            # [hz] BS-E drive rate
    "bs_f_hz": 60.0,                            # [hz] BS-F drive rate
    "bs_base_fraction": 0.5,                    # [fraction] base / BS-E rate
    "bs_speed_ref_mps": 0.2,                    # [mps] speed reference
    "bs_rate_sigma": 5.0,                       # [sigma] BS rate dispersion
    "bs_rate_min": 0.0,                         # [bound] BS rate lower
    "bs_rate_max": 250.0,                       # [bound] BS rate upper
    "base_hz": 2.0,                             # [hz] tonic base drive rate
    "base_rate_sigma": 5.0,                     # [sigma] base rate dispersion
    "base_rate_min": 0.0,                       # [bound] base rate lower
    "base_rate_max": 250.0,                     # [bound] base rate upper

    # ── L2CPG / brainstem projections ───────────────────────────────────────
    "scale_bs": 1.0,                            # [scale] BS->RG weight scale
    "scale_base": 1.0,                          # [scale] BASE->RG weight scale
    "w_con_bs_rg": 2.0,                         # [w_con] BS -> RG_E/RG_F
    "p_con_bs_rg": 0.45,                        # [p_con] BS -> RG_E/RG_F
    "w_con_base_rg": 1.0,                       # [w_con] BASE -> RG_E/RG_F
    "p_con_base_rg": 0.10,                      # [p_con] BASE -> RG_E/RG_F
    "delay_base_rg": 1.0,                       # [delay] BASE -> RG_E/RG_F
    "leading_leg": "L",                         # [choice] first-stepping leg
    "leading_gain": 1.5,                        # [gain] leading-leg BS gain
    "leading_gain_other": 1.0,                  # [gain] non-leading-leg BS gain
    "p_comm": 0.40,                             # [p_con] commissural density
    "delay_comm": 3.0,                          # [delay] commissural delay
    "w_con_v0v_in1": 1.3,                       # [w_con] V0v -> In1
    "w_con_v0d_rg_f": -1.3,                     # [w_con] V0d -l RG_F
    "w_con_v3f_rg_f": 0.5,                      # [w_con] V3-F -> RG_F

    # ── Leg half-centre, recurrence, motor output ───────────────────────────
    "w_con_rg_e_in_e": 12.0,                    # [w_con] RG_E -> InE
    "p_con_rg_e_in_e": 0.15,                    # [p_con] RG_E -> InE
    "w_con_in_e_rg_f": -8.0,                    # [w_con] InE -l RG_F
    "p_con_in_e_rg_f": 0.15,                    # [p_con] InE -l RG_F
    "w_con_rg_f_in_f": 18.0,                    # [w_con] RG_F -> InF
    "p_con_rg_f_in_f": 0.30,                    # [p_con] RG_F -> InF
    "w_con_in_f_rg_e": -48.0,                   # [w_con] InF -l RG_E
    "p_con_in_f_rg_e": 0.30,                    # [p_con] InF -l RG_E
    "w_con_in_e_m_f": -8.0,                     # [w_con] InE -l M-F
    "p_con_in_e_m_f": 0.15,                     # [p_con] InE -l M-F
    "w_con_in_f_m_e": -48.0,                    # [w_con] InF -l M-E
    "p_con_in_f_m_e": 0.30,                     # [p_con] InF -l M-E
    "w_con_rg_e_rg_e": 4.0,                     # [w_con] RG_E -> RG_E
    "p_con_rg_e_rg_e": 0.12,                    # [p_con] RG_E -> RG_E
    "w_con_rg_f_rg_f": 5.5,                     # [w_con] RG_F -> RG_F
    "p_con_rg_f_rg_f": 0.12,                    # [p_con] RG_F -> RG_F
    "w_con_rg_e_m_e": 2.75,                     # [w_con] RG_E -> M-E
    "p_con_rg_e_m_e": 0.50,                     # [p_con] RG_E -> M-E
    "w_con_rg_f_m_f": 2.75,                     # [w_con] RG_F -> M-F (C1 seed)
    "p_con_rg_f_m_f": 0.50,                     # [p_con] RG_F -> M-F
    "w_con_m_e_mus_e": 1.0,                     # [w_con] M-E -> mE
    "p_con_m_e_mus_e": 0.80,                    # [p_con] M-E -> mE
    "w_con_m_f_mus_f": 1.0,                     # [w_con] M-F -> mF
    "p_con_m_f_mus_f": 0.80,                    # [p_con] M-F -> mF

    # ── Ia (stretch) pathways ───────────────────────────────────────────────
    "scale_ia": 1.0,                            # [scale] Ia weight scale
    "w_con_ia_e_in_m_e": 12.0,                  # [w_con] Ia-E -> M-E
    "p_con_ia_e_in_m_e": 0.40,                  # [p_con] Ia-E -> M-E
    "w_con_ia_f_in_m_f": 12.0,                  # [w_con] Ia-F -> M-F (C1 seed)
    "p_con_ia_f_in_m_f": 0.40,                  # [p_con] Ia-F -> M-F
    "w_stdp_ia_e_in_rg_e": 12.0,                # [w_stdp] Ia-E -> RG_E
    "p_con_ia_e_in_rg_e": 0.40,                 # [p_con] Ia-E -> RG_E
    "w_stdp_ia_f_in_rg_f": 12.0,                # [w_stdp] Ia-F -> RG_F
    "p_con_ia_f_in_rg_f": 0.40,                 # [p_con] Ia-F -> RG_F
    "w_con_ia_e_in_ia_e": 6.0,                  # [w_con] Ia-E -> Ia-IN-E
    "p_con_ia_e_in_ia_e": 0.40,                 # [p_con] Ia-E -> Ia-IN-E
    "w_con_ia_f_in_ia_f": 6.0,                  # [w_con] Ia-F -> Ia-IN-F
    "p_con_ia_f_in_ia_f": 0.40,                 # [p_con] Ia-F -> Ia-IN-F
    "w_con_ia_e_m_f": -10.0,                    # [w_con] Ia-IN-E -l M-F
    "p_con_ia_e_m_f": 0.40,                     # [p_con] Ia-IN-E -l M-F
    "w_con_ia_f_m_e": -10.0,                    # [w_con] Ia-IN-F -l M-E
    "p_con_ia_f_m_e": 0.40,                     # [p_con] Ia-IN-F -l M-E
    "w_con_ia_e_in_in_e": 6.0,                  # [w_con] Ia-E -> InE
    "p_con_ia_e_in_in_e": 0.25,                 # [p_con] Ia-E -> InE
    "w_con_ia_f_in_in_f": 6.0,                  # [w_con] Ia-F -> InF
    "p_con_ia_f_in_in_f": 0.25,                 # [p_con] Ia-F -> InF

    # ── Renshaw recurrent inhibition ────────────────────────────────────────
    "w_con_m_e_r_e": 3.0,                       # [w_con] M-E -> R-E
    "p_con_m_e_r_e": 0.50,                      # [p_con] M-E -> R-E
    "w_con_r_e_m_e": -4.0,                      # [w_con] R-E -l M-E
    "p_con_r_e_m_e": 0.50,                      # [p_con] R-E -l M-E
    "w_con_r_e_ia_e": -2.4,                     # [w_con] R-E -l Ia-IN-E
    "p_con_r_e_ia_e": 0.40,                     # [p_con] R-E -l Ia-IN-E
    "w_con_m_f_r_f": 3.0,                       # [w_con] M-F -> R-F
    "p_con_m_f_r_f": 0.50,                      # [p_con] M-F -> R-F
    "w_con_r_f_m_f": -4.0,                      # [w_con] R-F -l M-F
    "p_con_r_f_m_f": 0.50,                      # [p_con] R-F -l M-F
    "w_con_r_f_ia_f": -2.4,                     # [w_con] R-F -l Ia-IN-F
    "p_con_r_f_ia_f": 0.40,                     # [p_con] R-F -l Ia-IN-F

    # ── Ib (force) pathways ─────────────────────────────────────────────────
    "scale_ib": 1.0,                            # [scale] Ib weight scale
    "w_con_ib_e_in_in_ib_e": 16.0,              # [w_con] GTO-E -> InIb-E
    "p_con_ib_e_in_in_ib_e": 0.40,              # [p_con] GTO-E -> InIb-E
    "w_con_in_ib_e_m_e": -18.0,                 # [w_con] InIb-E -l M-E
    "p_con_in_ib_e_m_e": 0.40,                  # [p_con] InIb-E -l M-E
    "w_con_in_ib_e_m_f": 12.0,                  # [w_con] InIb-E -> M-F
    "p_con_in_ib_e_m_f": 0.35,                  # [p_con] InIb-E -> M-F
    "w_con_in_ib_e_rg_f": 18.0,                 # [w_con] InIb-E -> RG_F
    "p_con_in_ib_e_rg_f": 0.35,                 # [p_con] InIb-E -> RG_F
    "w_con_ib_f_in_in_ib_f": 16.0,              # [w_con] GTO-F -> InIb-F
    "p_con_ib_f_in_in_ib_f": 0.40,              # [p_con] GTO-F -> InIb-F
    "w_con_in_ib_f_m_f": -18.0,                 # [w_con] InIb-F -l M-F
    "p_con_in_ib_f_m_f": 0.40,                  # [p_con] InIb-F -l M-F
    "w_con_in_ib_f_m_e": 12.0,                  # [w_con] InIb-F -> M-E
    "p_con_in_ib_f_m_e": 0.35,                  # [p_con] InIb-F -> M-E
    "w_con_in_ib_f_rg_e": 18.0,                 # [w_con] InIb-F -> RG_E
    "p_con_in_ib_f_rg_e": 0.35,                 # [p_con] InIb-F -> RG_E
    "w_con_ia_e_in_in_ib_e": 8.0,               # [w_con] Ia-E -> InIb-E
    "p_con_ia_e_in_in_ib_e": 0.30,              # [p_con] Ia-E -> InIb-E
    "w_con_ia_f_in_in_ib_f": 8.0,               # [w_con] Ia-F -> InIb-F
    "p_con_ia_f_in_in_ib_f": 0.30,              # [p_con] Ia-F -> InIb-F

    # ── CUT (cutaneous) stance reinforcement ────────────────────────────────
    "scale_cut": 1.0,                           # [scale] CUT weight scale
    "w_stdp_cut_rge": 14.0,                     # [w_stdp] CUT -> RG_E
    "p_con_cut_rge": 0.35,                      # [p_con] CUT -> RG_E
    "w_con_cut_in_e": 6.0,                      # [w_con] CUT -> InE
    "p_con_cut_in_e": 0.30,                     # [p_con] CUT -> InE

    # ── V-cell relay, ipsilateral inputs ────────────────────────────────────
    "delay_vcell": 3.0,                         # [delay] RG_F -> V-cell
    "w_con_rg_f_v2a": 0.75,                     # [w_con] RG_F -> V2a
    "p_con_rg_f_v2a": 0.50,                     # [p_con] RG_F -> V2a
    "w_con_rg_f_v0d": 0.75,                     # [w_con] RG_F -> V0d
    "p_con_rg_f_v0d": 0.50,                     # [p_con] RG_F -> V0d
    "w_con_rg_f_v3f": 1.5,                      # [w_con] RG_F -> V3-F
    "p_con_rg_f_v3f": 0.50,                     # [p_con] RG_F -> V3-F
    "w_con_v2a_v0v": 1.2,                       # [w_con] V2a -> V0v
    "p_con_v2a_v0v": 0.50,                      # [p_con] V2a -> V0v
    "delay_in1_rg_f": 1.0,                      # [delay] In1 -> RG_F
    "w_con_in1_rg_f": -0.5,                     # [w_con] In1 -l RG_F
    "p_con_in1_rg_f": 0.50,                     # [p_con] In1 -l RG_F

    # ── Flexor drive compensation (C1) ──────────────────────────────────────
    "scale_inhib": 1.0,                         # [scale] inhibition weight scale
    "mf_drive_max_scale": 4.0,                  # [gain] max M-F drive boost
}
