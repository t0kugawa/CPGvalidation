"""Single source of truth for the biomechanics / afferent initial values.

Every value that used to be a literal in ``model/*.py`` lives in one flat
``PARAMS`` dict.  Modules read their initial values from here; nothing else in
the package should define a model constant.

Row comments use a value-type shorthand:

    [ms]          simulation / time constant
    [s]           simulation duration
    [m]           length
    [rad]         angle
    [kg]          mass
    [n]           force
    [hz]          firing rate
    [fps]         render frame rate
    [px]          render pixels
    [percentile]  percentile selector
    [fraction]    normalised drive / unit prior
    [gain]        sensor gain
    [exponent]    power law exponent
    [pressure]    CUT pressure scale
    [choice]      non-numeric initial option
"""

PARAMS = {
    # ── Simulation / rig ────────────────────────────────────────────────────
    "model_dt_ms": 10.0,                        # [ms] Model control step
    "leg_dt_ms": 2.0,                           # [ms] Leg physics step
    "object_mass_kg": 300.0,                    # [kg] model object mass
    "sim_ms": 60000.0,                          # [ms] maximum simulated time
    "render_fps": 40,                           # [fps] replay / render rate
    "render_width_px": 1920,                    # [px] render width
    "render_height_px": 1080,                   # [px] render height
    "treadmill_penetration_m": 0.0005,          # [m] belt above foot rest
    "contact_z_eps_m": 0.003,                   # [m] retired contact gate
    "ankle_range_lo_rad": 0.19,                 # [rad] ankle stop, lower
    "ankle_range_hi_rad": 1.10,                 # [rad] ankle stop, upper
    "default_pose_hip_rad": -0.2,               # [rad] reset hip angle
    "default_pose_knee_rad": 0.6,               # [rad] reset knee angle
    "default_pose_ankle_rad": 0.3,              # [rad] reset ankle angle
    "muscle_drive_min": 0.0,                    # [fraction] drive lower clip
    "muscle_drive_max": 1.0,                    # [fraction] drive upper clip

    # ── Reference gait kinematics ───────────────────────────────────────────
    "reference_freq_hz": 1.5,                   # [hz] reference cycle rate
    "reference_hip_offset_rad": -0.25,          # [rad] hip offset
    "reference_hip_amp_rad": 0.35,              # [rad] hip amplitude
    "reference_knee_offset_rad": 0.65,          # [rad] knee offset
    "reference_knee_amp_rad": 0.30,             # [rad] knee amplitude
    "reference_ankle_offset_rad": 0.40,         # [rad] ankle offset
    "reference_ankle_amp_rad": 0.45,            # [rad] ankle amplitude
    "reference_ankle_phase_rad": -0.5,          # [rad] ankle phase lag
    "reference_right_phase_rad": 3.141592653589793,  # [rad] right-leg phase shift
    "target_foot_percentile": 5.0,              # [percentile] target-foot low

    # ── Plantar CUT ─────────────────────────────────────────────────────────
    "cut_rate_max_hz": 120.0,                   # [hz] CUT saturation rate
    "cut_active_force_n": 0.01,                 # [n] site contact threshold
    "cut_dt_ms": 2.0,                           # [ms] CUT afferent step
    "cut_beta": 0.3,                            # [gain] sigmoid slope
    "cut_p_theta": 10.0,                        # [pressure] sigmoid midpoint
    "cut_tau_ms": 20.0,                         # [ms] low-pass time constant
    "cut_force_floor_n": 1e-4,                  # [n] force dead zone
    "cut_pressure_cap": 100.0,                  # [pressure] pressure cap
    "cut_pressure_gain": 50.0,                  # [gain] force -> pressure
    "cut_contact_on_n": 0.010,                  # [n] stance entry threshold
    "cut_contact_off_n": 0.005,                 # [n] stance exit threshold
    "cut_contact_initial": "stance",            # [choice] initial phase state
    "cut_site_area": 1.0,                       # [fraction] tributary area prior
    "cut_site_density": 1.0,                    # [fraction] receptor density prior

    # ── Ia (muscle spindle) ─────────────────────────────────────────────────
    "ia_default_l0_m": 0.014,                   # [m] default reference length
    "ia_default_k_l": 2000.0,                   # [gain] length gain
    "ia_default_k_v": 150.0,                    # [gain] velocity gain
    "ia_default_k_force": 6.0,                  # [gain] load gain
    "ia_default_base_hz": 10.0,                 # [hz] rest rate
    "ia_default_max_hz": 250.0,                 # [hz] saturation rate
    "ia_velocity_exponent": 0.6,                # [exponent] velocity power
    "leg_ia_sol_l0_m": 0.0142,                  # [m] SOL reference length
    "leg_ia_ta_l0_m": 0.0072,                   # [m] TA reference length
    "leg_ia_k_l": 2000.0,                       # [gain] Leg length gain
    "leg_ia_k_v": 120.0,                        # [gain] Leg velocity gain
    "leg_ia_k_force": 6.0,                      # [gain] Leg load gain
    "leg_ia_base_hz": 10.0,                     # [hz] Leg rest rate
    "leg_ia_max_hz": 250.0,                     # [hz] Leg saturation rate

    # ── Ib (Golgi tendon organ) ─────────────────────────────────────────────
    "ib_default_k_force": 25.0,                 # [gain] default force gain
    "ib_default_k_act": 30.0,                   # [gain] default activation gain
    "ib_default_base_hz": 2.0,                  # [hz] default rest rate
    "ib_default_max_hz": 200.0,                 # [hz] default saturation rate
    "leg_ib_k_force": 15.0,                     # [gain] Leg force gain
    "leg_ib_k_act": 40.0,                       # [gain] Leg activation gain
    "leg_ib_base_hz": 2.0,                      # [hz] Leg rest rate
    "leg_ib_max_hz": 200.0,                     # [hz] Leg saturation rate
}
