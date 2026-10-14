import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Ensure root directory is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import nest
from persona.registry import Registry
from persona.agent import Agent
from t2xcpg.bs import BrainStem
from t2xcpg.leg import Leg as CPG_Leg
from t2xcpg.cpg import L2CPG
from model.leg import Leg as MuJoCo_Leg
from model.adapter import CPGOutput, CPGInput


def run_personalization(n_steps: int = 60,
                        dt_ms: float = 10.0,
                        target_tolerance_rad: float = 0.08,
                        plot_filename: str = "validation_results.png"):
    """
    Runs the closed-loop PERSONA optimization algorithm on the CPG connected
    to the MuJoCo rat hindlimb model until the desired ankle angle precision is achieved.
    Generates comprehensive biological validation plots.
    """
    print("=" * 70)
    print("STARTING PERSONA SPINAL CPG BIOMECHANICAL PERSONALIZATION")
    print("=" * 70)

    # 1. Initialize NEST Kernel
    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": 1.0, "print_time": False})

    # 2. Build PERSONA Registry and CPG Spiking Network (Diagram 2)
    reg = Registry()
    bs = BrainStem(registry=reg, n_bs=30, bs_e_hz=50.0, bs_f_hz=50.0)
    cpg_leg_l = CPG_Leg(leg_name="L", registry=reg, n_rg=20, n_mn=20, n_int=15, n_aff=20)
    cpg_leg_r = CPG_Leg(leg_name="R", registry=reg, n_rg=20, n_mn=20, n_int=15, n_aff=20)
    cpg = L2CPG(registry=reg, bs=bs, legs={"L": cpg_leg_l, "R": cpg_leg_r}, dt_ms=dt_ms)

    # 3. Build MuJoCo Biomechanical Hindlimb Simulation
    xml_path = os.path.join(BASE_DIR, "dual_rat_hindlimb.xml")
    mujoco_leg = MuJoCo_Leg(mujoco_cfg=xml_path, dt_ms=dt_ms)

    # 4. Instantiate PERSONA Natural Gradient Agent
    agent = Agent(
        registry=reg,
        lr_mu=0.03,
        lr_sigma_ctrl=0.005,
        lr_sigma_param=0.001,
        window_size=60
    )

    print(f"Registered optimizable parameters: {len(agent.params)}")
    print(f"Registered noise recorders: {list(reg.noise_recorders.keys())}")

    # Tracking records for plotting
    times_ms = []
    model_ankles = []
    ref_ankles = []
    errors = []
    mus_e_drives = []
    mus_f_drives = []
    cut_feedbacks = []
    ia_feedbacks = []
    ib_feedbacks = []
    phases = []
    stance_advantages = []
    swing_advantages = []

    # Initial Sensory Input
    cpg_in = CPGInput()

    print("\nStarting optimization loop...")
    converged = False

    for step in range(1, n_steps + 1):
        # 1. Agent samples exploratory parameter perturbations via controlling variance
        explored_actions = agent.sample_actions()
        reg.apply_explored_parameters(explored_actions)

        # 2. Update CPG with live sensory feedback and step spiking simulation
        cpg.update(cpg_in)
        cpg_out = cpg.step()

        # 3. Step MuJoCo physics and compute sensory afferents
        cpg_in = mujoco_leg.step(cpg_out)

        # 4. Measure ankle angles
        q_model = cpg_in.ankle_angle_m_L
        q_ref = cpg_in.ankle_angle_ref_L
        ankle_err = abs(q_ref - q_model)

        # 5. Preprocess and run Agent Natural Gradient / Fisher update
        # Vector output: [ankle_angle, muscle_E_drive, muscle_F_drive]
        # Target: [ankle_ref, target_E, target_F]
        # Extensor active in stance, Flexor active in swing
        target_e = 0.8 if cpg_in.phase_L == "stance" else 0.1
        target_f = 0.1 if cpg_in.phase_L == "stance" else 0.8

        vec_out = np.array([q_model, cpg_out.mus_E_L, cpg_out.mus_F_L])
        vec_target = np.array([q_ref, target_e, target_f])

        step_res = agent.step(
            output=vec_out,
            target=vec_target,
            phase=cpg_in.phase_L
        )

        # Log metrics
        t = cpg_in.ankle_angle_m_L
        times_ms.append(step * dt_ms)
        model_ankles.append(q_model)
        ref_ankles.append(q_ref)
        errors.append(ankle_err)
        mus_e_drives.append(cpg_out.mus_E_L)
        mus_f_drives.append(cpg_out.mus_F_L)
        cut_feedbacks.append(cpg_in.cut_L)
        ia_feedbacks.append(cpg_in.ia_rate_E_L)
        ib_feedbacks.append(cpg_in.ib_rate_E_L)
        phases.append(cpg_in.phase_L)

        adv = step_res["advantage_vector"]
        adv_mean = float(np.mean(adv))
        if cpg_in.phase_L == "stance":
            stance_advantages.append(adv_mean)
        else:
            swing_advantages.append(adv_mean)

        if step % 20 == 0 or step == n_steps:
            mean_recent_err = np.mean(errors[-20:])
            print(f"Step {step:3d}/{n_steps} | Phase: {cpg_in.phase_L:<6s} | "
                  f"Ref Ankle: {q_ref:+.3f} rad | Model Ankle: {q_model:+.3f} rad | "
                  f"Mean Error (20st): {mean_recent_err:.4f} rad")

        # Convergence condition check over recent window
        if step >= 40 and np.mean(errors[-15:]) < target_tolerance_rad:
            print(f"\nTarget ankle angle precision achieved at step {step}! (Mean Error: {np.mean(errors[-15:]):.4f} < {target_tolerance_rad})")
            converged = True
            break

    print("\n" + "=" * 70)
    print("SIMULATION COMPLETED. GENERATING BIOLOGICAL VALIDATION FIGURES...")
    print("=" * 70)

    # Generate Matplotlib validation figures
    fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)
    t_axis = np.array(times_ms) / 1000.0  # seconds

    # Panel 1: Ankle Kinematics Comparison
    ax1 = axes[0]
    ax1.plot(t_axis, ref_ankles, 'k--', label="Target Reference Ankle (rad)", linewidth=2.2)
    ax1.plot(t_axis, model_ankles, 'r-', label="CPG Model Ankle (rad)", linewidth=2.0)
    ax1.set_ylabel("Ankle Angle [rad]", fontsize=11, fontweight='bold')
    ax1.set_title("Biological Validation of Hindlimb Model Personalized by PERSONA", fontsize=14, fontweight='bold')
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Panel 2: Muscle Activation Drives
    ax2 = axes[1]
    ax2.plot(t_axis, mus_e_drives, 'tab:red', label="Soleus Extensor (mus_E)", linewidth=1.8)
    ax2.plot(t_axis, mus_f_drives, 'tab:blue', label="Tibialis Anterior Flexor (mus_F)", linewidth=1.8)
    ax2.set_ylabel("Muscle Drive", fontsize=11, fontweight='bold')
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(loc="upper right", framealpha=0.9)
    ax2.grid(True, linestyle=":", alpha=0.6)

    # Panel 3: Afferent Sensory Feedback (CUT, Ia, Ib)
    ax3 = axes[2]
    ax3.plot(t_axis, cut_feedbacks, 'tab:green', label="Plantar CUT Mechanoreceptor (Hz)", linewidth=1.8)
    ax3.plot(t_axis, ia_feedbacks, 'tab:orange', label="Spindle Ia Extensor Stretch (Hz)", linewidth=1.8)
    ax3.plot(t_axis, ib_feedbacks, 'tab:purple', label="Golgi Ib Extensor Tension (Hz)", linewidth=1.8)
    ax3.set_ylabel("Afferent Rate [Hz]", fontsize=11, fontweight='bold')
    ax3.legend(loc="upper right", framealpha=0.9)
    ax3.grid(True, linestyle=":", alpha=0.6)

    # Panel 4: Kinematic Tracking Error
    ax4 = axes[3]
    ax4.plot(t_axis, errors, 'tab:brown', label="|Ankle Error| [rad]", linewidth=1.8)
    ax4.axhline(target_tolerance_rad, color='gray', linestyle='--', label=f"Tolerance Target ({target_tolerance_rad} rad)")
    ax4.set_ylabel("Error [rad]", fontsize=11, fontweight='bold')
    ax4.legend(loc="upper right", framealpha=0.9)
    ax4.grid(True, linestyle=":", alpha=0.6)

    # Panel 5: Stance vs Swing Advantage Adaptation
    ax5 = axes[4]
    # Mark stance vs swing background shading
    for i in range(len(phases) - 1):
        if phases[i] == "stance":
            ax5.axvspan(t_axis[i], t_axis[i+1], color='whitesmoke', alpha=0.6)
        else:
            ax5.axvspan(t_axis[i], t_axis[i+1], color='mistyrose', alpha=0.4)

    adv_all = [float(np.mean(agent.history["advantage_vector"][i])) for i in range(len(times_ms))]
    ax5.plot(t_axis, adv_all, 'tab:blue', label="Component Advantage A(t)", linewidth=1.8)
    ax5.axhline(0.0, color='black', linestyle=':', alpha=0.7)
    ax5.set_xlabel("Time [s]", fontsize=12, fontweight='bold')
    ax5.set_ylabel("Advantage", fontsize=11, fontweight='bold')
    ax5.legend(loc="upper right", framealpha=0.9)
    ax5.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    output_path = os.path.join(BASE_DIR, plot_filename)
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"\nValidation plots saved successfully to: {output_path}")
    print("=" * 70)
    return {
        "converged": converged,
        "final_error": errors[-1],
        "mean_last20_error": float(np.mean(errors[-20:])),
        "plot_path": output_path
    }


if __name__ == "__main__":
    run_personalization()
