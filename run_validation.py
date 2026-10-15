"""
20-second full-framework validation run:
  * builds the closed loop (bilateral CPG + MuJoCo rig + mocap target),
  * runs PERSONA personalization inside the loop,
  * renders an oblique-view HUD movie (mp4),
  * writes the run log and the validation figures (tracking, learning,
    parameter distribution shift, Fisher matrix).

Usage (headless):
    MUJOCO_GL=osmesa python run_validation.py --duration-s 20 --view oblique_view
"""
import argparse
import os
import sys

os.environ.setdefault("MUJOCO_GL", "osmesa")

import numpy as np
import nest

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import imageio

from personalize import build_personalizer, DEFAULT_REFERENCE, DEFAULT_STATS, DT_MS
from persona.personalization import RunLog
from persona.validation import ValidationReport
from model.render import LegRenderer, hud_lines
from model.adapter import CPGInput


def run(duration_s, view, width, height, fps, out_dir,
        w_scale, ref_hz, belt_mps, belt_start_s, belt_ramp_s,
        n_rg, n_mn, n_int, n_aff, n_bs, render=True):
    os.makedirs(out_dir, exist_ok=True)
    steps = int(round(duration_s / (DT_MS / 1000.0)))
    n_frames = int(round(duration_s * fps))
    frame_steps = {int(round(k / (fps * (DT_MS / 1000.0)))) for k in range(n_frames)}

    pers, loader = build_personalizer(
        reference_path=DEFAULT_REFERENCE,
        n_rg=n_rg, n_mn=n_mn, n_int=n_int, n_aff=n_aff, n_bs=n_bs,
        max_steps=steps, eps_pose=0.10, pose_hold_steps=3,
        loss_stop=1e-4, loss_window=20, w_scale=w_scale, ref_hz=ref_hz)
    reg, cpg, rig, agent, driver = (pers.registry, pers.cpg, pers.rig,
                                    pers.agent, pers.target_driver)
    renderer = LegRenderer(rig.model, rig.data, width=width, height=height) if render else None
    writer = None
    if render:
        writer = imageio.get_writer(os.path.join(out_dir, f"locomotion_{view}.mp4"),
                                    fps=fps, codec="libx264", quality=8,
                                    macro_block_size=1)

    belt_start_step = int(round(belt_start_s / (DT_MS / 1000.0)))
    belt_ramp_steps = max(1, int(round(belt_ramp_s / (DT_MS / 1000.0))))

    log = RunLog()
    cpg_in = CPGInput()
    stage = "pose"
    pose_ok = 0
    tgt = driver.pose()
    for step in range(steps):
        if step >= belt_start_step:
            frac = min(1.0, (step - belt_start_step + 1) / belt_ramp_steps)
            rig.set_treadmill_surfacevel(belt_mps * frac)

        if stage == "pose":
            tgt_L, tgt_R = driver.pose()
            rig_kwargs = {"target_ankle_L": tgt_L, "target_ankle_R": tgt_R}
        else:
            rig_kwargs = driver.target(cpg.current_time_ms)
            tgt_L = rig_kwargs["target_ankle_L"]
            tgt_R = rig_kwargs["target_ankle_R"]

        actions = agent.sample_actions()
        reg.apply_explored_parameters(actions)

        cpg.update(cpg_in)
        cpg_out = cpg.step()
        cpg_in = rig.step(cpg_out, **rig_kwargs)

        res = agent.step(
            output=np.array([cpg_in.ankle_angle_m_L, cpg_in.ankle_angle_m_R]),
            target=np.array([tgt_L, tgt_R]),
            phase=[cpg_in.phase_L, cpg_in.phase_R])

        log.time_ms.append(cpg.current_time_ms)
        log.ankle_model_L.append(cpg_in.ankle_angle_m_L)
        log.ankle_model_R.append(cpg_in.ankle_angle_m_R)
        log.ankle_target_L.append(tgt_L)
        log.ankle_target_R.append(tgt_R)
        log.loss.append(float(res["loss"]))
        log.advantage.append(float(np.mean(res["advantage_vector"])))
        log.phase_L.append(cpg_in.phase_L)
        log.phase_R.append(cpg_in.phase_R)
        log.stage.append(stage)
        log.mus_E_L.append(cpg_out.mus_E_L)
        log.mus_F_L.append(cpg_out.mus_F_L)
        log.mus_E_R.append(cpg_out.mus_E_R)
        log.mus_F_R.append(cpg_out.mus_F_R)
        log.cut_L.append(cpg_in.cut_L)
        log.ia_rate_E_L.append(cpg_in.ia_rate_E_L)
        log.ib_rate_E_L.append(cpg_in.ib_rate_E_L)

        if render and step in frame_steps:
            lines = hud_lines(cpg_in, cpg_out, cpg.current_time_ms, stage)
            writer.append_data(renderer.frame(view, lines))

        if stage == "pose":
            err = max(abs(cpg_in.ankle_angle_m_L - tgt_L), abs(cpg_in.ankle_angle_m_R - tgt_R))
            pose_ok = pose_ok + 1 if err < 0.10 else 0
            if pose_ok >= 3:
                stage = "track"
        if step % 100 == 0:
            print(f"[{cpg.current_time_ms:6.0f} ms] {stage:5s} loss {res['loss']:.4f} "
                  f"L {cpg_in.ankle_angle_m_L:+.3f}/{tgt_L:+.3f} "
                  f"drv {cpg_out.mus_E_L:.2f}/{cpg_out.mus_F_L:.2f}", flush=True)

    log.finalize()
    log.to_csv(os.path.join(out_dir, "run_log.csv"))
    if writer is not None:
        writer.close()
    if renderer is not None:
        renderer.close()

    rep = ValidationReport()
    rep.plot_tracking(log)
    rep.plot_learning(log)
    rep.plot_param_shift(agent, top_n=10)
    rep.plot_fisher(agent)
    rep.save_all(out_dir)
    print("figures + mp4 written to", out_dir)
    print("first10 loss", float(np.mean(log.loss[:10])), "last10", float(np.mean(log.loss[-10:])))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration-s", type=float, default=20.0)
    ap.add_argument("--view", type=str, default="oblique_view")
    ap.add_argument("--fps", type=int, default=40)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--out", type=str, default=os.path.join(BASE_DIR, "validation_run"))
    ap.add_argument("--w-scale", type=float, default=0.5)
    ap.add_argument("--ref-hz", type=float, default=100.0)
    ap.add_argument("--belt-mps", type=float, default=0.2)
    ap.add_argument("--belt-start-s", type=float, default=0.5)
    ap.add_argument("--belt-ramp-s", type=float, default=0.5)
    ap.add_argument("--n-rg", type=int, default=10)
    ap.add_argument("--n-mn", type=int, default=8)
    ap.add_argument("--n-int", type=int, default=8)
    ap.add_argument("--n-aff", type=int, default=6)
    ap.add_argument("--n-bs", type=int, default=8)
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()
    run(args.duration_s, args.view, args.width, args.height, args.fps, args.out,
        args.w_scale, args.ref_hz, args.belt_mps, args.belt_start_s, args.belt_ramp_s,
        args.n_rg, args.n_mn, args.n_int, args.n_aff, args.n_bs, render=not args.no_render)