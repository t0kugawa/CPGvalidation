#!/usr/bin/env python
"""Single closed-loop run: CPG + MuJoCo rig + PERSONA optimization.

Examples:
    python run.py --sim-time 120 --render --out runs/full
    python run.py --preset fast --sim-time 4 --out runs/smoke
    python run.py --sim-time 60 --set w_con_rg_e_m_e=3.0 --set scale_ia=0.8
"""
import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time

os.environ.setdefault("MUJOCO_GL", "osmesa")
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

DEFAULT_REF_HZ = 700.0
DEFAULT_RESOLUTION_MS = 0.2

PRESETS = {
    "full": {},
    "half": {"n_rg": 50, "n_mn": 50, "n_int": 25, "n_aff": 50, "n_bs": 50},
    "fifth": {"n_rg": 20, "n_mn": 20, "n_int": 10, "n_aff": 20, "n_bs": 20},
    "fast": {"n_rg": 6, "n_mn": 5, "n_int": 5, "n_aff": 3, "n_bs": 5,
             "sim_time": 4.0, "width": 480, "height": 270, "fps": 20},
}


def _coerce(text: str):
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return text


def _load_params(module_name: str, path: str):
    """Loads a params module without importing its parent package."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.PARAMS


def _parse_set(items):
    values = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--set expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        values[key.strip()] = _coerce(value.strip())
    return values


def _apply_overrides(cpg_params, model_params, overrides):
    applied = {}
    for key, value in overrides.items():
        hit = False
        if key in cpg_params:
            cpg_params[key] = value
            hit = True
        if key in model_params:
            model_params[key] = value
            hit = True
        if not hit:
            raise SystemExit(f"--set {key}: unknown parameter")
        applied[key] = value
    return applied


def _git_hash():
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=BASE_DIR,
            stderr=subprocess.DEVNULL, text=True).strip()
        return out
    except Exception:
        return ""


def _write_params_snapshot(out_dir, args, cpg_params, model_params, suffix=""):
    folder = os.path.join(out_dir, "params_snapshot")
    os.makedirs(folder, exist_ok=True)
    for rel, name in (("t2xcpg/params.py", "t2xcpg_params.py"),
                      ("model/params.py", "model_params.py")):
        shutil.copy2(os.path.join(BASE_DIR, rel), os.path.join(folder, name))
    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git": _git_hash(),
        "argv": sys.argv,
        "args": vars(args),
        "t2xcpg": dict(cpg_params),
        "model": dict(model_params),
    }
    target = os.path.join(folder, f"resolved{suffix}.json")
    with open(target, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    return folder


class _Snapshotter:
    """Writes run.h5/run_log.csv atomically at a wall-clock interval."""

    def __init__(self, out_dir, every_s, meta_fn):
        self.out_dir = out_dir
        self.every_s = float(every_s or 0.0)
        self.meta_fn = meta_fn
        self.t_last = time.monotonic()
        self.count = 0

    def maybe(self, step, log, agent):
        if self.every_s <= 0.0:
            return
        now = time.monotonic()
        if now - self.t_last < self.every_s:
            return
        self.t_last = now
        self.write(step, log, agent)

    def write(self, step, log, agent):
        from persona.h5io import save_run
        t0 = time.monotonic()
        try:
            tmp = os.path.join(self.out_dir, "run.h5.tmp")
            save_run(tmp, log, agent, meta=self.meta_fn(step, log))
            os.replace(tmp, os.path.join(self.out_dir, "run.h5"))
            tmp_csv = os.path.join(self.out_dir, "run_log.csv.tmp")
            log.to_csv(tmp_csv)
            os.replace(tmp_csv, os.path.join(self.out_dir, "run_log.csv"))
            self.count += 1
            print(f"[snapshot {self.count}] step {step} "
                  f"t={log.time_ms[-1] / 1000.0:.1f}s "
                  f"samples={len(log.time_ms)} "
                  f"({time.monotonic() - t0:.1f}s)", flush=True)
        except Exception as exc:
            print(f"[snapshot] failed at step {step}: {exc}", flush=True)


def run(duration_s, view, width, height, fps, out_dir,
        w_scale, ref_hz, belt_mps, belt_start_s, belt_ramp_s,
        n_rg, n_mn, n_int, n_aff, n_bs, render=True, budget_s=None,
        pose_max_s=1.0, belt_mode="dataset", snapshot_s=900.0,
        tau_mem_s=2.0, realize_every_k=1,
        resolution_ms=DEFAULT_RESOLUTION_MS, cut_multisite=False,
        game=None, seed=0, lr_mode="constant", lr_start=0.03,
        lr_end=None, nest_threads=1, noise_config="all",
        trace_strategy=False, trace_float32=False, snapshot_strategy=0,
        loss_stop=1e-4, loss_window=20, accept_stop=False,
        accept_margin=0.05, accept_coverage=0.70, accept_window=50,
        spikes="full", params_snapshot=False, params_snapshot_s=0.0,
        cpg_params=None, model_params=None, snapshot_args=None):
    import numpy as np
    import imageio

    from personalize import build_personalizer, DEFAULT_REFERENCE, DT_MS
    from persona.validation import ValidationReport, biological_acceptance
    from persona.h5io import save_run
    from model.render import LegRenderer, hud_lines

    os.makedirs(out_dir, exist_ok=True)
    steps = int(round(duration_s / (DT_MS / 1000.0)))
    n_frames = int(round(duration_s * fps))
    frame_steps = {int(round(k / (fps * (DT_MS / 1000.0))))
                   for k in range(n_frames)}

    t_build0 = time.perf_counter()
    pers, loader = build_personalizer(
        reference_path=DEFAULT_REFERENCE,
        n_rg=n_rg, n_mn=n_mn, n_int=n_int, n_aff=n_aff, n_bs=n_bs,
        max_steps=steps, eps_pose=0.10, pose_hold_steps=3,
        loss_stop=loss_stop, loss_window=loss_window, w_scale=w_scale,
        ref_hz=ref_hz, belt_mode=belt_mode, loop=True, tau_mem_s=tau_mem_s,
        resolution_ms=resolution_ms, cut_multisite=cut_multisite,
        game=game, seed=seed, lr_mode=lr_mode, lr_start=lr_start,
        lr_end=lr_end, nest_threads=nest_threads,
        noise_config=noise_config, trace_strategy=trace_strategy,
        trace_float32=trace_float32, snapshot_strategy=snapshot_strategy)
    build_s = time.perf_counter() - t_build0
    print(f"[timing] nest_threads={int(nest_threads)} build_s={build_s:.2f}",
          flush=True)
    pers.belt_mps = float(belt_mps)
    pers.belt_start_step = int(round(belt_start_s / (DT_MS / 1000.0)))
    pers.belt_ramp_steps = max(1, int(round(belt_ramp_s / (DT_MS / 1000.0))))
    pers.log_every = 100
    pers.stdp_every = 20
    pers.accept_stop = bool(accept_stop)
    pers.accept_margin = float(accept_margin)
    pers.accept_coverage = float(accept_coverage)
    pers.accept_window = int(accept_window)
    pers.record_spikes = spikes == "full"
    agent = pers.agent
    rig = pers.rig

    snapshot_args = snapshot_args or {}
    if params_snapshot:
        folder = _write_params_snapshot(
            out_dir, snapshot_args, cpg_params or {}, model_params or {})
        print(f"[params] snapshot -> {folder}", flush=True)
    t_params = time.monotonic()

    def _meta(step, log):
        return {"duration_s": float(duration_s), "dt_ms": float(DT_MS),
                "resolution_ms": float(resolution_ms),
                "cut_multisite": bool(cut_multisite),
                "belt_mode": str(belt_mode), "belt_z": float(rig.belt_z),
                "n_steps": int(len(log.time_ms)), "snapshot_step": int(step),
                "ref_hz": float(ref_hz), "w_scale": float(w_scale),
                "n_rg": int(n_rg), "n_mn": int(n_mn), "n_int": int(n_int),
                "n_aff": int(n_aff), "n_bs": int(n_bs),
                "belt_mps": float(belt_mps),
                "belt_start_s": float(belt_start_s),
                "belt_ramp_s": float(belt_ramp_s),
                "tau_mem_s": float(tau_mem_s),
                "realize_every_k": int(realize_every_k),
                "pose_max_s": float(pose_max_s), "seed": int(seed),
                "game": "selected" if game is not None else "none",
                "stop_reason": getattr(log, "stop_reason", "")}

    snap = _Snapshotter(out_dir, snapshot_s, _meta)
    renderer = (LegRenderer(rig.model, rig.data, width=width, height=height)
                if render else None)
    writer = None
    if render:
        path = os.path.join(out_dir, f"locomotion_{view}.mp4")
        writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8,
                                    macro_block_size=1)

    def _hook(step, log, cpg_in, cpg_out, stage):
        nonlocal t_params
        if render and step in frame_steps:
            lines = hud_lines(cpg_in, cpg_out, log.time_ms[-1], stage)
            writer.append_data(renderer.frame(view, lines))
        snap.maybe(step, log, agent)
        if params_snapshot and params_snapshot_s and \
                time.monotonic() - t_params >= params_snapshot_s:
            t_params = time.monotonic()
            _write_params_snapshot(
                out_dir, snapshot_args, cpg_params or {}, model_params or {},
                suffix=f"_{int(log.time_ms[-1])}")

    status = {}
    t_run0 = time.perf_counter()
    try:
        log = pers.run(step_hook=_hook, wall_clock_s=budget_s,
                       pose_max_steps=int(round(pose_max_s / (DT_MS / 1000.0))),
                       realize_every_k=realize_every_k)
    finally:
        if writer is not None:
            try:
                writer.close()
                status["locomotion.mp4"] = "ok"
            except Exception as exc:
                status["locomotion.mp4"] = f"ERROR {exc}"
        if renderer is not None:
            try:
                renderer.close()
            except Exception:
                pass
    run_s = time.perf_counter() - t_run0
    n_done = int(len(log.time_ms))
    print(f"[timing] nest_threads={int(nest_threads)} steps={n_done} "
          f"run_s={run_s:.2f} steps_per_s={n_done / max(run_s, 1e-9):.3f}",
          flush=True)
    print(f"[stop] {log.stop_reason} at {n_done} steps", flush=True)

    for name, fn in (("run_log.csv", log.to_csv),
                     ("run.h5", lambda p: save_run(
                         p, log, agent, meta=_meta(n_done, log)))):
        try:
            fn(os.path.join(out_dir, name))
            status[name] = "ok"
        except Exception as exc:
            status[name] = f"ERROR {exc}"

    stats = None
    try:
        from loader.stats import StatsLoader
        from personalize import DEFAULT_STATS, acceptance_from_log
        if os.path.exists(DEFAULT_STATS):
            stats = StatsLoader(DEFAULT_STATS)
        acceptance = acceptance_from_log(
            log, stats, margin=accept_margin, coverage=accept_coverage)
    except Exception:
        acceptance = {"acceptable": False, "angle_coverage": 0.0}

    rep = ValidationReport()
    for name, fn in (("tracking", rep.plot_tracking),
                     ("model_vs_target", rep.plot_model_vs_target),
                     ("learning", rep.plot_learning),
                     ("sensory", rep.plot_sensory),
                     ("stdp", rep.plot_stdp)):
        try:
            fn(log)
            status[f"{name}.png"] = "ok"
        except Exception as exc:
            status[f"{name}.png"] = f"ERROR {exc}"
    for name, fn in (("param_shift", lambda: rep.plot_param_shift(agent, top_n=10)),
                     ("fisher", lambda: rep.plot_fisher(agent)),
                     ("influence", lambda: rep.plot_influence(agent)),
                     ("metric", lambda: rep.plot_metric(agent))):
        try:
            fn()
            status[f"{name}.png"] = "ok"
        except Exception as exc:
            status[f"{name}.png"] = f"ERROR {exc}"
    try:
        rep.save_all(out_dir)
    except Exception as exc:
        status["save_all"] = f"ERROR {exc}"

    print("---- output status ----")
    for key, value in status.items():
        print(f"  {key}: {value}")
    print("files in", out_dir, ":", sorted(os.listdir(out_dir)))
    print(f"stop={log.stop_reason} acceptable={acceptance['acceptable']} "
          f"coverage={acceptance['angle_coverage']:.2f}")
    if len(log.loss):
        print("first10 loss", float(np.mean(log.loss[:10])),
              "last10", float(np.mean(log.loss[-10:])))
    return {"stop_reason": log.stop_reason, "acceptance": acceptance,
            "out_dir": out_dir, "n_steps": n_done}


def _build_parser():
    ap = argparse.ArgumentParser(
        description="Closed-loop CPG + MuJoCo + PERSONA run.")
    ap.add_argument("--out", default=os.path.join(BASE_DIR, "run"))
    ap.add_argument("--sim-time", "--duration-s", dest="sim_time", type=float,
                    default=None, help="simulated duration in seconds")
    ap.add_argument("--preset", choices=tuple(PRESETS), default="full")
    ap.add_argument("--final", action="store_true",
                    help="alias for --preset full")
    ap.add_argument("--fast", action="store_true",
                    help="alias for --preset fast")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a t2xcpg/model PARAMS value")
    ap.add_argument("--render", action="store_true",
                    help="write locomotion_<view>.mp4")
    ap.add_argument("--no-render", dest="render", action="store_false")
    ap.add_argument("--view", default=None)
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--fps", type=int, default=None)
    ap.add_argument("--w-scale", type=float, default=1.0)
    ap.add_argument("--ref-hz", type=float, default=DEFAULT_REF_HZ)
    ap.add_argument("--resolution-ms", type=float, default=DEFAULT_RESOLUTION_MS)
    ap.add_argument("--belt-mps", type=float, default=0.2)
    ap.add_argument("--belt-start-s", type=float, default=0.5)
    ap.add_argument("--belt-ramp-s", type=float, default=0.5)
    ap.add_argument("--belt", default="dataset", choices=("dataset", "model"))
    ap.add_argument("--n-rg", type=int, default=None)
    ap.add_argument("--n-mn", type=int, default=None)
    ap.add_argument("--n-int", type=int, default=None)
    ap.add_argument("--n-aff", type=int, default=None)
    ap.add_argument("--n-bs", type=int, default=None)
    ap.add_argument("--budget-s", type=float, default=None,
                    help="wall-clock budget in seconds (0 disables)")
    ap.add_argument("--pose-max-s", type=float, default=1.0)
    ap.add_argument("--snapshot-s", type=float, default=900.0,
                    help="wall-clock interval between data snapshots")
    ap.add_argument("--tau-mem-s", type=float, default=2.0)
    ap.add_argument("--realize-every-k", type=int, default=1)
    ap.add_argument("--cut-multisite", action="store_true")
    ap.add_argument("--game", choices=("none", "selected"), default="selected")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nest-threads", type=int, default=1)
    ap.add_argument("--lr-mode", choices=("constant", "cosine"),
                    default="constant")
    ap.add_argument("--lr-start", type=float, default=0.03)
    ap.add_argument("--lr-end", type=float, default=None)
    ap.add_argument("--noise-config", default="all")
    ap.add_argument("--trace-strategy", action="store_true")
    ap.add_argument("--trace-float32", action="store_true")
    ap.add_argument("--snapshot-strategy", type=int, default=0)
    ap.add_argument("--loss-stop", type=float, default=1e-4)
    ap.add_argument("--loss-window", type=int, default=20)
    ap.add_argument("--accept-stop", dest="accept_stop", action="store_true",
                    default=True)
    ap.add_argument("--no-accept-stop", dest="accept_stop",
                    action="store_false")
    ap.add_argument("--accept-margin", type=float, default=0.05)
    ap.add_argument("--accept-coverage", type=float, default=0.70)
    ap.add_argument("--accept-window", type=int, default=50)
    ap.add_argument("--spikes", choices=("full", "rates", "none"),
                    default="full")
    ap.add_argument("--params-snapshot", dest="params_snapshot",
                    action="store_true", default=True)
    ap.add_argument("--no-params-snapshot", dest="params_snapshot",
                    action="store_false")
    ap.add_argument("--params-snapshot-s", type=float, default=0.0,
                    help="wall-clock interval between parameter snapshots")
    return ap


def main():
    ap = _build_parser()
    args = ap.parse_args()
    if args.nest_threads < 1:
        ap.error("--nest-threads must be >= 1")
    if args.lr_mode == "cosine" and args.lr_end is None:
        ap.error("--lr-end is required with --lr-mode cosine")

    cpg_params = _load_params("t2xcpg.params",
                              os.path.join(BASE_DIR, "t2xcpg", "params.py"))
    model_params = _load_params("model.params",
                                os.path.join(BASE_DIR, "model", "params.py"))
    overrides = _parse_set(args.set)
    _apply_overrides(cpg_params, model_params, overrides)

    if args.fast:
        args.preset = "fast"
    if args.final:
        args.preset = "full"
    preset = PRESETS[args.preset]
    if args.sim_time is None:
        args.sim_time = preset.get("sim_time", 120.0)
    args.width = args.width or preset.get("width", 1920)
    args.height = args.height or preset.get("height", 1080)
    args.fps = args.fps or preset.get("fps", 40)
    args.view = args.view or "oblique_view"
    sizes = {
        "n_rg": args.n_rg or preset.get("n_rg",
                                        cpg_params.get("n_rg_final", 20)),
        "n_mn": args.n_mn or preset.get("n_mn",
                                        cpg_params.get("n_mn_final", 20)),
        "n_int": args.n_int or preset.get("n_int",
                                          cpg_params.get("n_int_final", 15)),
        "n_aff": args.n_aff or preset.get("n_aff",
                                          cpg_params.get("n_aff_final", 20)),
        "n_bs": args.n_bs or preset.get("n_bs",
                                        cpg_params.get("n_bs_final", 30)),
    }
    for key, value in sizes.items():
        setattr(args, key, value)

    from persona.game import GameConfig
    result = run(
        duration_s=args.sim_time, view=args.view, width=args.width,
        height=args.height, fps=args.fps, out_dir=args.out,
        w_scale=args.w_scale, ref_hz=args.ref_hz, belt_mps=args.belt_mps,
        belt_start_s=args.belt_start_s, belt_ramp_s=args.belt_ramp_s,
        render=args.render, budget_s=args.budget_s,
        pose_max_s=args.pose_max_s, belt_mode=args.belt,
        snapshot_s=args.snapshot_s, tau_mem_s=args.tau_mem_s,
        realize_every_k=args.realize_every_k,
        resolution_ms=args.resolution_ms,
        cut_multisite=args.cut_multisite,
        game=(GameConfig.current_best() if args.game == "selected" else None),
        seed=args.seed, lr_mode=args.lr_mode, lr_start=args.lr_start,
        lr_end=args.lr_end, nest_threads=args.nest_threads,
        noise_config=args.noise_config, trace_strategy=args.trace_strategy,
        trace_float32=args.trace_float32,
        snapshot_strategy=args.snapshot_strategy,
        loss_stop=args.loss_stop, loss_window=args.loss_window,
        accept_stop=args.accept_stop, accept_margin=args.accept_margin,
        accept_coverage=args.accept_coverage,
        accept_window=args.accept_window, spikes=args.spikes,
        params_snapshot=args.params_snapshot,
        params_snapshot_s=args.params_snapshot_s,
        cpg_params=cpg_params, model_params=model_params,
        snapshot_args=args, **sizes)
    return result


if __name__ == "__main__":
    main()
