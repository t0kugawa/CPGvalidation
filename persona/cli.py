"""Simulator-independent CLI for the PERSONA experiment."""
import argparse

from persona.runtime import validate_threads


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Run the PERSONA personalization experiment.")
    ap.add_argument("--steps", type=int, default=200,
                    help="run horizon in control steps (also the cosine horizon)")
    ap.add_argument("--tau-mem-s", type=float, default=2.0,
                    help="trace/credit memory time constant in seconds "
                         "(beta = 1 - dt/tau_mem)")
    ap.add_argument("--realize-every-k", type=int, default=1,
                    help="sample + realize + record the network every k steps "
                         "(strategy still updates every dt)")
    ap.add_argument("--cut-multisite", action="store_true",
                    help="Option B: one CUT Poisson relay and STDP projection "
                         "per plantar site (changes the PERSONA dimension)")
    ap.add_argument("--lr-mode", choices=("constant", "cosine"),
                    default="constant",
                    help="constant learning rate or a cosine decay over --steps")
    ap.add_argument("--lr-start", type=float, default=0.03,
                    help="initial mean learning rate (also the constant rate)")
    ap.add_argument("--lr-end", type=float, default=None,
                    help="target mean learning rate; required for cosine")
    ap.add_argument("--game", choices=("none", "selected"), default="selected",
                    help="conserved-stack game; 'selected' uses "
                         "GameConfig.current_best()")
    ap.add_argument("--seed", type=int, default=0,
                    help="run seed (also the NEST rng_seed)")
    ap.add_argument("--nest-threads", type=validate_threads, default=1,
                    help="NEST local threads for the single CPG/MuJoCo run; "
                         "this does not build replicas")
    ap.add_argument("--noise-config", default="all",
                    help="producer filter: all|none|poisson|stdp|one-relay:<name>")
    ap.add_argument("--trace-strategy", action="store_true",
                    help="record the applied-sample strategy trace (float64)")
    ap.add_argument("--trace-float32", action="store_true",
                    help="store the trace as float32 (approximate replay)")
    ap.add_argument("--snapshot-strategy", type=int, default=0,
                    help="take a strategy snapshot every N steps (0 = off)")
    ap.add_argument("--training", choices=("manual", "evo"), default="manual",
                    help="manual registry players or PERSONA-Evo")
    ap.add_argument("--player-map", default=None,
                    help="saved PlayerPartition JSON for manual mode")
    ap.add_argument("--evo-players", type=int, default=24)
    ap.add_argument("--evo-generations", type=int, default=6)
    ap.add_argument("--evo-steps", type=int, default=400)
    ap.add_argument("--evo-eval-window", type=int, default=50)
    ap.add_argument("--evo-init", choices=("territory", "random"),
                    default="territory")
    ap.add_argument("--evo-stop", choices=("partition", "nash", "both"),
                    default="both")
    ap.add_argument("--evo-patience", type=int, default=2)
    ap.add_argument("--evo-nash-tol", type=float, default=1e-3)
    ap.add_argument("--evo-impact-eps", type=float, default=0.0)
    ap.add_argument("--evo-min-players", type=int, default=2)
    ap.add_argument("--evo-seed", type=int, default=None)
    ap.add_argument("--evo-out", default=None)
    ap.add_argument("--evo-fresh-process", action=argparse.BooleanOptionalAction,
                    default=True)
    ap.add_argument("--evo-final-run", action=argparse.BooleanOptionalAction,
                    default=True)
    return ap


def validate_args(args) -> None:
    """Validate cross-flag rules; raises SystemExit like argparse."""
    if args.training == "evo" and args.game == "none":
        raise SystemExit("--training evo requires --game selected")
    if args.training == "evo" and args.player_map:
        raise SystemExit("--player-map cannot be combined with --training evo")
    if args.evo_players < 1:
        raise SystemExit("--evo-players must be >= 1")
    if args.evo_generations < 1:
        raise SystemExit("--evo-generations must be >= 1")
    if args.evo_steps < 1:
        raise SystemExit("--evo-steps must be >= 1")
    if args.evo_eval_window < 1:
        raise SystemExit("--evo-eval-window must be >= 1")
    if args.evo_patience < 1:
        raise SystemExit("--evo-patience must be >= 1")
    if args.evo_min_players < 1:
        raise SystemExit("--evo-min-players must be >= 1")


def parse_cli(argv=None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.lr_mode == "cosine" and args.lr_end is None:
        parser.error("--lr-end is required with --lr-mode cosine")
    validate_args(args)
    return args
