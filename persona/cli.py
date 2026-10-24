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
    ap.add_argument("--threads", type=validate_threads, default=1,
                    help="NEST local threads and the number of independent "
                         "PERSONA/CPG replicas. Each replica samples different "
                         "parameters; strategies merge at reference phase "
                         "changes. Results are reproducible for a fixed count.")
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
    return ap


def parse_cli(argv=None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.lr_mode == "cosine" and args.lr_end is None:
        parser.error("--lr-end is required with --lr-mode cosine")
    return args
