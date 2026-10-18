"""Rebuild the validation figures from a saved run HDF5 file.

Usage: python read_h5.py --h5 <run.h5> --out <fig_dir>
"""
import argparse
import os
from types import SimpleNamespace

import numpy as np

from persona.h5io import load_run
from persona.validation import ValidationReport


class _StrategyView:
    """Minimal strategy view over the saved influence history."""

    def __init__(self, influence, names):
        self._influence = influence
        self.param_names = list(names)
        self.all_names = list(names)

    def influence(self):
        if self._influence is None:
            return np.ones(len(self.param_names))
        arr = np.asarray(self._influence, dtype=float)
        tail = arr[-min(len(arr), 100):]
        return tail.mean(axis=0)


def make_figures(h5_path, out_dir):
    data, meta = load_run(h5_path)
    log = SimpleNamespace(**data["log"])
    agent = SimpleNamespace(
        history={"means": data["agent"]["means"],
                 "sigmas": data["agent"]["sigmas"]},
        strategy=_StrategyView(data["agent"].get("influence"),
                               data["agent"].get("all_names")
                               or data["agent"].get("param_names", [])),
    )
    rep = ValidationReport()
    rep.plot_tracking(log)
    try:
        rep.plot_model_vs_target(log)
    except Exception as e:  # noqa: BLE001
        print(f"model_vs_target: {e}")
    rep.plot_learning(log)
    for name, fn in (("influence", lambda: rep.plot_influence(agent)),
                     ("param_shift", lambda: rep.plot_param_shift(agent, top_n=10))):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            print(f"{name}: {e}")
    rep.save_all(out_dir)
    print("figures written to", out_dir, "from", h5_path)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.splitext(args.h5)[0] + "_figures"
    make_figures(args.h5, out)


if __name__ == "__main__":
    main()