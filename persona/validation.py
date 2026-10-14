import os
from typing import Any, Dict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class ValidationReport:
    """
    Produces the matplotlib validation figures for a personalization run and
    the biological-acceptance verdict (global parameters within the rat
    distribution; ankle angles within `margin` for `coverage` of the phase).
    """
    def __init__(self):
        self.figures: Dict[str, Any] = {}

    def plot_tracking(self, log):
        fig, ax = plt.subplots(figsize=(12, 5))
        t = np.asarray(log.time_ms) / 1000.0
        ax.plot(t, log.ankle_target_L, "k--", label="target L")
        ax.plot(t, log.ankle_model_L, "r-", label="model L")
        ax.plot(t, log.ankle_target_R, "b--", label="target R")
        ax.plot(t, log.ankle_model_R, "g-", label="model R")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("ankle angle [rad]")
        ax.set_title("Ankle tracking: model vs target")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)
        self.figures["tracking"] = fig
        return fig

    def plot_inputs(self, log):
        fig, ax = plt.subplots(figsize=(12, 4))
        t = np.asarray(log.time_ms) / 1000.0
        ax.plot(t, log.loss, color="tab:brown", label="loss")
        ax.plot(t, log.advantage, color="tab:blue", label="advantage")
        ax.axhline(0.0, color="k", ls=":", alpha=0.6)
        ax.set_xlabel("time [s]")
        ax.set_title("Learning signals")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)
        self.figures["inputs"] = fig
        return fig

    def plot_learning(self, log):
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        t = np.asarray(log.time_ms) / 1000.0
        axes[0].plot(t, log.loss, color="tab:brown")
        axes[0].set_ylabel("loss")
        axes[0].grid(alpha=0.3)
        axes[1].plot(t, log.advantage, color="tab:blue")
        axes[1].axhline(0.0, color="k", ls=":", alpha=0.6)
        axes[1].set_xlabel("time [s]")
        axes[1].set_ylabel("advantage")
        axes[1].grid(alpha=0.3)
        self.figures["learning"] = fig
        return fig

    def plot_distribution(self, metrics: Dict[str, float], stats):
        keys = [k for k in ("duty_factor_L", "duty_factor_R", "double_support")
                if stats is not None and k in metrics]
        if not keys:
            keys = ["duty_factor_L"]
        fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 4))
        axes = np.atleast_1d(axes)
        field_map = {"duty_factor_L": "LH_duty_factor",
                     "duty_factor_R": "RH_duty_factor",
                     "double_support": "LH_double_support"}
        for ax, k in zip(axes, keys):
            if stats is not None:
                dist = stats.distribution(field_map.get(k, k))
                ax.hist(dist, bins=10, alpha=0.5, color="tab:gray", label="rats")
            ax.axvline(metrics.get(k, 0.0), color="red", lw=2, label="model")
            ax.set_title(k)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
        self.figures["distribution"] = fig
        return fig

    def save_all(self, path):
        os.makedirs(path, exist_ok=True)
        for name, fig in self.figures.items():
            fig.savefig(os.path.join(path, f"{name}.png"), dpi=150, bbox_inches="tight")
            plt.close(fig)


def biological_acceptance(metrics: Dict[str, float],
                          stats: Any,
                          angle_errors: np.ndarray,
                          phase_mask: np.ndarray,
                          margin: float = 0.05,
                          coverage: float = 0.70) -> Dict[str, Any]:
    """
    The model is acceptable if its global locomotion parameters fall in the
    rat distribution and its ankle angles stay within `margin` for at least
    `coverage` of the locomotion phase.
    """
    from loader.stats import in_distribution

    err = np.asarray(angle_errors, dtype=np.float64)
    mask = np.asarray(phase_mask, dtype=bool)
    if err.size and mask.size and mask.any():
        within = (err[mask] <= margin)
        angle_coverage = float(np.mean(within))
    else:
        angle_coverage = 0.0

    if stats is None:
        global_ok = True
    else:
        checks = []
        field_map = {"duty_factor_L": "LH_duty_factor",
                     "duty_factor_R": "RH_duty_factor"}
        for k, field in field_map.items():
            if k in metrics:
                checks.append(in_distribution(metrics[k], stats.distribution(field)))
        global_ok = all(checks) if checks else True

    return {
        "global_in_distribution": bool(global_ok),
        "angle_coverage": angle_coverage,
        "acceptable": bool(global_ok and angle_coverage >= coverage),
    }