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
        # Shade stance (light) for each leg so phase structure is visible.
        self._shade_phases(ax, t, getattr(log, "phase_L", None), "red")
        self._shade_phases(ax, t, getattr(log, "phase_R", None), "blue")
        ax.plot(t, log.ankle_target_L, "k--", label="target L")
        ax.plot(t, log.ankle_model_L, "r-", label="model L")
        ax.plot(t, log.ankle_target_R, "b--", label="target R")
        ax.plot(t, log.ankle_model_R, "g-", label="model R")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("ankle angle [rad]")
        ax.set_title("Ankle tracking: model vs target (shaded = stance)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)
        self.figures["tracking"] = fig
        return fig

    @staticmethod
    def _shade_phases(ax, t, phase, color):
        if phase is None or len(t) == 0:
            return
        phase = list(phase)
        n = min(len(phase), len(t))
        step = float(t[1] - t[0]) if len(t) > 1 else 1.0
        i = 0
        while i < n:
            if phase[i] == "stance":
                j = i
                while j < n and phase[j] == "stance":
                    j += 1
                ax.axvspan(t[i], t[j - 1] + step, color=color, alpha=0.10, lw=0)
                i = j
            else:
                i += 1

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

    def plot_param_shift(self, agent, top_n: int = 10):
        """
        Gaussian before/after for the `top_n` most influential parameters:
        for each parameter, overlay the initial and final N(mean, sigma_ctrl)
        distributions so the shift from one distribution to another is visible.
        """
        means = agent.history["means"]
        sigmas = agent.history["sigmas_ctrl"]
        if not means:
            return None
        names = list(means.keys())
        n_steps = len(next(iter(means.values())))
        # influence = relative shift + change in spread
        score = {}
        for k in names:
            m = np.asarray(means[k], dtype=float)
            s0 = np.asarray(sigmas[k], dtype=float)
            shift = abs(m[-1] - m[0]) / (abs(m[0]) + 1e-6)
            spread = abs(s0[-1] - s0[0]) / (abs(s0[0]) + 1e-6)
            score[k] = shift + spread
        top = sorted(names, key=lambda k: score[k], reverse=True)[:top_n]
        ncol = 5
        nrow = int(np.ceil(len(top) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.6 * nrow))
        axes = np.atleast_1d(axes).ravel()
        for ax, k in zip(axes, top):
            m = np.asarray(means[k], dtype=float)
            s = np.asarray(sigmas[k], dtype=float)
            for mu, sig, lab, c in ((m[0], s[0], "initial", "#4C72B0"),
                                    (m[-1], s[-1], "final", "#DD8452")):
                xs = np.linspace(mu - 4 * abs(sig), mu + 4 * abs(sig), 200)
                pdf = np.exp(-0.5 * ((xs - mu) / sig) ** 2) / (abs(sig) * np.sqrt(2 * np.pi))
                ax.plot(xs, pdf, color=c, lw=1.6, label=lab)
                ax.axvline(mu, color=c, ls="--", lw=1.0)
            ax.set_title(k.replace("conn_", "").replace("_weight", ""), fontsize=7)
            ax.set_yticks([])
            ax.legend(fontsize=6)
        for ax in axes[len(top):]:
            ax.axis("off")
        fig.suptitle(f"Parameter distributions: initial vs final ({len(top)} most changed)",
                     fontsize=12)
        fig.tight_layout()
        self.figures["param_shift"] = fig
        return fig

    def plot_fisher(self, agent):
        """Heat-map of the correlation-style Fisher matrix at the end of the run."""
        F = agent.strategy.compute_correlation_matrix()
        names = agent.strategy.all_names
        show = min(len(names), 60)  # keep the heat-map legible
        # pick the highest-variance (most active) features
        X = np.asarray(agent.strategy.perturbation_history, dtype=float)
        if X.size:
            idx = np.argsort(-X.std(axis=0))[:show]
        else:
            idx = np.arange(show)
        Fsub = F[np.ix_(idx, idx)]
        labels = [names[i].replace("conn_", "").replace("_weight", "")[:14] for i in idx]
        fig, ax = plt.subplots(figsize=(10, 9))
        im = ax.imshow(Fsub, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(idx)))
        ax.set_yticks(range(len(idx)))
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_title(f"Fisher (correlation) matrix, final — {len(idx)} most active features")
        fig.colorbar(im, ax=ax, label="correlation")
        fig.tight_layout()
        self.figures["fisher"] = fig
        return fig

    def plot_influence(self, agent):
        """Bar chart of the per-parameter impact |Cov(Δθ, A)| on the tracked error."""
        infl = np.asarray(agent.strategy.influence(), dtype=float)
        names = list(agent.strategy.all_names)
        order = np.argsort(-infl)[:min(len(infl), 25)]
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(range(len(order)), infl[order], color="tab:purple")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(
            [names[i].replace("conn_", "").replace("_weight", "")[:22] for i in order],
            rotation=90, fontsize=7)
        ax.set_ylabel("influence  |Cov(Δθ, A)|  (norm.)")
        ax.set_title("Per-parameter impact on the tracked error (top 25)")
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        self.figures["influence"] = fig
        return fig

    def plot_metric(self, agent):
        """
        Heat-map of the metric F = Cov(Δθ) + λ_F I. The diagonal is the
        perturbation variance (the natural-gradient preconditioner); the
        per-parameter impact on the error is |Cov(Δθ, A)| — see `influence()`.
        """
        F = np.asarray(agent.strategy.compute_metric()[0], dtype=float)
        names = agent.strategy.all_names
        show = min(len(names), 60)
        Fsub = F[:show, :show]
        labels = [names[i].replace("conn_", "").replace("_weight", "")[:14]
                  for i in range(show)]
        fig, ax = plt.subplots(figsize=(10, 9))
        vmax = float(np.max(np.abs(Fsub))) or 1.0
        im = ax.imshow(Fsub, cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(show))
        ax.set_yticks(range(show))
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_title("Metric F = Cov(Δθ) + λ_F I — diagonal = perturbation variance;"
                     " impact = |Cov(Δθ, A)|")
        fig.colorbar(im, ax=ax, label="Cov(Δθ)")
        fig.tight_layout()
        self.figures["metric"] = fig
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