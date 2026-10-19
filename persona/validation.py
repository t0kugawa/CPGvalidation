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

    def plot_model_vs_target(self, log):
        """
        Diagnostic comparison of the MODEL limb vs the TARGET (reference) limb in
        MuJoCo: ankle/knee/hip, SOL/TA drives, Ia/Ib afferents, foot contact, and
        stance/swing (model shaded red, target shaded blue). Answers whether the
        target limb itself makes ground contact and how the two limbs differ.
        """
        t = np.asarray(log.time_ms) / 1000.0

        def col(name):
            v = getattr(log, name, None)
            return np.asarray(v, dtype=float) if v is not None and len(v) == len(t) else None

        fig, axes = plt.subplots(7, 1, figsize=(12, 17), sharex=True)

        ax = axes[0]
        if col("ankle_target_L") is not None:
            ax.plot(t, col("ankle_target_L"), "k--", label="target traj L")
        ax.plot(t, log.ankle_model_L, "r-", label="model ankle L")
        if col("ankle_ref_L") is not None:
            ax.plot(t, col("ankle_ref_L"), color="magenta", lw=1.0, label="target ankle L")
        ax.set_ylabel("ankle [rad]"); ax.legend(fontsize=7)

        ax = axes[1]
        ax.plot(t, log.knee_model_L, "r-", label="model knee L")
        ax.plot(t, log.hip_model_L, "b-", label="model hip L")
        if col("knee_ref_L") is not None:
            ax.plot(t, col("knee_ref_L"), "r--", lw=1.0, label="target knee L")
        if col("hip_ref_L") is not None:
            ax.plot(t, col("hip_ref_L"), "b--", lw=1.0, label="target hip L")
        ax.set_ylabel("knee/hip [rad]"); ax.legend(fontsize=7)

        ax = axes[2]
        ax.plot(t, log.mus_E_L, "r-", label="SOL drive L")
        ax.plot(t, log.mus_F_L, "b-", label="TA drive L")
        ax.set_ylabel("muscle drive"); ax.legend(fontsize=7)

        ax = axes[3]
        ax.plot(t, log.ia_rate_E_L, "r-", label="Ia SOL L")
        if col("ia_rate_F_L") is not None:
            ax.plot(t, col("ia_rate_F_L"), "b-", label="Ia TA L")
        ax.plot(t, log.ib_rate_E_L, "k-", label="Ib SOL L")
        ax.set_ylabel("afferent [Hz]"); ax.legend(fontsize=7)

        ax = axes[4]
        ax.plot(t, log.contact_m_L, "r-", label="model contact L")
        if col("contact_ref_L") is not None:
            ax.plot(t, col("contact_ref_L"), color="magenta", lw=1.0, label="target contact L")
        self._shade_phases(ax, t, getattr(log, "phase_L", None), "red")
        self._shade_phases(ax, t, getattr(log, "phase_ref_L", None), "blue")
        ax.set_ylabel("contact [N]"); ax.legend(fontsize=7)

        ax = axes[5]
        ax.plot(t, log.cut_L, color="tab:orange", label="CUT L [Hz]")
        ax.set_ylabel("CUT [Hz]"); ax.legend(fontsize=7)

        ax = axes[6]
        ax.plot(t, log.foot_z_m_L, "r-", label="model foot z L")
        if col("foot_z_ref_L") is not None:
            ax.plot(t, col("foot_z_ref_L"), color="magenta", lw=1.0, label="target foot z L")
        ax.axhline(0.0, color="k", ls=":", alpha=0.6)
        ax.set_ylabel("foot z [m]"); ax.set_xlabel("time [s]"); ax.legend(fontsize=7)

        for a in axes:
            a.grid(alpha=0.3)

        def stance_frac(phase):
            if phase is None or len(phase) == 0:
                return float("nan")
            return 100.0 * sum(1 for p in phase if p == "stance") / len(phase)

        fig.suptitle(
            "Model vs target limb (L): stance model "
            f"{stance_frac(getattr(log, 'phase_L', None)):.1f}% / target "
            f"{stance_frac(getattr(log, 'phase_ref_L', None)):.1f}%", fontsize=12)
        fig.tight_layout()
        self.figures["model_vs_target"] = fig
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
        axes[1].plot(t, log.advantage, color="tab:blue", label="mean advantage")
        adv_l = getattr(log, "adv_L", None)
        adv_r = getattr(log, "adv_R", None)
        if adv_l is not None and len(adv_l):
            axes[1].plot(t, adv_l, color="tab:red", lw=0.8, alpha=0.8, label="A L (left ankle)")
        if adv_r is not None and len(adv_r):
            axes[1].plot(t, adv_r, color="tab:green", lw=0.8, alpha=0.8, label="A R (right ankle)")
        axes[1].axhline(0.0, color="k", ls=":", alpha=0.6)
        axes[1].set_xlabel("time [s]")
        axes[1].set_ylabel("advantage")
        axes[1].legend(fontsize=8)
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
        for each parameter, overlay the initial and final N(mean, sigma)
        distributions so the shift from one distribution to another is visible.
        """
        means = agent.history["means"]
        sigmas = agent.history["sigmas"]
        if not means:
            return None
        names = list(means.keys())
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

    @staticmethod
    def _channel_labels(strategy):
        """Row labels of F/Xbar/G: sites [μ; σ] in param_names order, then noise."""
        labels = []
        for g in list(getattr(strategy, "param_names", [])):
            labels += [f"{g} μ", f"{g} σ"]
        labels += [f"noise {k}" for k in list(getattr(strategy, "noise_names", []))]
        return labels

    def plot_fisher(self, agent):
        """Heat-map of F = β-EWMA score covariance (diagonal = channel activity)."""
        F = np.asarray(agent.strategy.F, dtype=float)
        names = self._channel_labels(agent.strategy)
        show = min(len(names), 60)  # keep the heat-map legible
        # pick the most active channels (largest diagonal of F)
        d = np.diag(F)
        if d.size:
            idx = np.argsort(-d)[:show]
        else:
            idx = np.arange(show)
        Fsub = F[np.ix_(idx, idx)]
        labels = [names[i].replace("conn_", "").replace("_weight", "")[:14] for i in idx]
        fig, ax = plt.subplots(figsize=(10, 9))
        vmax = float(np.max(np.abs(Fsub))) or 1.0
        im = ax.imshow(Fsub, cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(idx)))
        ax.set_yticks(range(len(idx)))
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_title(f"Fisher F = score covariance, final — {len(idx)} most active channels")
        fig.colorbar(im, ax=ax, label="F")
        fig.tight_layout()
        self.figures["fisher"] = fig
        return fig

    def plot_influence(self, agent):
        """Bar chart of the per-channel impact |G| row norms (normalised)."""
        infl = np.asarray(agent.strategy.influence(), dtype=float)
        names = self._channel_labels(agent.strategy)
        order = np.argsort(-infl)[:min(len(infl), 25)]
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(range(len(order)), infl[order], color="tab:purple")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(
            [names[i].replace("conn_", "").replace("_weight", "")[:22] for i in order],
            rotation=90, fontsize=7)
        ax.set_ylabel("influence  |G| (norm.)")
        ax.set_title("Per-channel impact on the tracked error (top 25)")
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        self.figures["influence"] = fig
        return fig

    def plot_metric(self, agent):
        """
        Heat-map of the Fisher F = β-EWMA score covariance; the diagonal is the
        per-channel activity (the natural-gradient preconditioner). The
        per-channel impact on the error is |G| — see `influence()`.
        """
        F = np.asarray(agent.strategy.F, dtype=float)
        names = self._channel_labels(agent.strategy)
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
        ax.set_title("Fisher F = score covariance — diagonal = channel activity;"
                     " impact = |G|")
        fig.colorbar(im, ax=ax, label="F")
        fig.tight_layout()
        self.figures["metric"] = fig
        return fig

    def plot_sensory(self, log):
        """
        Afferent traffic and muscle drives: CUT, Ia (length- and
        velocity-derived rate), Ib (force/activation-derived rate) and the
        extensor/flexor muscle drives, for both limbs.
        """
        t = np.asarray(log.time_ms) / 1000.0

        def col(name):
            v = getattr(log, name, None)
            return None if v is None or not len(v) else np.asarray(v, dtype=float)

        fig, axes = plt.subplots(5, 1, figsize=(13, 14), sharex=True)
        for c, lab, style in (("L", "left", "-"), ("R", "right", "--")):
            v = col(f"cut_{c}")
            if v is not None:
                axes[0].plot(t, v, style, label=f"CUT {lab}", lw=1.0)
        axes[0].set_ylabel("CUT [Hz]")
        axes[0].set_title("Cutaneous afferents (plantar contact)")
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.3)

        for c, lab, style in (("L", "left", "-"), ("R", "right", "--")):
            for m, mlab in (("E", "ext"), ("F", "flex")):
                v = col(f"ia_rate_{m}_{c}")
                if v is not None:
                    axes[1].plot(t, v, style, lw=1.0, label=f"Ia {mlab} {lab}")
        axes[1].set_ylabel("Ia rate [Hz]")
        axes[1].set_title("Ia (muscle spindle: length + stretch velocity)")
        axes[1].legend(fontsize=7, ncol=2)
        axes[1].grid(alpha=0.3)

        for c, lab, style in (("L", "left", "-"), ("R", "right", "--")):
            for m, mlab in (("E", "ext"), ("F", "flex")):
                v = col(f"ib_rate_{m}_{c}")
                if v is not None:
                    axes[2].plot(t, v, style, lw=1.0, label=f"Ib {mlab} {lab}")
        axes[2].set_ylabel("Ib rate [Hz]")
        axes[2].set_title("Ib (Golgi tendon organ: tendon force + activation)")
        axes[2].legend(fontsize=7, ncol=2)
        axes[2].grid(alpha=0.3)

        for c, lab, style in (("L", "left", "-"), ("R", "right", "--")):
            for m, mlab in (("E", "ext"), ("F", "flex")):
                v = col(f"rg_{m.lower()}_{c}")
                if v is not None:
                    axes[3].plot(t, v, style, lw=1.0, label=f"RG {mlab} {lab}")
        axes[3].set_ylabel("RG activity")
        axes[3].set_xlabel("time [s]")
        axes[3].set_title("Rhythm-generator population activity (read + cleared each step)")
        axes[3].legend(fontsize=7, ncol=2)
        axes[3].grid(alpha=0.3)

        for c, lab, style in (("L", "left", "-"), ("R", "right", "--")):
            for m, mlab in (("E", "ext"), ("F", "flex")):
                v = col(f"mus_{m}_{c}")
                if v is not None:
                    axes[4].plot(t, v, style, lw=1.0, label=f"mus {mlab} {lab}")
        axes[4].set_ylabel("muscle drive")
        axes[4].set_xlabel("time [s]")
        axes[4].set_title("Muscle drives sent to MuJoCo")
        axes[4].legend(fontsize=7, ncol=2)
        axes[4].grid(alpha=0.3)

        fig.tight_layout()
        self.figures["sensory"] = fig
        return fig

    def plot_stdp(self, log):
        """
        STDP trace from the weight recorders and the live synapse weights:
        mean weight per plastic projection, the per-step mean weight change and
        the number of STDP updates per step.
        """
        stdp = getattr(log, "stdp", None) or {}
        if not stdp or not stdp.get("weight_mean"):
            return None
        t = np.asarray(log.time_ms) / 1000.0
        fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=False)

        def legend(ax):
            h, _ = ax.get_legend_handles_labels()
            if h:
                ax.legend(fontsize=7)

        ts = np.asarray(stdp.get("sample_time_ms", []), dtype=float) / 1000.0
        for key, vals in (stdp.get("weight_mean") or {}).items():
            axes[0].plot(ts[:len(vals)], np.asarray(vals, dtype=float), lw=1.0,
                         label=f"{key} (mean)")
            lo = np.asarray(stdp.get("weight_min", {}).get(key, []), dtype=float)
            hi = np.asarray(stdp.get("weight_max", {}).get(key, []), dtype=float)
            if lo.size and hi.size:
                axes[0].fill_between(ts[:len(lo)], lo, hi, alpha=0.10)
        axes[0].set_ylabel("synaptic weight")
        axes[0].set_title("STDP weights (mean over each plastic projection; band = min..max)")
        legend(axes[0])
        axes[0].grid(alpha=0.3)

        for key, vals in (stdp.get("delta") or {}).items():
            v = np.asarray(vals, dtype=float)
            axes[1].plot(t[:len(v)], v, lw=0.8, label=f"{key}")
        axes[1].set_xlabel("time [s]")
        axes[1].set_ylabel("mean Δw / step")
        axes[1].set_title("STDP weight-recorder: mean weight change per step")
        legend(axes[1])
        axes[1].grid(alpha=0.3)

        for key, vals in (stdp.get("n") or {}).items():
            v = np.asarray(vals, dtype=float)
            axes[2].plot(t[:len(v)], v, lw=0.8, label=f"{key}")
        axes[2].set_xlabel("time [s]")
        axes[2].set_ylabel("updates / step")
        axes[2].set_title("STDP update count per step (weight-recorder events)")
        legend(axes[2])
        axes[2].grid(alpha=0.3)

        fig.tight_layout()
        self.figures["stdp"] = fig
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
        "duty_factor_in_distribution": bool(global_ok),
        "angle_coverage": angle_coverage,
        "acceptable": bool(global_ok and angle_coverage >= coverage),
    }