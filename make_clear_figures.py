"""
Clear figures for a PERSONA / CPG / MuJoCo validation run.

Reads run.h5 and writes five figures into ./figures.  All time-series panels
are drawn over an expanded 60-70 s window (the full run is 120 s) so the
~5 Hz locomotor structure is actually visible instead of being compressed
into a solid band.

    1_activity_MUS_CUT_IA_IB.png
    2_stdp_weights.png
    3_persona_advantage_loss_distributions.png
    4_angles_target_controlled.png
    5_fisher_strategy_heatmap.png
"""

from __future__ import annotations

import os
import re
import time

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from scipy.ndimage import uniform_filter1d

# ----------------------------------------------------------------------------
plt.rcParams.update({
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11.5,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.solid_capstyle": "round",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

BLUE = "#1f77b4"
ORANGE = "#ff7f0e"
RED = "#d62728"
GREEN = "#2ca02c"
BROWN = "#8c564b"
GREY = "#555555"

OUT = "figures"
T0, T1 = 60.0, 70.0          # expanded time window shown in every timeline


# ----------------------------------------------------------------------------
def decode(value):
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)


def rolling(x, samples):
    n = max(1, int(samples))
    if n <= 1:
        return np.asarray(x, dtype=float)
    return uniform_filter1d(np.asarray(x, dtype=float), size=n, mode="nearest")


def short_name(name: str, limit: int = 28) -> str:
    name = (name.replace("conn_", "")
                .replace("_weight", "")
                .replace("producer:", "")
                .replace(":poisson", "")
                .replace(":stdp", ""))
    return name if len(name) <= limit else name[: limit - 1] + "…"


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    # Windows can briefly refuse to overwrite a PNG that an image viewer has
    # just read; retry instead of losing the whole run.
    for attempt in range(8):
        try:
            fig.savefig(path, dpi=220)
            break
        except OSError:
            if attempt == 7:
                raise
            time.sleep(1.0)
    plt.close(fig)
    print("wrote", path)


def load_all(h5_path="run.h5"):
    with h5py.File(h5_path, "r") as f:
        log = {k: f["log"][k][()] for k in f["log"]}
        for k, v in list(log.items()):
            if getattr(v, "dtype", None) is not None and v.dtype.kind in ("S", "O"):
                log[k] = np.array([decode(x) for x in v])
        agent = {k: f["agent"][k][()] for k in
                 ("fisher", "feature_scales", "trace_final", "credit_final",
                  "influence", "loss", "advantage_vector", "error_vector")}
        agent["means"] = {k: f["agent/means"][k][()] for k in f["agent/means"]}
        agent["sigmas"] = {k: f["agent/sigmas"][k][()] for k in f["agent/sigmas"]}
        for key in ("param_names", "noise_names", "noise_scales", "all_names"):
            agent[key] = [decode(x) for x in f["agent"].attrs.get(key, [])]
        stdp = {}
        for grp in f["stdp"]:
            if isinstance(f["stdp"][grp], h5py.Group):
                stdp[grp] = {k: f["stdp"][grp][k][()] for k in f["stdp"][grp]}
            else:
                stdp[grp] = f["stdp"][grp][()]
    return log, agent, stdp


# ----------------------------------------------------------------------------
# 1. MUS / CUT / Ia / Ib
# ----------------------------------------------------------------------------
def figure_activity(log):
    t = np.asarray(log["time_ms"], float) / 1000.0
    m = (t >= T0) & (t <= T1)
    tw = t[m]

    fig = plt.figure(figsize=(24, 14))
    gs = GridSpec(4, 1, figure=fig, height_ratios=[1.15, 0.85, 1.0, 1.0],
                  hspace=0.40,
                  left=0.055, right=0.99, top=0.92, bottom=0.06)

    def draw(ax, specs, ylabel, title, ylim, ncol=4, smooth=1):
        for key, lab, color, ls in specs:
            y = np.asarray(log[key], float)[m]
            if smooth > 1:
                y = rolling(y, smooth)
            ax.plot(tw, y, color=color, ls=ls, lw=1.15, label=lab)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", pad=6)
        ax.set_ylim(*ylim)
        ax.set_xlim(T0, T1)
        ax.legend(loc="upper right", ncol=ncol, framealpha=0.9,
                  handlelength=1.8, columnspacing=1.0, fontsize=8)

    # All four muscles on one timeline, as before.
    ax_mus = fig.add_subplot(gs[0, 0])
    draw(ax_mus, [
        ("mus_E_L", "ext L", BLUE, "-"),
        ("mus_F_L", "flex L", ORANGE, "-"),
        ("mus_E_R", "ext R", BLUE, "--"),
        ("mus_F_R", "flex R", ORANGE, "--"),
    ], "drive", "MUS", (-0.03, 1.12), ncol=4, smooth=3)
    ax_mus.text(T0 + 0.15, 1.06, "ext L / flex L / ext R clipped at 1.0",
                fontsize=8, color=GREY, va="top")

    ax_cut = fig.add_subplot(gs[1, 0])
    draw(ax_cut, [
        ("cut_L", "left", BLUE, "-"),
        ("cut_R", "right", RED, "--"),
    ], "Hz", "CUT", (-3, 126), ncol=2)

    ax_ia = fig.add_subplot(gs[2, :], sharex=ax_cut)
    draw(ax_ia, [
        ("ia_rate_E_L", "ext L", BLUE, "-"),
        ("ia_rate_F_L", "flex L", ORANGE, "-"),
        ("ia_rate_E_R", "ext R", BLUE, "--"),
        ("ia_rate_F_R", "flex R", ORANGE, "--"),
    ], "Hz", "Ia", (0, 150))

    ax_ib = fig.add_subplot(gs[3, :], sharex=ax_cut)
    draw(ax_ib, [
        ("ib_rate_E_L", "ext L", BLUE, "-"),
        ("ib_rate_F_L", "flex L", ORANGE, "-"),
        ("ib_rate_E_R", "ext R", BLUE, "--"),
        ("ib_rate_F_R", "flex R", ORANGE, "--"),
    ], "Hz", "Ib", (0, 195))
    ax_ib.set_xlabel("time [s]")

    for ax in (ax_cut, ax_ia, ax_ib):
        ax.set_xticks(np.arange(T0, T1 + 0.1, 1))
        ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(1))
        ax.tick_params(axis="x", which="minor", length=2)

    fig.suptitle("MUS / CUT / Ia / Ib activity", fontsize=18, y=0.972)
    fig.text(0.5, 0.012, "all panels: 60–70 s of the 120 s run", ha="center",
             fontsize=10, color=GREY)
    save(fig, "1_activity_MUS_CUT_IA_IB.png")


# ----------------------------------------------------------------------------
# 2. STDP weights
# ----------------------------------------------------------------------------
def figure_stdp(stdp):
    ts = np.asarray(stdp["sample_time_ms"], float) / 1000.0
    t = np.asarray(stdp["time_ms"], float) / 1000.0
    # The plastic weights move most during the first 20 s; show that window
    # so the STDP transients are legible (the final distributions show the end).
    s0, s1 = 0.0, 20.0
    m = (ts >= s0) & (ts <= s1)
    mt = (t >= s0) & (t <= s1)
    keys = list(stdp["weight_mean"].keys())
    colors = {k: c for k, c in zip(keys, plt.cm.tab10(np.linspace(0, 1, 10)))}
    cut_keys = [k for k in keys if k.startswith("cut")]
    ia_keys = [k for k in keys if k.startswith("ia")]

    fig = plt.figure(figsize=(24, 14.5))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1.15, 1.0, 1.0],
                  hspace=0.42, wspace=0.16,
                  left=0.055, right=0.985, top=0.92, bottom=0.065)

    # cut pathways -----------------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    for k in cut_keys:
        mu = np.asarray(stdp["weight_mean"][k], float)
        lo = np.asarray(stdp["weight_min"][k], float)
        hi = np.asarray(stdp["weight_max"][k], float)
        n = min(len(ts), len(mu), len(lo), len(hi))
        w = m[:n]
        ax.fill_between(ts[:n][w], lo[:n][w], hi[:n][w], color=colors[k],
                        alpha=0.16, lw=0)
        ax.plot(ts[:n][w], mu[:n][w], color=colors[k], lw=2.0, label=k)
    ax.set_xlim(s0, s1)
    ax.set_ylabel("weight")
    ax.set_title("Cutaneous pathway", loc="left")
    ax.legend(loc="center right", framealpha=0.9)

    # Ia pathways ------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    for k in ia_keys:
        mu = np.asarray(stdp["weight_mean"][k], float)
        lo = np.asarray(stdp["weight_min"][k], float)
        hi = np.asarray(stdp["weight_max"][k], float)
        n = min(len(ts), len(mu), len(lo), len(hi))
        w = m[:n]
        ax.fill_between(ts[:n][w], lo[:n][w], hi[:n][w], color=colors[k],
                        alpha=0.13, lw=0)
        ax.plot(ts[:n][w], mu[:n][w], color=colors[k], lw=2.0, label=k)
    ax.set_xlim(s0, s1)
    ax.set_ylabel("weight")
    ax.set_title("Ia pathway", loc="left")
    ax.legend(loc="upper right", ncol=2, framealpha=0.9)

    # final distributions ----------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    data = [np.asarray(stdp["final_weights"][k], float) for k in keys]
    parts = ax.violinplot(data, positions=np.arange(len(keys)),
                          widths=0.85, showextrema=False, showmedians=False)
    for body, k in zip(parts["bodies"], keys):
        body.set_facecolor(colors[k])
        body.set_edgecolor(colors[k])
        body.set_alpha(0.6)
    for i, vals in enumerate(data):
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        ax.vlines(i, q1, q3, color="k", lw=3.0, alpha=0.75)
        ax.plot(i, med, "o", color="white", mec="k", ms=5, zorder=5)
        ax.text(i, med * 1.18, f"{med:.1f}", ha="center", va="bottom",
                fontsize=8.5)
    ax.set_yscale("log")
    ax.set_ylim(2, 220)
    ax.set_xticks(np.arange(len(keys)))
    ax.set_xticklabels([short_name(k, 18) for k in keys], rotation=25,
                       ha="right", fontsize=9)
    ax.set_ylabel("final weight (log)")
    ax.set_title("Final distributions", loc="left")

    # drift ------------------------------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    for k in keys:
        v = np.asarray(stdp["delta"][k], float)
        ax.plot(t[mt], rolling(v, 51)[mt], color=colors[k], lw=1.5, label=k)
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.set_xlim(s0, s1)
    ax.set_ylabel("mean Δw / step")
    ax.set_title("Weight drift (0.5 s mean)", loc="left")
    ax.legend(loc="lower right", ncol=2, fontsize=7.5, framealpha=0.9)

    # event count ------------------------------------------------------------
    ax = fig.add_subplot(gs[2, :])
    for k in keys:
        v = np.asarray(stdp["n"][k], float)
        ax.plot(t[mt], rolling(v, 31)[mt], color=colors[k], lw=1.6, label=k)
    ax.set_xlim(s0, s1)
    ax.set_ylabel("updates / step")
    ax.set_xlabel("time [s]")
    ax.set_title("Update count", loc="left")
    ax.legend(loc="upper right", ncol=6, framealpha=0.9, fontsize=8)

    fig.suptitle("STDP weights", fontsize=18, y=0.972)
    fig.text(0.5, 0.012, "window 0–20 s of the 120 s run", ha="center",
             fontsize=10, color=GREY)
    save(fig, "2_stdp_weights.png")


# ----------------------------------------------------------------------------
# 3. PERSONA
# ----------------------------------------------------------------------------
def figure_persona(log, agent):
    t = np.asarray(log["time_ms"], float) / 1000.0
    m = (t >= T0) & (t <= T1)
    tw = t[m]

    fig = plt.figure(figsize=(25, 22))
    gs = GridSpec(5, 4, figure=fig, height_ratios=[1.0, 1.0, 1.15, 1.15, 1.15],
                  hspace=0.42, wspace=0.30,
                  left=0.06, right=0.985, top=0.93, bottom=0.055)

    # loss -------------------------------------------------------------------
    ax = fig.add_subplot(gs[0, :])
    loss = np.asarray(log["loss"], float)
    ax.plot(tw, loss[m], color=BROWN, lw=0.7, alpha=0.45, label="loss")
    ax.plot(tw, rolling(loss, 25)[m], color=BROWN, lw=2.2, label="250 ms mean")
    ax.set_xlim(T0, T1)
    ax.set_ylabel("loss")
    ax.set_title("Loss", loc="left")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_xticks(np.arange(T0, T1 + 0.1, 1))

    # advantage --------------------------------------------------------------
    ax = fig.add_subplot(gs[1, :])
    adv = np.asarray(log["advantage"], float)
    adv_l = np.asarray(log["adv_L"], float)
    adv_r = np.asarray(log["adv_R"], float)
    ax.plot(tw, adv_l[m], color=BLUE, lw=0.6, alpha=0.30)
    ax.plot(tw, adv_r[m], color=ORANGE, lw=0.6, alpha=0.30)
    ax.plot(tw, rolling(adv, 25)[m], color="k", lw=2.4, label="mean")
    ax.plot(tw, rolling(adv_l, 25)[m], color=BLUE, lw=2.0, label="left")
    ax.plot(tw, rolling(adv_r, 25)[m], color=ORANGE, lw=2.0, label="right")
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.set_xlim(T0, T1)
    ax.set_ylabel("advantage")
    ax.set_xlabel("time [s]")
    ax.set_title("Advantage", loc="left")
    ax.legend(loc="upper right", ncol=3, framealpha=0.9)
    ax.set_xticks(np.arange(T0, T1 + 0.1, 1))

    # old / new parameter policy --------------------------------------------
    names = list(agent["param_names"])
    rows = []
    for k in names:
        m0, m1 = agent["means"][k][0], agent["means"][k][-1]
        s0, s1 = abs(agent["sigmas"][k][0]), abs(agent["sigmas"][k][-1])
        eff = abs(m1 - m0) / max(s0, s1, 1e-12)
        rows.append((eff, k, m0, m1, s0, s1))
    rows.sort(reverse=True, key=lambda r: r[0])

    for j, (eff, k, m0, m1, s0, s1) in enumerate(rows[:12]):
        ax = fig.add_subplot(gs[2 + j // 4, j % 4])
        lo = min(m0 - 4.3 * s0, m1 - 4.3 * s1)
        hi = max(m0 + 4.3 * s0, m1 + 4.3 * s1)
        pad = 0.06 * (hi - lo + 1e-12)
        x = np.linspace(lo - pad, hi + pad, 600)
        for mu, sig, lab, color in ((m0, s0, "old", BLUE),
                                    (m1, s1, "new", ORANGE)):
            pdf = np.exp(-0.5 * ((x - mu) / sig) ** 2) / (
                sig * np.sqrt(2 * np.pi))
            ax.plot(x, pdf, color=color, lw=2.2, label=lab)
            ax.axvline(mu, color=color, ls="--", lw=1.0, alpha=0.85)
        ax.set_yticks([])
        ax.set_title(f"{short_name(k, 26)}\n"
                     f"μ {m0:.3g} → {m1:.3g}   ·   Δμ/σ {eff:.1f}",
                     fontsize=10.2, pad=4)
        ax.tick_params(axis="x", labelsize=8.5)
        if j == 0:
            ax.legend(loc="center", fontsize=8, framealpha=0.9)

    fig.suptitle("PERSONA", fontsize=18, y=0.965)
    fig.text(0.5, 0.012, "learning signals: 60–70 s window   ·   distributions: "
             "old (initial) vs new (final) policy", ha="center",
             fontsize=10, color=GREY)
    save(fig, "3_persona_advantage_loss_distributions.png")


# ----------------------------------------------------------------------------
# 4. Joint angles
# ----------------------------------------------------------------------------
def figure_angles(log):
    t = np.asarray(log["time_ms"], float) / 1000.0
    m = (t >= T0) & (t <= T1)
    tw = t[m]

    fig, axes = plt.subplots(3, 2, figsize=(24, 13), sharex=True,
                             gridspec_kw={"hspace": 0.36, "wspace": 0.10})
    fig.subplots_adjust(left=0.055, right=0.99, top=0.92, bottom=0.06)

    joints = [
        ("hip", "hip_ref", "hip_model", "hip [rad]"),
        ("knee", "knee_ref", "knee_model", "knee [rad]"),
        ("ankle", "ankle_target", "ankle_model", "ankle [rad]"),
    ]

    for col, (leg, color) in enumerate((("L", BLUE), ("R", RED))):
        for row, (joint, ref, mod, ylabel) in enumerate(joints):
            ax = axes[row, col]
            target = np.asarray(log[f"{ref}_{leg}"], float)
            ctrl = np.asarray(log[f"{mod}_{leg}"], float)
            ax.plot(tw, target[m], color="k", ls="--", lw=1.15,
                    label="target")
            ax.plot(tw, ctrl[m], color=color, lw=1.2, label="controlled")
            if joint == "ankle":
                presc = np.asarray(log[f"ankle_target_{leg}"], float)
                ax.plot(tw, presc[m], color=GREEN, ls=":", lw=1.1,
                        label="prescribed")
            rmse = float(np.sqrt(np.mean((ctrl[m] - target[m]) ** 2)))
            ax.set_title(f"{leg} {joint} · RMSE {rmse:.3f} rad",
                         loc="left", fontsize=11.5)
            ax.set_ylabel(ylabel)
            ax.set_xlim(T0, T1)
            ax.set_xticks(np.arange(T0, T1 + 0.1, 1))
            ax.set_ylim(min(target[m].min(), ctrl[m].min()) - 0.08,
                        max(target[m].max(), ctrl[m].max()) + 0.08)
            if row == 0:
                ax.legend(loc="upper left", ncol=2, framealpha=0.9,
                          fontsize=8.5)
            if row == 2:
                ax.legend(loc="upper right", ncol=1, framealpha=0.9,
                          fontsize=7.5)
                ax.set_xlabel("time [s]")

    fig.suptitle("Joint angles", fontsize=18, y=0.972)
    fig.text(0.5, 0.012, "target (dashed) vs controlled (solid)   ·   "
             "window 60–70 s of the 120 s run", ha="center", fontsize=10,
             color=GREY)
    save(fig, "4_angles_target_controlled.png")


# ----------------------------------------------------------------------------
# 5. Fisher strategy
# ----------------------------------------------------------------------------
def figure_fisher(agent):
    F = np.asarray(agent["fisher"], float)
    names = list(agent["all_names"])
    n = F.shape[0]
    if len(names) < n:
        names += [f"ch{i}" for i in range(len(names), n)]
    d = np.diag(F).astype(float)
    order = np.argsort(-d)
    C = F / np.sqrt(np.outer(d, d) + 1e-300)
    C = np.clip(C, -1.0, 1.0)

    fig = plt.figure(figsize=(26, 15))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 4.4],
                  width_ratios=[1.0, 2.15],
                  hspace=0.30, wspace=0.18,
                  left=0.055, right=0.975, top=0.925, bottom=0.075)
    axd = fig.add_subplot(gs[0, :])
    axf = fig.add_subplot(gs[1, 0])
    axz = fig.add_subplot(gs[1, 1])

    # diagonal activity ------------------------------------------------------
    x = np.arange(n)
    axd.semilogy(x[:269], d[:269], ".", ms=3.2, color=BLUE, label="μ")
    axd.semilogy(x[269:538], d[269:538], ".", ms=3.2, color=ORANGE, label="σ")
    axd.semilogy(x[538:], d[538:], ".", ms=4.0, color=GREEN, label="noise")
    for bound in (269, 538):
        axd.axvline(bound, color="k", lw=1.0, ls="--", alpha=0.7)
    axd.set_xlim(-5, n + 5)
    axd.set_ylim(1e-10, 1e10)
    axd.set_ylabel("F$_{ii}$")
    axd.set_xlabel("channel  [μ | σ | noise]")
    axd.set_title("Fisher diagonal", loc="left")
    handles = []
    for rank, i in enumerate(order[:6]):
        axd.plot(i, d[i], "o", ms=7, mfc="none", mec=RED, mew=1.3)
        axd.annotate(str(rank + 1), (i, d[i]), xytext=(7, 2),
                     textcoords="offset points", fontsize=8, color=RED)
        handles.append(Line2D([], [], color=RED, marker="o", lw=0,
                              mfc="none", label=f"{rank + 1}  {short_name(names[i], 26)}"))
    numbered = axd.legend(handles=handles, loc="lower right", ncol=2,
                          fontsize=7.2, framealpha=0.95,
                          handletextpad=0.3, columnspacing=1.0)
    axd.add_artist(numbered)
    axd.legend(loc="lower left", ncol=3, framealpha=0.9)

    v = 0.15
    im = axf.imshow(C, cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest",
                    aspect="equal")
    for bound in (268.5, 537.5):
        axf.axhline(bound, color="k", lw=0.9, ls="--", alpha=0.7)
        axf.axvline(bound, color="k", lw=0.9, ls="--", alpha=0.7)
    axf.set_xticks([134, 403, 546])
    axf.set_xticklabels(["μ", "σ", "noise"], fontsize=9)
    axf.set_yticks([134, 403, 546])
    axf.set_yticklabels(["μ", "σ", "noise"], fontsize=9)
    axf.set_title("Full 554 × 554", loc="left")

    top = order[:60]
    sub = C[np.ix_(top, top)]
    im2 = axz.imshow(sub, cmap="RdBu_r", vmin=-v, vmax=v,
                     interpolation="nearest", aspect="auto")
    labels = [short_name(names[i], 24) for i in top]
    axz.set_xticks(range(len(top)))
    axz.set_yticks(range(len(top)))
    axz.set_xticklabels(labels, rotation=90, fontsize=7.5)
    axz.set_yticklabels(labels, fontsize=7.5)
    axz.set_title("Top 60 channels", loc="left")
    cb2 = fig.colorbar(im2, ax=[axf, axz], fraction=0.022, pad=0.012)
    cb2.set_label("correlation", fontsize=9)

    fig.suptitle("Fisher strategy", fontsize=18, y=0.965)
    save(fig, "5_fisher_strategy_heatmap.png")


# ----------------------------------------------------------------------------
def main():
    log, agent, stdp = load_all("run.h5")
    figure_activity(log)
    figure_stdp(stdp)
    figure_persona(log, agent)
    figure_angles(log)
    figure_fisher(agent)
    print("done")


if __name__ == "__main__":
    main()
