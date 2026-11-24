"""Shared style, loaders and renderer for the run plots.

Every figure is drawn with the same rcParams, the same colour semantics and
the same window/axis treatment.  The data is plotted exactly as it is stored
in ``run.h5`` -- no normalisation, no scaling, no filtering is applied to the
stored signal.  The only optional overlays are (a) a moving mean drawn on top
of the untouched raw trace and (b) pale stance bands drawn behind the data;
both are labelled in the figure footer.

Axis limits are fitted to the plotted window only, so a detail window is
never flattened by an excursion that happened somewhere else in the run.
"""

from __future__ import annotations

import os
import time

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MultipleLocator
from scipy.ndimage import uniform_filter1d

# ---------------------------------------------------------------------------
# palette -- one meaning per colour across every figure
# ---------------------------------------------------------------------------
BLUE = "#1f77b4"      # left
ORANGE = "#ff7f0e"    # flexor / knee
RED = "#d62728"       # right
GREEN = "#2ca02c"     # ankle
PURPLE = "#9467bd"
BROWN = "#8c564b"
GREY = "#555555"
BLACK = "#111111"

JOINT_COLOR = {"hip": BLUE, "knee": ORANGE, "ankle": GREEN}

#: left leg solid, right leg dashed.
SIDE_LS = {"L": "-", "R": (0, (4.5, 2.0))}
#: extensor solid, flexor dashed.
MUS_LS = {"E": "-", "F": (0, (4.5, 2.0))}

#: band colours (pale, drawn behind the data)
BAND_ALPHA = 0.13


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.solid_capstyle": "round",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
    })


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def decode(value):
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)


def load_signals(h5_path: str, groups=("log", "activity")) -> dict:
    """Load every per-step signal as ``<group>/<name>``."""
    signals: dict[str, np.ndarray] = {}
    with h5py.File(h5_path, "r") as f:
        for group in groups:
            g = f[group]
            for key in g:
                values = g[key][()]
                if values.dtype.kind in ("S", "O"):
                    values = np.array([decode(v) for v in values])
                signals[f"{group}/{key}"] = values
    return signals


def load_stdp(h5_path: str) -> dict:
    """Load the STDP recorder as ``{subgroup: {projection: array}}``."""
    out: dict[str, dict] = {}
    with h5py.File(h5_path, "r") as f:
        for group in f["stdp"]:
            node = f["stdp"][group]
            if isinstance(node, h5py.Group):
                out[group] = {k: node[k][()] for k in node}
            else:
                out[group] = node[()]
    return out


def load_agent(h5_path: str) -> dict:
    """Load the policy summary used for the context figures."""
    out = {}
    with h5py.File(h5_path, "r") as f:
        g = f["agent"]
        for key in ("fisher", "feature_scales", "trace_final", "credit_final",
                    "loss", "advantage_vector", "error_vector"):
            out[key] = g[key][()]
        out["means"] = {k: g["means"][k][()] for k in g["means"]}
        out["sigmas"] = {k: g["sigmas"][k][()] for k in g["sigmas"]}
        for key in ("param_names", "noise_names", "noise_scales", "all_names"):
            out[key] = [decode(v) for v in g.attrs.get(key, [])]
    return out


# ---------------------------------------------------------------------------
# windows
# ---------------------------------------------------------------------------
def run_time(signals: dict) -> np.ndarray:
    return np.asarray(signals["log/time_ms"], float) / 1000.0


def tick_style(ax, t0: float, t1: float) -> None:
    span = t1 - t0
    if span <= 3.0:
        major, minor, fmt = 0.25, 0.05, "{:.2f}"
    elif span <= 6.0:
        major, minor, fmt = 0.5, 0.1, "{:.1f}"
    elif span <= 30.0:
        major, minor, fmt = 1.0, 0.2, "{:.0f}"
    elif span <= 120.0:
        major, minor, fmt = 10.0, 2.0, "{:.0f}"
    else:
        major, minor, fmt = 100.0, 20.0, "{:.0f}"
    ax.set_xlim(t0, t1)
    ax.xaxis.set_major_locator(MultipleLocator(major))
    ax.xaxis.set_minor_locator(MultipleLocator(minor))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt.format(v)))
    ax.tick_params(axis="x", which="minor", length=2)
    ax.margins(x=0)


# ---------------------------------------------------------------------------
# series / row helpers
# ---------------------------------------------------------------------------
def S(key, label, color, ls="-", lw=1.1, alpha=1.0):
    """Series spec resolved against the signal dict inside :func:`render`."""
    return dict(key=key, label=label, color=color, ls=ls, lw=lw, alpha=alpha)


def ROW(title, ylabel, series, logy=False, ylim=None, smooth=None, bands=None):
    """One panel.

    ``smooth`` -- optional moving-mean length in samples (100 ms == 10) drawn
    over the untouched raw trace.
    ``bands`` -- optional list of ``(values, color, label)``; every contiguous
    stretch where ``values > 0.5`` is shaded behind the data.
    """
    return dict(title=title, ylabel=ylabel, series=series, logy=logy,
                ylim=ylim, smooth=smooth, bands=bands)


def _limits(row, t, t0, t1):
    if row.get("ylim") is not None:
        return row["ylim"]
    lo, hi = np.inf, -np.inf
    for spec in row["series"]:
        m = (t >= t0) & (t <= t1)
        yw = np.asarray(spec["values"], float)[m]
        yw = yw[np.isfinite(yw)]
        if yw.size:
            lo = min(lo, float(yw.min()))
            hi = max(hi, float(yw.max()))
    if not np.isfinite(lo):
        return (0.0, 1.0)
    if hi - lo < 1e-12:
        pad = max(abs(hi) * 0.05, 1e-6)
        return lo - pad, hi + pad
    pad = 0.06 * (hi - lo)
    return lo - pad, hi + pad


def _shade(ax, t, values, color, label=None):
    mask = np.asarray(values, float) > 0.5
    m = (t >= ax.get_xlim()[0]) & (t <= ax.get_xlim()[1])
    mask = mask & m
    if not mask.any():
        return None
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    for a, b in zip(edges[::2], edges[1::2]):
        ax.axvspan(t[a], t[min(b, len(t) - 1)], color=color,
                   alpha=BAND_ALPHA, lw=0, zorder=0)
    if label:
        return Patch(facecolor=color, alpha=0.35, label=label, lw=0)
    return None


def render(rows, t, span, out_path, suptitle, note="", ncols=1,
           t_end=None, t_start=None, figsize=None, marker=None):
    """Draw ``rows`` into a grid and save.  Returns the figure path."""
    t_end = float(t[-1]) if t_end is None else float(t_end)
    t0 = (t_end - span) if t_start is None else float(t_start)
    t1 = t_end
    t = np.asarray(t, float)
    nrows = int(np.ceil(len(rows) / ncols))
    if figsize is None:
        if ncols == 1:
            figsize = (16.0, 2.05 * nrows + 1.35)
        else:
            figsize = (21.5, 2.35 * nrows + 1.45)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True,
                             squeeze=False)
    axes = axes.ravel()

    smooth_used = any(row.get("smooth") for row in rows)
    bands_used = any(row.get("bands") for row in rows)

    for ax, row in zip(axes, rows):
        if not row.get("logy"):
            tick_style(ax, t0, t1)
        patches = []
        for values, color, label in row.get("bands") or []:
            patch = _shade(ax, t, values, color, label)
            if patch is not None:
                patches.append(patch)
        for spec in row["series"]:
            y = np.asarray(spec["values"], float)
            if row.get("smooth"):
                ax.plot(t, y, color=spec["color"], ls=spec["ls"],
                        lw=max(0.55, spec["lw"] * 0.6), alpha=0.45)
                ys = uniform_filter1d(y, size=int(row["smooth"]), mode="nearest")
                ax.plot(t, ys, color=spec["color"], ls=spec["ls"],
                        lw=spec["lw"] * 1.6, alpha=1.0, label=spec["label"])
            else:
                kw = dict(color=spec["color"], ls=spec["ls"], lw=spec["lw"],
                          alpha=spec["alpha"], label=spec["label"])
                if marker is not None:
                    kw.update(marker=marker, ms=3.0, mec="none")
                if row.get("logy"):
                    ax.semilogy(t, y, **kw)
                else:
                    ax.plot(t, y, **kw)
        ax.set_ylabel(row["ylabel"])
        ax.set_title(row["title"], loc="left", pad=5)
        ax.set_ylim(*(row["ylim"] if row["ylim"] is not None
                      else _limits(row, t, t0, t1)))
        lo, hi = ax.get_ylim()
        if row.get("zero", True) and lo < 0 < hi and not row.get("logy"):
            ax.axhline(0.0, color=BLACK, lw=0.7, ls=":")
        handles, labels = ax.get_legend_handles_labels()
        handles += patches
        labels += [p.get_label() for p in patches]
        if handles:
            ax.legend(handles, labels, loc="upper right",
                      ncol=min(len(handles), 3 if len(handles) <= 6 else 4),
                      framealpha=0.92, handlelength=2.0, columnspacing=1.0,
                      borderaxespad=0.3)

    for ax in axes[len(rows):]:
        ax.set_visible(False)
    # only the bottom panel of each column carries the time label
    for col in range(ncols):
        bottom = min((col + 1) * nrows, len(rows)) - 1
        if bottom >= 0:
            axes[bottom].set_xlabel("time [s]")

    fig.suptitle(suptitle, fontsize=15, y=0.995)
    foot = [note]
    if smooth_used:
        foot.append("thin = raw samples, thick = 100 ms moving mean")
    if bands_used:
        foot.append("shaded bands = stance intervals (pale blue left, pale red right)")
    foot = "   ·   ".join(f for f in foot if f)
    if foot:
        fig.text(0.5, -0.004, foot, ha="center", va="top", fontsize=8.5,
                 color=GREY)
    fig.tight_layout(rect=(0, 0.012 if foot else 0, 1, 0.975))
    save(fig, out_path)
    return out_path


def save(fig, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    for attempt in range(8):
        try:
            fig.savefig(path, dpi=200)
            break
        except OSError:
            if attempt == 7:
                raise
            time.sleep(1.0)
    plt.close(fig)
    print("wrote", path)
