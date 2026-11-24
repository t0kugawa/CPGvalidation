"""Plot every recorded signal of the run at two timescales near the end.

For each signal family one figure is written per timescale::

    plots/figures/10s/<family>.png     last 10 s of the run
    plots/figures/2s/<family>.png      last 2 s of the run

and the requested overview (angles, angular velocity, muscle length, ground
force / weight transfer, EMG, motor neurons, RG, BS) is combined into::

    plots/figures/report/combined_report_10s.png
    plots/figures/report/combined_report_2s.png

Whole-run context (STDP, policy, Fisher, overview) goes to::

    plots/figures/context/*.png

Everything is drawn exactly as stored in ``run.h5``: raw units, no
normalisation, no rescaling.  Two optional aids are used and labelled in the
figure footer:

* a 100 ms moving mean drawn on top of the untouched raw trace for per-step
  population / spike-count channels;
* pale stance bands drawn behind state, force and weight-transfer panels.

Where two log channels are bit-identical this is stated in the figure note
instead of drawing the same line twice.

Usage:
    python plots/make_plots.py
    python plots/make_plots.py --h5 run.h5 --out plots/figures
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common as C
from common import BLUE, ORANGE, RED, GREEN, PURPLE, BROWN, GREY, BLACK

SMOOTH = 10          # 100 ms at the 10 ms log step


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def stance(sig, leg):
    return (np.asarray(sig[f"log/phase_{leg}"], str) == "stance").astype(float)


def stance_bands(sig):
    return [(stance(sig, "L"), BLUE, "stance L"),
            (stance(sig, "R"), RED, "stance R")]


def joint_series(sig, with_ref=False):
    """Flat list of joint-angle series (report panel)."""
    out = []
    for joint, color in C.JOINT_COLOR.items():
        for leg in ("L", "R"):
            out.append(C.S(f"log/{joint}_model_{leg}", f"{joint} {leg}",
                           color, C.SIDE_LS[leg], 1.15))
            if with_ref:
                out.append(C.S(f"log/{joint}_ref_{leg}", f"{joint} ref {leg}",
                               GREY, C.SIDE_LS[leg], 0.9, alpha=0.8))
    return out


def joint_rows(sig, ylabel, with_ref=True):
    rows = []
    for joint, color in C.JOINT_COLOR.items():
        series = [C.S(f"log/{joint}_model_{leg}", f"{joint} {leg}",
                      color, C.SIDE_LS[leg], 1.15) for leg in ("L", "R")]
        if with_ref:
            if joint == "ankle":
                # ankle has both the commanded target and the CPG reference
                series += [C.S(f"log/ankle_target_{leg}", f"target {leg}",
                               GREY, C.SIDE_LS[leg], 1.0)
                           for leg in ("L", "R")]
                series += [C.S(f"log/ankle_ref_{leg}", f"CPG ref {leg}",
                               GREY, ":", 0.9, alpha=0.9)
                           for leg in ("L", "R")]
            else:
                series += [C.S(f"log/{joint}_ref_{leg}", f"{joint} ref {leg}",
                               GREY, C.SIDE_LS[leg], 0.9, alpha=0.8)
                           for leg in ("L", "R")]
        rows.append(C.ROW(f"{joint} · model (coloured) vs reference (grey)",
                          ylabel, series))
    return rows


def muscle_rows(sig, field, ylabel, title, smooth=None):
    series = []
    for mus, ls in C.MUS_LS.items():
        for leg, color in (("L", BLUE), ("R", RED)):
            series.append(C.S(f"log/{field}_{mus}_{leg}",
                              f"{'ext' if mus == 'E' else 'flex'} {leg}",
                              color, ls, 1.1))
    return [C.ROW(title, ylabel, series, smooth=smooth)]


def population_series(sig, items, smooth=True):
    """Series list for raw ``activity`` channels (mean overlay handled by row)."""
    return [C.S(f"activity/{name}", label, color, ls, 1.0)
            for name, label, color, ls in items]


# ---------------------------------------------------------------------------
# families
# ---------------------------------------------------------------------------
def angle_rows(sig):
    return joint_rows(sig, "angle [rad]")


def angle_velocity_rows(sig):
    rows = []
    for joint, color in C.JOINT_COLOR.items():
        series = [C.S(f"log/{joint}_vel_m_{leg}", f"{joint} {leg}", color,
                      C.SIDE_LS[leg], 1.1) for leg in ("L", "R")]
        rows.append(C.ROW(joint, "angular velocity [rad/s]", series))
    return rows


def muscle_length_rows(sig):
    return muscle_rows(sig, "mus_len", "length [m]", "muscle length")


def muscle_velocity_rows(sig):
    return muscle_rows(sig, "mus_vel", "velocity [m/s]", "muscle velocity")


def emg_rows(sig):
    return muscle_rows(sig, "emg", "EMG [0–1]", "EMG envelope")


def drive_rows(sig):
    return muscle_rows(sig, "mus", "drive [0–1]", "muscle drive sent to MuJoCo")


def ground_force_rows(sig):
    total = np.asarray(sig["log/ground_force_L"], float) + \
        np.asarray(sig["log/ground_force_R"], float)
    rows = [C.ROW("ground reaction force", "force [N]", [
        C.S("log/ground_force_L", "left", BLUE, "-", 1.15),
        C.S("log/ground_force_R", "right", RED, "-", 1.15),
        dict(key="total", values=total, label="total", color=BLACK,
             ls=(0, (1.5, 1.5)), lw=1.0, alpha=0.9),
    ], bands=stance_bands(sig))]
    rows.append(C.ROW("load share (same as force / total force)", "fraction", [
        C.S("log/load_share_L", "left", BLUE, "-", 1.1),
        C.S("log/load_share_R", "right", RED, "-", 1.1),
    ], bands=stance_bands(sig)))
    rows.append(C.ROW("foot height", "z [m]", [
        C.S("log/foot_z_m_L", "left", BLUE, "-", 1.0),
        C.S("log/foot_z_m_R", "right", RED, "-", 1.0),
        C.S("log/foot_z_ref_L", "ref left", BLUE, ":", 0.9, alpha=0.7),
        C.S("log/foot_z_ref_R", "ref right", RED, ":", 0.9, alpha=0.7),
    ]))
    return rows


def foot_rows(sig):
    return [
        C.ROW("foot height", "z [m]", [
            C.S("log/foot_z_m_L", "left", BLUE, "-", 1.0),
            C.S("log/foot_z_m_R", "right", RED, "-", 1.0),
            C.S("log/foot_z_ref_L", "ref left", BLUE, ":", 0.9, alpha=0.7),
            C.S("log/foot_z_ref_R", "ref right", RED, ":", 0.9, alpha=0.7),
        ]),
        C.ROW("foot horizontal position", "x [m]", [
            C.S("log/foot_x_m_L", "left", BLUE, "-", 1.0),
            C.S("log/foot_x_m_R", "right", RED, "-", 1.0),
        ]),
    ]


def motor_neuron_rows(sig):
    rows = []
    for mus, ls in C.MUS_LS.items():
        rows.append(C.ROW(
            f"{'extensor' if mus == 'E' else 'flexor'} motor neurons",
            "mn_rate (as logged)", [
                C.S(f"log/mn_rate_{mus}_{leg}", f"{mus} {leg}", color, ls, 1.1)
                for leg, color in (("L", BLUE), ("R", RED))],
            smooth=SMOOTH))
    return rows


def rg_rows(sig):
    series = population_series(sig, [
        (f"rg_{mus.lower()}_{leg}", f"rg_{mus.lower()} {leg}", color, ls)
        for mus, ls in C.MUS_LS.items()
        for leg, color in (("L", BLUE), ("R", RED))])
    return [C.ROW("rhythm-generator population activity (raw)",
                  "activity (as logged)", series, smooth=SMOOTH)]


def bs_rows(sig):
    return [C.ROW("brainstem drive", "activity (as logged)", [
        C.S("activity/BS_in_e", "BS ext", BLUE, "-", 1.1),
        C.S("activity/BS_in_f", "BS flex", ORANGE, "-", 1.1),
        C.S("activity/BASE_in", "BASE tonic", GREY, "-", 1.0),
    ], smooth=SMOOTH)]


def load_share_rows(sig):
    return [C.ROW("weight transfer · load share (solid) vs contact command (dashed)",
                  "fraction", [
                      C.S("log/load_share_L", "share L", BLUE, "-", 1.1),
                      C.S("log/load_share_R", "share R", RED, "-", 1.1),
                      C.S("log/contact_ref_L", "contact ref L", BLUE, ":", 1.0),
                      C.S("log/contact_ref_R", "contact ref R", RED, ":", 1.0),
                  ], bands=stance_bands(sig))]


def gait_state_rows(sig):
    ph = {}
    for key in ("phase_L", "phase_R", "phase_ref_L", "phase_ref_R"):
        ph[key] = (np.asarray(sig[f"log/{key}"], str) == "stance").astype(float)
    st = (np.asarray(sig["log/stage"], str) == "track").astype(float)
    return [C.ROW("stance state (1 = stance) · raw step on stance bands", "state", [
        dict(key="phase_L", values=ph["phase_L"], label="phase L", color=BLUE,
             ls="-", lw=0.8, alpha=1.0),
        dict(key="phase_R", values=ph["phase_R"], label="phase R", color=RED,
             ls="-", lw=0.8, alpha=1.0),
        dict(key="phase_ref_L", values=ph["phase_ref_L"], label="ref L",
             color=BLUE, ls=":", lw=1.0, alpha=0.9),
        dict(key="phase_ref_R", values=ph["phase_ref_R"], label="ref R",
             color=RED, ls=":", lw=1.0, alpha=0.9),
        dict(key="stage", values=st, label="track stage", color=BLACK,
             ls=(0, (2, 2)), lw=1.0, alpha=0.7),
    ], ylim=(-0.1, 1.3), bands=stance_bands(sig))]


def cut_rows(sig):
    return [C.ROW("cutaneous afferent rate", "rate [Hz]", [
        C.S("log/cut_L", "CUT left", BLUE, "-", 1.1),
        C.S("log/cut_R", "CUT right", RED, "-", 1.1),
    ]), C.ROW("cutaneous input population activity (raw)",
              "activity (as logged)", [
                  C.S("activity/cut_in_L", "cut_in L", BLUE, "-", 1.0),
                  C.S("activity/cut_in_R", "cut_in R", RED, "-", 1.0),
              ], smooth=SMOOTH)]


def ia_rows(sig):
    rows = []
    for field, label, ylabel in (("ia_rate", "Ia rate", "rate [Hz]"),
                                 ("ia_len", "Ia length", "length [m]"),
                                 ("ia_vel", "Ia velocity", "velocity [m/s]")):
        series = []
        for mus, ls in C.MUS_LS.items():
            for leg, color in (("L", BLUE), ("R", RED)):
                series.append(C.S(f"log/{field}_{mus}_{leg}",
                                  f"{'ext' if mus == 'E' else 'flex'} {leg}",
                                  color, ls, 1.05))
        rows.append(C.ROW(label, ylabel, series))
    return rows


def ib_rows(sig):
    rows = []
    for field, label, ylabel in (("ib_rate", "Ib rate", "rate [Hz]"),
                                 ("ib_force", "Ib force", "force [N]"),
                                 ("ib_act", "Ib activation", "activation [0–1]")):
        series = []
        for mus, ls in C.MUS_LS.items():
            for leg, color in (("L", BLUE), ("R", RED)):
                series.append(C.S(f"log/{field}_{mus}_{leg}",
                                  f"{'ext' if mus == 'E' else 'flex'} {leg}",
                                  color, ls, 1.05))
        rows.append(C.ROW(label, ylabel, series))
    return rows


def afferent_activity_rows(sig):
    rows = []

    def row(label, rate_prefix, input_prefix):
        series = []
        for leg, color in (("L", BLUE), ("R", RED)):
            if rate_prefix:
                series.append(C.S(f"activity/{rate_prefix}_{leg}",
                                  f"{rate_prefix} {leg}", color, "-", 1.0))
            series.append(C.S(f"activity/{input_prefix}_{leg}",
                              f"{input_prefix} {leg}", color, ":", 1.0,
                              alpha=0.85))
        rows.append(C.ROW(label, "activity (as logged)", series, smooth=SMOOTH))

    row("Ia extensor (ia_e solid, ia_e_in dotted)", "ia_e", "ia_e_in")
    row("Ia flexor (ia_f solid, ia_f_in dotted)", "ia_f", "ia_f_in")
    row("Ib extensor incoming activity", None, "ib_e_in")
    row("Ib flexor incoming activity", None, "ib_f_in")
    return rows


def interneuron_rows(sig):
    rows = []

    def row(label, prefixes):
        series = []
        for prefix, color in prefixes:
            for leg, ls in (("L", "-"), ("R", C.SIDE_LS["R"])):
                series.append(C.S(f"activity/{prefix}_{leg}",
                                  f"{prefix} {leg}", color, ls, 1.0))
        rows.append(C.ROW(label, "activity (as logged)", series, smooth=SMOOTH))

    row("reciprocal Ia interneurons", [("in_e", BLUE), ("in_f", ORANGE)])
    row("Ib interneurons", [("in_ib_e", BLUE), ("in_ib_f", ORANGE)])
    row("in1", [("in1", PURPLE)])
    return rows


def vcell_rows(sig):
    rows = []
    for vc in ("v0d", "v0v", "v2a", "v3f"):
        rows.append(C.ROW(vc, "activity (as logged)", [
            C.S(f"activity/{vc}_{leg}", f"{vc} {leg}", color, C.SIDE_LS[leg], 1.0)
            for leg, color in (("L", BLUE), ("R", RED))], smooth=SMOOTH))
    return rows


def renshaw_rows(sig):
    rows = []
    for rc, color in (("r_e", BLUE), ("r_f", ORANGE)):
        rows.append(C.ROW(rc, "activity (as logged)", [
            C.S(f"activity/{rc}_{leg}", f"{rc} {leg}", color, C.SIDE_LS[leg], 1.0)
            for leg in ("L", "R")], smooth=SMOOTH))
    return rows


def muscle_activation_rows(sig):
    series = population_series(sig, [
        (f"mus_{mus.lower()}_{leg}", f"mus_{mus.lower()} {leg}", color, ls)
        for mus, ls in C.MUS_LS.items()
        for leg, color in (("L", BLUE), ("R", RED))])
    return [C.ROW("muscle activation (raw population signal)",
                  "activity (as logged)", series, smooth=SMOOTH)]


def learning_rows(sig):
    return [
        C.ROW("loss", "loss", [C.S("log/loss", "loss", BROWN, "-", 0.9)]),
        C.ROW("advantage", "advantage", [
            C.S("log/advantage", "mean", BLACK, "-", 1.2),
            C.S("log/adv_L", "left", BLUE, "-", 0.9, alpha=0.8),
            C.S("log/adv_R", "right", RED, "-", 0.9, alpha=0.8),
        ]),
    ]


def cpg_diag_rows(sig):
    rows = []
    for key, label in (("phase_clock", "phase clock"),
                       ("phase_period", "phase period"),
                       ("phase_confidence", "phase confidence"),
                       ("phase_error", "phase error")):
        rows.append(C.ROW(label, "value (as logged)", [
            C.S(f"log/{key}_L", f"{key} L", BLUE, "-", 1.1),
            C.S(f"log/{key}_R", f"{key} R", RED, "-", 1.1),
        ], ylim=(-0.05, 0.05)))
    return rows


def families(sig):
    return [
        dict(name="01_joint_angles", title="Joint angles",
             rows=angle_rows(sig),
             note="coloured = model (left solid, right dashed); grey = "
                  "reference / target"),
        dict(name="02_joint_angular_velocity", title="Joint angular velocity",
             rows=angle_velocity_rows(sig),
             note="model joint velocities; left solid, right dashed"),
        dict(name="03_muscle_length", title="Muscle length",
             rows=muscle_length_rows(sig),
             note="extensor solid, flexor dashed; left blue, right red  ·  "
                  "ia_len_* is bit-identical to mus_len_* and is plotted once"),
        dict(name="04_muscle_velocity", title="Muscle velocity",
             rows=muscle_velocity_rows(sig),
             note="extensor solid, flexor dashed; ia_vel_* is bit-identical "
                  "to mus_vel_* and is plotted once"),
        dict(name="05_ground_force", title="Ground force and weight transfer",
             rows=ground_force_rows(sig),
             note="contact_m_* is bit-identical to ground_force_*; "
                  "load_share_* equals force / (force L + force R)"),
        dict(name="06_foot_trajectory", title="Foot trajectory",
             rows=foot_rows(sig),
             note="foot position in the model frame, metres"),
        dict(name="07_emg", title="EMG",
             rows=emg_rows(sig),
             note="EMG envelope as logged (0–1 range), not rescaled"),
        dict(name="08_muscle_drive", title="Muscle drive",
             rows=drive_rows(sig),
             note="applied_mus_* is bit-identical to mus_* and is plotted once"),
        dict(name="09_motor_neuron", title="Motor neuron activity",
             rows=motor_neuron_rows(sig),
             note="mn_rate_* as logged  ·  activity/m_e_* is bit-identical "
                  "to mn_rate_E_* and is plotted once"),
        dict(name="10_rg_activity", title="Rhythm-generator activity (raw)",
             rows=rg_rows(sig),
             note="raw activity/rg_* channels, plotted without normalisation"),
        dict(name="11_rg_log_norm", title="Rhythm-generator activity (log channel)",
             rows=[C.ROW("rg_* as written to the log (0–1)", "activity (as logged)", [
                 C.S(f"log/rg_{mus.lower()}_{leg}", f"rg_{mus.lower()} {leg}",
                     color, ls, 1.05)
                 for mus, ls in C.MUS_LS.items()
                 for leg, color in (("L", BLUE), ("R", RED))])],
             note="these log channels are already unit-normalised by the "
                  "model itself; they are not rescaled here"),
        dict(name="12_bs_activity", title="Brainstem activity (raw)",
             rows=bs_rows(sig),
             note="BS_in_e / BS_in_f are the brainstem drives, BASE_in is the "
                  "tonic drive"),
        dict(name="13_cut", title="Cutaneous afferents",
             rows=cut_rows(sig),
             note="CUT rate in Hz (log) and the raw incoming population "
                  "activity (activity group)"),
        dict(name="14_ia", title="Ia muscle spindle afferents",
             rows=ia_rows(sig),
             note="extensor solid, flexor dashed; left blue, right red"),
        dict(name="15_ib", title="Ib Golgi tendon afferents",
             rows=ib_rows(sig),
             note="extensor solid, flexor dashed; left blue, right red"),
        dict(name="16_afferent_populations", title="Afferent population activity (raw)",
             rows=afferent_activity_rows(sig),
             note="rate solid, incoming activity dotted"),
        dict(name="17_interneurons", title="Spinal interneurons (raw)",
             rows=interneuron_rows(sig),
             note="activity channels plotted as stored"),
        dict(name="18_v_cells", title="V-cells (raw)",
             rows=vcell_rows(sig),
             note="activity channels plotted as stored"),
        dict(name="19_renshaw", title="Renshaw cells (raw)",
             rows=renshaw_rows(sig),
             note="activity channels plotted as stored"),
        dict(name="20_muscle_activation", title="Muscle activation (raw)",
             rows=muscle_activation_rows(sig),
             note="raw activity/mus_* channels; the stored values span "
                  "0 to 4.1e5 and are shown on that scale"),
        dict(name="21_gait_state", title="Gait state and CPG phase",
             rows=gait_state_rows(sig),
             note="1 = stance / track, 0 = swing / pose  ·  the logged phase "
                  "channels are exactly 0, which is why the derived state is "
                  "shown instead  ·  stance/swing alternate every ~0.2 s "
                  "(2.4 Hz), so the 10 s view is a dense comb by construction"),
        dict(name="22_cpg_diagnostics", title="CPG phase diagnostics",
             rows=cpg_diag_rows(sig),
             note="all four logged phase channels are exactly 0 for the "
                  "entire run; the flat traces are shown as stored"),
        dict(name="23_learning", title="Learning signals",
             rows=learning_rows(sig),
             note="loss and advantage as logged, no smoothing"),
    ]


def report_rows(sig):
    return [
        C.ROW("1 · joint angles", "angle [rad]", joint_series(sig)),
        C.ROW("2 · angular velocity", "rad/s",
              angle_velocity_rows(sig)[0]["series"] +
              angle_velocity_rows(sig)[1]["series"] +
              angle_velocity_rows(sig)[2]["series"]),
        C.ROW("3 · muscle length", "length [m]",
              muscle_length_rows(sig)[0]["series"]),
        C.ROW("4 · ground force", "force [N]",
              ground_force_rows(sig)[0]["series"], bands=stance_bands(sig)),
        C.ROW("5 · weight transfer", "fraction",
              load_share_rows(sig)[0]["series"], bands=stance_bands(sig)),
        C.ROW("6 · EMG", "EMG [0–1]", emg_rows(sig)[0]["series"]),
        C.ROW("7 · motor neuron activity", "mn_rate (as logged)",
              motor_neuron_rows(sig)[0]["series"] +
              motor_neuron_rows(sig)[1]["series"], smooth=SMOOTH),
        C.ROW("8 · RG activity (raw)", "activity (as logged)",
              rg_rows(sig)[0]["series"], smooth=SMOOTH),
        C.ROW("9 · BS activity (raw)", "activity (as logged)",
              bs_rows(sig)[0]["series"], smooth=SMOOTH),
        C.ROW("10 · gait state", "state",
              gait_state_rows(sig)[0]["series"], bands=stance_bands(sig)),
    ]


# ---------------------------------------------------------------------------
# STDP (own clock) and whole-run context
# ---------------------------------------------------------------------------
def make_stdp(sta, t_log, out, spans):
    ts = np.asarray(sta["sample_time_ms"], float) / 1000.0
    keys = list(sta["weight_mean"])
    colors = dict(zip(keys, [BLUE, RED, ORANGE, GREEN, PURPLE, BROWN]))
    label = {k: k.replace("_", " ") for k in keys}

    # weight means on the recorder clock (0.2 s) -----------------------------
    for span in spans:
        rows = [C.ROW("STDP weight mean per plastic projection", "weight", [
            C.S(k, label[k], colors[k], "-", 1.6) for k in keys])]
        for row in rows:
            for spec in row["series"]:
                spec["values"] = np.asarray(sta["weight_mean"][spec["key"]], float)
        C.render(rows, ts, span,
                 os.path.join(out, f"{span:g}s", "24_stdp_weights.png"),
                 f"STDP weights · last {span:g} s",
                 f"recorder samples every 0.2 s; {span:g} s window holds "
                 f"{int(round(span / 0.2)) + 1} points",
                 marker="o")

    # drift and update count on the log clock (100 Hz) -----------------------
    for span in spans:
        for field, fname, title, ylabel in (
                ("delta", "25_stdp_drift.png", "STDP weight drift", "mean Δw / step"),
                ("n", "26_stdp_updates.png", "STDP update count", "updates / step")):
            rows = [C.ROW("per plastic projection", ylabel, [
                C.S(k, label[k], colors[k], "-", 1.0) for k in keys])]
            for row in rows:
                for spec in row["series"]:
                    spec["values"] = np.asarray(sta[field][spec["key"]], float)
            C.render(rows, t_log, span,
                     os.path.join(out, f"{span:g}s", fname),
                     f"{title} · last {span:g} s",
                     "log-clock per-step values, raw")

    # whole-run STDP ---------------------------------------------------------
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(16, 12.5), sharex=True)
    ax = axes[0]
    for k in keys:
        mu = np.asarray(sta["weight_mean"][k], float)
        lo = np.asarray(sta["weight_min"][k], float)
        hi = np.asarray(sta["weight_max"][k], float)
        n = min(len(ts), len(mu), len(lo), len(hi))
        ax.fill_between(ts[:n], lo[:n], hi[:n], color=colors[k], alpha=0.13, lw=0)
        ax.plot(ts[:n], mu[:n], color=colors[k], lw=1.6, label=label[k])
    ax.set_ylabel("weight")
    ax.set_title("STDP weight mean (band = recorded min–max)", loc="left")
    ax.legend(loc="center right", ncol=2, framealpha=0.9)
    for ax, field, ylabel, title in (
            (axes[1], "delta", "mean Δw / step", "weight drift"),
            (axes[2], "n", "updates / step", "update count")):
        for k in keys:
            ax.plot(t_log, np.asarray(sta[field][k], float), color=colors[k],
                    lw=0.8, label=label[k])
        ax.set_ylabel(ylabel)
        ax.set_title(f"STDP {title}", loc="left")
        ax.legend(loc="upper right", ncol=3, framealpha=0.9, fontsize=7.5)
    axes[-1].set_xlabel("time [s]")
    fig.suptitle("STDP · whole run", fontsize=15)
    fig.text(0.5, 0.005, "weights recorded every 0.2 s; drift and counts on the "
             "100 Hz log clock", ha="center", fontsize=8.5, color=GREY)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    C.save(fig, os.path.join(out, "context", "stdp_whole_run.png"))


def _short(name: str, limit: int = 28) -> str:
    name = (name.replace("conn_", "").replace("_weight", "")
                .replace("producer:", "").replace(":poisson", "")
                .replace(":stdp", ""))
    return name if len(name) <= limit else name[: limit - 1] + "…"


def make_persona(agent, t_log, out):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    names = list(agent["param_names"])
    rows = []
    for k in names:
        m0, m1 = agent["means"][k][0], agent["means"][k][-1]
        s0, s1 = abs(agent["sigmas"][k][0]), abs(agent["sigmas"][k][-1])
        eff = abs(m1 - m0) / max(s0, s1, 1e-12)
        rows.append((eff, k, m0, m1, s0, s1))
    rows.sort(reverse=True, key=lambda r: r[0])

    fig = plt.figure(figsize=(22, 17.5))
    gs = GridSpec(5, 4, figure=fig, height_ratios=[1.0, 1.0, 1.2, 1.2, 1.2],
                  hspace=0.62, wspace=0.28,
                  left=0.055, right=0.985, top=0.935, bottom=0.06)

    ax = fig.add_subplot(gs[0, :])
    loss = np.asarray(agent["loss"], float)
    ax.plot(t_log, loss, color=BROWN, lw=0.5, alpha=0.35)
    ax.plot(t_log, C.uniform_filter1d(loss, 100, mode="nearest"), color=BROWN,
            lw=2.2, label="1 s mean")
    ax.set_ylabel("loss")
    ax.set_title("Policy loss (whole run)", loc="left")
    ax.legend(loc="upper right", framealpha=0.9)

    ax = fig.add_subplot(gs[1, :])
    adv = np.asarray(agent["advantage_vector"], float)
    ax.plot(t_log, adv[:, 0], color=BLUE, lw=0.5, alpha=0.25)
    ax.plot(t_log, adv[:, 1], color=ORANGE, lw=0.5, alpha=0.25)
    ax.plot(t_log, C.uniform_filter1d(adv[:, 0], 100, mode="nearest"),
            color=BLUE, lw=2.0, label="left 1 s mean")
    ax.plot(t_log, C.uniform_filter1d(adv[:, 1], 100, mode="nearest"),
            color=ORANGE, lw=2.0, label="right 1 s mean")
    ax.axhline(0, color=BLACK, lw=0.8, ls=":")
    ax.set_ylabel("advantage")
    ax.set_xlabel("time [s]")
    ax.set_title("Advantage (whole run)", loc="left")
    ax.legend(loc="upper right", ncol=2, framealpha=0.9)

    for j, (eff, k, m0, m1, s0, s1) in enumerate(rows[:12]):
        ax = fig.add_subplot(gs[2 + j // 4, j % 4])
        lo = min(m0 - 4.3 * s0, m1 - 4.3 * s1)
        hi = max(m0 + 4.3 * s0, m1 + 4.3 * s1)
        pad = 0.06 * (hi - lo + 1e-12)
        x = np.linspace(lo - pad, hi + pad, 600)
        for mu, sig, lab, color in ((m0, s0, "initial", BLUE),
                                    (m1, s1, "final", ORANGE)):
            pdf = np.exp(-0.5 * ((x - mu) / sig) ** 2) / (sig * np.sqrt(2 * np.pi))
            ax.plot(x, pdf, color=color, lw=2.0, label=lab)
            ax.axvline(mu, color=color, ls="--", lw=1.0, alpha=0.85)
        ax.set_yticks([])
        ax.set_title(f"{_short(k, 26)}\nμ {m0:.3g} → {m1:.3g}   ·   Δμ/σ {eff:.1f}",
                     fontsize=9.5, pad=4)
        ax.tick_params(axis="x", labelsize=8)
        if j == 0:
            ax.legend(loc="center", fontsize=8, framealpha=0.9)

    fig.suptitle("PERSONA · initial vs final policy", fontsize=15)
    fig.text(0.5, 0.006, "loss / advantage: whole run, raw with a 1 s mean  ·  "
             "distributions: first vs last stored policy sample",
             ha="center", fontsize=8.5, color=GREY)
    C.save(fig, os.path.join(out, "context", "persona_policy.png"))


def make_fisher(agent, out):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.lines import Line2D

    F = np.asarray(agent["fisher"], float)
    names = list(agent["all_names"])
    n = F.shape[0]
    if len(names) < n:
        names += [f"ch{i}" for i in range(len(names), n)]
    d = np.diag(F).astype(float)
    order = np.argsort(-d)
    Cm = F / np.sqrt(np.outer(d, d) + 1e-300)
    Cm = np.clip(Cm, -1.0, 1.0)

    n_mu = len(agent["param_names"]) or n // 2
    n_sigma = len(agent["sigmas"]) or n_mu
    if n_mu + n_sigma > n:
        n_sigma = max(0, n - n_mu)
    k_mu, k_sigma = n_mu, n_mu + n_sigma
    bounds = [n_mu - 0.5, k_sigma - 0.5]
    ticks, tick_labels = [], []
    if n_mu:
        ticks.append((n_mu - 1) / 2)
        tick_labels.append("μ")
    if n_sigma:
        ticks.append(n_mu + (n_sigma - 1) / 2)
        tick_labels.append("σ")
    if n - k_sigma:
        ticks.append(k_sigma + (n - k_sigma - 1) / 2)
        tick_labels.append("noise")

    fig = plt.figure(figsize=(24, 14.5))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 4.2],
                  width_ratios=[1.0, 2.1], hspace=0.32, wspace=0.16,
                  left=0.055, right=0.975, top=0.925, bottom=0.07)
    axd = fig.add_subplot(gs[0, :])
    axf = fig.add_subplot(gs[1, 0])
    axz = fig.add_subplot(gs[1, 1])

    x = np.arange(n)
    axd.semilogy(x[:k_mu], d[:k_mu], ".", ms=3.0, color=BLUE, label="μ")
    axd.semilogy(x[k_mu:k_sigma], d[k_mu:k_sigma], ".", ms=3.0,
                 color=ORANGE, label="σ")
    axd.semilogy(x[k_sigma:], d[k_sigma:], ".", ms=3.8, color=GREEN,
                 label="noise")
    for bound in bounds:
        axd.axvline(bound, color=BLACK, lw=0.9, ls="--", alpha=0.7)
    positive = d[d > 0]
    axd.set_ylim(10 ** np.floor(np.log10(positive.min())) / 10.0,
                 10 ** np.ceil(np.log10(positive.max())) * 10.0)
    axd.set_xlim(-5, n + 5)
    axd.set_ylabel("F$_{ii}$")
    axd.set_xlabel("channel  [μ | σ | noise]")
    axd.set_title("Fisher diagonal", loc="left")
    handles = []
    for rank, i in enumerate(order[:6]):
        axd.plot(i, d[i], "o", ms=6.5, mfc="none", mec=RED, mew=1.2)
        axd.annotate(str(rank + 1), (i, d[i]), xytext=(7, 2),
                     textcoords="offset points", fontsize=8, color=RED)
        handles.append(Line2D([], [], color=RED, marker="o", lw=0,
                              mfc="none", label=f"{rank + 1}  {_short(names[i], 26)}"))
    numbered = axd.legend(handles=handles, loc="upper right", ncol=2,
                          fontsize=7.0, framealpha=0.95, handletextpad=0.3,
                          columnspacing=1.0)
    axd.add_artist(numbered)
    axd.legend(loc="lower left", ncol=3, framealpha=0.9)

    v = 0.15
    axf.imshow(Cm, cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest",
               aspect="equal")
    for bound in bounds:
        axf.axhline(bound, color=BLACK, lw=0.8, ls="--", alpha=0.7)
        axf.axvline(bound, color=BLACK, lw=0.8, ls="--", alpha=0.7)
    axf.set_xticks(ticks)
    axf.set_xticklabels(tick_labels, fontsize=9)
    axf.set_yticks(ticks)
    axf.set_yticklabels(tick_labels, fontsize=9)
    axf.set_title(f"full {n} × {n}", loc="left")

    top = order[:60]
    sub = Cm[np.ix_(top, top)]
    im2 = axz.imshow(sub, cmap="RdBu_r", vmin=-v, vmax=v,
                     interpolation="nearest", aspect="auto")
    labels = [_short(names[i], 20) for i in top]
    axz.set_xticks(range(len(top)))
    axz.set_yticks(range(len(top)))
    axz.set_xticklabels(labels, rotation=90, fontsize=7.0)
    axz.set_yticklabels(labels, fontsize=7.0)
    axz.set_title("top 60 channels", loc="left")
    cb = fig.colorbar(im2, ax=[axf, axz], fraction=0.022, pad=0.012)
    cb.set_label("correlation", fontsize=9)

    fig.suptitle("Fisher strategy · whole run", fontsize=15)
    fig.text(0.5, -0.035, "normalised Fisher matrix; diagonal channels ordered "
             "by F$_{ii}$", ha="center", fontsize=8.5, color=GREY)
    C.save(fig, os.path.join(out, "context", "fisher_strategy.png"))


def make_overview(sig, t, out):
    rows = [
        C.ROW("loss", "loss", [C.S("log/loss", "loss", BROWN, "-", 0.7)],
              smooth=SMOOTH),
        C.ROW("joint angles", "angle [rad]", joint_series(sig)),
        C.ROW("ground force", "force [N]",
              [C.S("log/ground_force_L", "left", BLUE, "-", 0.8),
               C.S("log/ground_force_R", "right", RED, "-", 0.8)]),
        C.ROW("EMG", "EMG [0–1]", emg_rows(sig)[0]["series"]),
        C.ROW("motor neuron activity", "mn_rate (as logged)",
              motor_neuron_rows(sig)[0]["series"] +
              motor_neuron_rows(sig)[1]["series"], smooth=SMOOTH),
        C.ROW("RG activity (raw)", "activity (as logged)",
              rg_rows(sig)[0]["series"], smooth=SMOOTH),
        C.ROW("BS activity (raw)", "activity (as logged)",
              bs_rows(sig)[0]["series"], smooth=SMOOTH),
    ]
    for row in rows:
        for spec in row["series"]:
            if "values" not in spec:
                spec["values"] = np.asarray(sig[spec["key"]], float)
    C.render(rows, t, t[-1] - t[0],
             os.path.join(out, "context", "overview_whole_run.png"),
             "Whole-run overview", "raw signals over the full run",
             t_start=float(t[0]), figsize=(16, 17.5))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def resolve(rows, sig, where):
    for row in rows:
        for spec in row["series"]:
            if "values" not in spec:
                if spec["key"] not in sig:
                    raise KeyError(f"{where}: missing {spec['key']}")
                spec["values"] = np.asarray(sig[spec["key"]], float)


def generate_run_plots(h5_path: str, out_dir: str, spans: list[float] | None = None, skip_context: bool = False):
    """Generates the full server_24h plot suite (combined report, family breakdown, context)."""
    C.apply_style()
    sig = C.load_signals(h5_path)
    t = C.run_time(sig)
    t_end = float(t[-1])
    print(f"[plots] run ends at {t_end:.2f} s, {len(t)} samples, {len(sig)} channels -> {out_dir}")

    if spans is None:
        spans = [10.0, 2.0]
    valid_spans = []
    for s in spans:
        actual_s = min(float(s), t_end) if t_end > 0.5 else t_end
        if actual_s not in valid_spans:
            valid_spans.append(actual_s)

    fams = families(sig)
    for span in valid_spans:
        tag = f"{span:g}s"
        t0 = max(0.0, t_end - span)
        print(f"\n== {tag} window: {t0:.2f}–{t_end:.2f} s ==")
        for fam in fams:
            resolve(fam["rows"], sig, fam["name"])
            note = f"{fam['note']}   ·   window {t0:.2f}–{t_end:.2f} s of the " \
                   f"{t_end:.1f} s run"
            C.render(fam["rows"], t, span,
                     os.path.join(out_dir, tag, fam["name"] + ".png"),
                     fam["title"], note)

        rows = report_rows(sig)
        resolve(rows, sig, "report")
        C.render(rows, t, span,
                 os.path.join(out_dir, "report", f"combined_report_{tag}.png"),
                 f"Combined report · last {span:g} s",
                 f"all panels plotted as logged (no normalisation, no filtering)   ·   "
                 f"{t0:.2f}–{t_end:.2f} s of the {t_end:.1f} s run   ·   stance "
                 f"alternates every ~0.2 s, which is why the 10 s state panel is dense",
                 ncols=2, figsize=(21.5, 23.0))

    if not skip_context:
        print("\n== context ==")
        try:
            sta = C.load_stdp(h5_path)
            make_stdp(sta, t, out_dir, valid_spans)
        except Exception as exc:
            print(f"[plots] skip stdp context: {exc}")
        try:
            agent = C.load_agent(h5_path)
            make_persona(agent, t, out_dir)
            make_fisher(agent, out_dir)
        except Exception as exc:
            print(f"[plots] skip agent context: {exc}")
        try:
            make_overview(sig, t, out_dir)
        except Exception as exc:
            print(f"[plots] skip overview context: {exc}")

    print("[plots] done generating all figures")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", default="run.h5")
    parser.add_argument("--out", default=os.path.join("plots", "figures"))
    parser.add_argument("--spans", nargs="*", type=float, default=[10.0, 2.0])
    parser.add_argument("--skip-context", action="store_true")
    args = parser.parse_args()

    generate_run_plots(args.h5, args.out, spans=args.spans, skip_context=args.skip_context)


if __name__ == "__main__":
    main()
