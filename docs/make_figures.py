import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "docs", "figures")
os.makedirs(OUT, exist_ok=True)

d = np.load(os.path.join(BASE, "run_log.npz"))
t = d["time_ms"] / 1000.0

# Fig 1: ankle tracking
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(t, d["ankle_target_L"], "k--", label="target L")
ax.plot(t, d["ankle_model_L"], "r-", label="model L")
ax.plot(t, d["ankle_target_R"], "b--", label="target R")
ax.plot(t, d["ankle_model_R"], "g-", label="model R")
ax.set_xlabel("time [s]"); ax.set_ylabel("ankle angle [rad]")
ax.set_title("Ankle tracking (model vs mocap-driven target)")
ax.legend(fontsize=8); ax.grid(alpha=0.3); fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_tracking.png"), dpi=150); plt.close(fig)

# Fig 2: muscle drives
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(t, d["mus_E_L"], "r-", label="SOL (extensor)")
ax.plot(t, d["mus_F_L"], "b-", label="TA (flexor)")
ax.set_ylim(-0.05, 1.05)
ax.set_xlabel("time [s]"); ax.set_ylabel("normalized drive")
ax.set_title("Controlled-leg muscle drives (saturated co-contraction)")
ax.legend(fontsize=8); ax.grid(alpha=0.3); fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_drives.png"), dpi=150); plt.close(fig)

# Fig 3: loss + advantage
fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
axes[0].plot(t, d["loss"], color="tab:brown"); axes[0].set_ylabel("loss"); axes[0].grid(alpha=0.3)
axes[1].plot(t, d["advantage"], color="tab:blue"); axes[1].axhline(0, color="k", ls=":")
axes[1].set_ylabel("advantage"); axes[1].set_xlabel("time [s]"); axes[1].grid(alpha=0.3)
fig.suptitle("PERSONA learning signals (no downward trend: no control authority)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_learning.png"), dpi=150); plt.close(fig)

# Fig 4: afferents
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(t, d["cut_L"], label="CUT L [Hz]")
ax2 = ax.twinx()
ax2.plot(t, d["ia_rate_E_L"], "g-", label="Ia E L [Hz]")
ax2.plot(t, d["ib_rate_E_L"], "m-", label="Ib E L [Hz]")
ax.set_xlabel("time [s]"); ax.set_ylabel("CUT [Hz]"); ax2.set_ylabel("Ia/Ib [Hz]")
ax.set_title("Sensory afferents delivered to the CPG")
ax.legend(loc="upper left", fontsize=8); ax2.legend(loc="upper right", fontsize=8)
ax.grid(alpha=0.3); fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_afferents.png"), dpi=150); plt.close(fig)

print("figures ->", OUT)
for f in sorted(os.listdir(OUT)):
    print(" ", f)

# quick numbers for the report text
trk = d["time_ms"] * 0 + 1
print("drive E sat frac", float(np.mean(d["mus_E_L"] >= 0.99)))
print("drive F sat frac", float(np.mean(d["mus_F_L"] >= 0.99)))
print("loss start/end", float(np.mean(d["loss"][:20])), float(np.mean(d["loss"][-20:])))
print("model ankle range", float(d["ankle_model_L"].min()), float(d["ankle_model_L"].max()))
print("target ankle range", float(d["ankle_target_L"].min()), float(d["ankle_target_L"].max()))
print("CUT max", float(d["cut_L"].max()), "Ib max", float(d["ib_rate_E_L"].max()))