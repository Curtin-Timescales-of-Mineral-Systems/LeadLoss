import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib as mpl

# manuscript house-style (copy/paste from your other script)
mpl.rcParams.update({
    "font.family":      "serif",
    "font.size":        10,
    "mathtext.fontset": "stix",
    "axes.labelsize":   10,
    "axes.titlesize":   11,
    "axes.linewidth":   0.8,
    "xtick.direction":  "in",
    "ytick.direction":  "in",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "legend.fontsize":  8,
    "lines.linewidth":  1.2,
    "savefig.dpi":      300,
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
})

# ---- paths ----
from pathlib import Path

KS_DIR = Path("papers/2025-peak-picking/data/derived/ks_exports")

pen_path = KS_DIR / "KS_goodness_pen.csv"
raw_path = KS_DIR / "KS_goodness_raw.csv"
opt_pen_path = KS_DIR / "KS_run_optima_pen.csv"
opt_raw_path = KS_DIR / "KS_run_optima_raw.csv"

# ---- load ----
pen = pd.read_csv(pen_path)
raw = pd.read_csv(raw_path)
pen.columns = pen.columns.str.strip()
raw.columns = raw.columns.str.strip()

# Use penalised file as base grid
df = pen.copy()

# Bring across any missing D columns from the raw file WITHOUT creating suffixes
need_cols = [c for c in ["D_raw", "D_pen"] if (c not in df.columns and c in raw.columns)]
if need_cols:
    df = df.merge(raw[["age_Ma"] + need_cols], on="age_Ma", how="inner")

missing = [c for c in ["D_raw", "D_pen"] if c not in df.columns]
if missing:
    raise KeyError(f"Missing required columns after merge: {missing}. "
                   f"pen cols={list(pen.columns)}, raw cols={list(raw.columns)}")

# Compute goodness
df["S_raw"] = 1.0 - df["D_raw"].astype(float)
df["S_pen"] = 1.0 - df["D_pen"].astype(float)

t     = df["age_Ma"].to_numpy(float)
S_raw = df["S_raw"].to_numpy(float)
S_pen = df["S_pen"].to_numpy(float)

# ---- CI bands + UI point estimate from per-run optima ----
opt_pen_runs = pd.read_csv(opt_pen_path)["opt_age_Ma"].to_numpy(float)
opt_raw_runs = pd.read_csv(opt_raw_path)["opt_age_Ma"].to_numpy(float)

opt_pen_runs = opt_pen_runs[np.isfinite(opt_pen_runs)]
opt_raw_runs = opt_raw_runs[np.isfinite(opt_raw_runs)]

ci_pen = np.quantile(opt_pen_runs, [0.025, 0.975])
ci_raw = np.quantile(opt_raw_runs, [0.025, 0.975])

opt_pen_ui = float(np.median(opt_pen_runs))
opt_raw_ui = float(np.median(opt_raw_runs))

# ---- plot styling ----
# ---- plot styling ----
fig, ax = plt.subplots(figsize=(7.5, 3.4))
ax.set_axisbelow(True)

# If your other figures do NOT use grids, set this to False.
# If you like a subtle y-grid, keep it very light:
ax.grid(True, axis="y", alpha=0.15, linewidth=0.5)

# Ensure tick style matches your other plots (inward ticks, including top/right)
ax.tick_params(which="both", direction="in", top=True, right=True)

# CI shading (different tones so they don't look like one band)
ax.axvspan(ci_raw[0], ci_raw[1], alpha=0.4, color="0.75",
           label="raw 95% run-optima interval", zorder=0)
ax.axvspan(ci_pen[0], ci_pen[1], alpha=0.18, color="tab:red",
           label="penalised 95% run-optima interval", zorder=0)

# Curves (leave linewidths explicit if you want emphasis beyond rcParams)
ax.plot(t, S_raw, ls="-", color="0.25", lw=1.2, label="raw (no penalty)", zorder=2)
ax.plot(t, S_pen, ls="-",  color="tab:red", lw=1.6, label="penalised", zorder=3)

# True episodes (thicker dashed + direct labels)
true_ls = (0, (6, 3))
for x, lab, ha in [(300, "true 300 Ma", "right"), (1800, "true 1800 Ma", "right")]:
    ax.axvline(x, color="darkblue", ls=":", lw=1.2, zorder=4)
    ax.text(x, 0.98, lab, rotation=90,
            transform=ax.get_xaxis_transform(),
            va="top", ha=ha, color="black", fontsize=8)

# Run-median optima (both)
raw_ls = (0, (3, 2))
ax.axvline(opt_raw_ui, color="0.15", ls=raw_ls, lw=1.4, zorder=5)
ax.text(opt_raw_ui, 0.98, f"raw median {opt_raw_ui:.0f} Ma",
        rotation=90, transform=ax.get_xaxis_transform(),
        va="top", ha="left", color="k", fontsize=8)

ax.axvline(opt_pen_ui, color="tab:red", ls="--", lw=1.6, zorder=6)
ax.text(opt_pen_ui, 0.98, f"pen median {opt_pen_ui:.0f} Ma",
        rotation=90, transform=ax.get_xaxis_transform(),
        va="top", ha="left", color="k", fontsize=8)

# Labels (mathtext will now match your other figures via stix)
ax.set_xlabel("Pb-loss age (Ma)")
ax.set_ylabel(r"Goodness $S(t)=1-D^{*}(t)$")
ax.set_ylim(0, 0.75)
ax.set_xlim(0, 2000)
# IMPORTANT for matching your other script:
# Your other plots keep the full frame, so do NOT hide spines here.
# (Remove these two lines if you previously had them.)
# ax.spines["top"].set_visible(False)
# ax.spines["right"].set_visible(False)

# --- plateau callout (choose bounds that match your plateau visually) ---
x0, x1 = 300, 450
mask = (t >= x0) & (t <= x1)
if mask.any():
    y_plateau = float(np.nanmax(S_pen[mask]))
    dS = float(np.nanmax(S_pen[mask]) - np.nanmin(S_pen[mask]))

    # faint bracket band *behind* (very light, so it doesn't fight CI shading)
    ax.axvspan(x0, x1, alpha=0.04, zorder=0)

    # arrow + text pointing to plateau
    ax.annotate(
        f"broad plateau (weak identifiability)\nΔS ≈ {dS:.3f} over {x0}–{x1} Ma",
        xy=((x0 + x1) / 2, y_plateau),
        xytext=(520, y_plateau + 0.10),
        arrowprops=dict(arrowstyle="->", lw=0.8),
        fontsize=9,
        ha="left",
        va="bottom",
    )

# --- optional: call out raw peak as “identifiable” ---
# pick peak location from the raw curve (or just use your median 1779)
x_peak = 1779  # or 1779
y_peak = float(S_raw[np.argmin(np.abs(t - x_peak))])
ax.annotate(
    "sharp maximum\n(identifiable)",
    xy=(x_peak, y_peak),
    xytext=(1550, y_peak + 0.10),
    arrowprops=dict(arrowstyle="->", lw=0.8),
    fontsize=9,
    ha="left",
    va="bottom",
)

ax.legend(frameon=False, loc="lower right")

fig.tight_layout()
plt.show()
plt.close(fig)
