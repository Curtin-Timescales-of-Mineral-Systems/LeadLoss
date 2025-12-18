#!/usr/bin/env python3
"""
Concordant-fraction sweep summary figure:
Panel (a): reported Pb-loss age vs C for legacy single optimum and CDC ensemble
Panel (b): 95% interval width vs C for both methods
"""

import numpy as np
import matplotlib.pyplot as plt

# --- Data from the text -------------------------------------------------------

# Concordant fractions C (%)
C = np.array([5, 10, 20, 30, 40, 50, 60], dtype=float)

# True Pb-loss age for the sweep
TRUE_AGE = 25.0  # Ma

# Legacy single-optimum ages: always 26 Ma according to the text
legacy_age = np.full_like(C, 26.0)

# Legacy 95% intervals (lo, hi) from the text:
# C05: 21–46 Ma
# C10: 21–36 Ma
# C20: 21–41 Ma
# C30: 21–36 Ma
# C40–C60: 21–26 Ma
legacy_lo = np.array([21, 21, 21, 21, 21, 21, 21], dtype=float)
legacy_hi = np.array([46, 36, 41, 36, 26, 26, 26], dtype=float)
legacy_width = legacy_hi - legacy_lo

# CDC ensemble ages: ~30 Ma for all C (from text)
cdc_age = np.full_like(C, 30.0)

# CDC 95% interval width: ≈25 Ma for all C (text gives width only)
# For plotting, assume symmetric ±12.5 around 30 Ma
cdc_width = np.full_like(C, 25.0)
cdc_lo = cdc_age - 0.5 * cdc_width
cdc_hi = cdc_age + 0.5 * cdc_width

# --- Make the figure ----------------------------------------------------------

plt.rcParams.update({
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

fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2), sharex=True)

ax_age, ax_width = axes

# ---- Panel (a): age vs C -----------------------------------------------------

# True age reference line
ax_age.axhline(TRUE_AGE, color="0.8", ls="--", lw=1.0, label="True age (25 Ma)")

# Legacy single optimum with asymmetric CI
legacy_yerr = np.vstack((legacy_age - legacy_lo, legacy_hi - legacy_age))
ax_age.errorbar(
    C, legacy_age,
    yerr=legacy_yerr,
    fmt="o-", mfc="white", mec="firebrick",
    ecolor="firebrick", elinewidth=1.0, capsize=3,
    label="Legacy single optimum",
)

# CDC ensemble with symmetric CI (approximate)
cdc_yerr = 0.5 * cdc_width
ax_age.errorbar(
    C, cdc_age,
    yerr=cdc_yerr,
    fmt="s-", mfc="mediumseagreen", mec="mediumseagreen",
    ecolor="mediumseagreen", elinewidth=1.0, capsize=3,
    label="CDC ensemble peak",
)

ax_age.set_xlabel("Concordant fraction $C$ (\\%)")
ax_age.set_ylabel("Pb-loss age (Ma)")
ax_age.set_title("(a) Reported age vs $C$")
ax_age.set_xlim(0, 65)
ax_age.set_ylim(0, 60)
ax_age.legend(loc="upper right", frameon=False)

# ---- Panel (b): interval width vs C -----------------------------------------

ax_width.plot(C, legacy_width, "o-", mfc="white", mec="firebrick",
              label="Legacy 95\\% width")
ax_width.plot(C, cdc_width, "s-", mfc="mediumseagreen", mec="mediumseagreen",
              label="CDC 95\\% width")

ax_width.set_xlabel("Concordant fraction $C$ (\\%)")
ax_width.set_ylabel("95\\% interval width (Ma)")
ax_width.set_title("(b) Interval width vs $C$")
ax_width.set_xlim(0, 65)
ax_width.set_ylim(0, max(legacy_width.max(), cdc_width.max()) + 5)
ax_width.legend(loc="upper right", frameon=False)

fig.tight_layout()
fig.savefig("C_sweep_summary.pdf")
fig.savefig("C_sweep_summary.png", dpi=300)
plt.show()
