#!/usr/bin/env python3
"""
Case 8 synthetic benchmark: fan-to-zero discordance geometry.

This script intentionally reuses the same generation logic as
`fig01_synthetic_cases1to4.py`, but defines a new case where multiple upper
concordant populations fan to a common lower intercept at 0 Ma.

Outputs:
  - Figure: case8_fan_to_zero_Wetherill_grid.{svg,tiff,png}
  - CSVs:
      case8_fan_to_zero_synth_TW.csv
      case8_fan_to_zero_synth_reim.csv
      case8_fan_to_zero_synth_weth.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import fig01_synthetic_cases1to4 as base


CASE8 = base.CaseDef(
    name="8",
    chords=[
        # Deliberately widen upper-age spacing so the three fan branches are
        # visually distinct in Tier A while still staying below the TW cap.
        base.Chord(t_up=3150, t_low=0, f_min=0.0, f_max=0.95),
        base.Chord(t_up=2500, t_low=0, f_min=0.0, f_max=0.95),
        base.Chord(t_up=1850, t_low=0, f_min=0.0, f_max=0.95),
    ],
)


def build_case8_panels() -> List[pd.DataFrame]:
    panels: List[pd.DataFrame] = []
    for tier_name in ["A", "B", "C"]:
        tier = base.TIERS[tier_name]
        dfp = base.simulate_case(CASE8, tier)

        # Fail fast if any branch silently drops out under filtering.
        if len(dfp) != base.N_POINTS:
            raise RuntimeError(
                f"Case 8 Tier {tier_name}: expected {base.N_POINTS} points, got {len(dfp)}."
            )
        disc_ups = (
            dfp.loc[~dfp["is_concordant"], "t_up_true"]
            .dropna()
            .astype(int)
            .unique()
        )
        if len(disc_ups) != len(CASE8.chords):
            raise RuntimeError(
                f"Case 8 Tier {tier_name}: expected {len(CASE8.chords)} discordia branches, got {len(disc_ups)}."
            )

        # Keep the same panel metadata logic used in cases 1-4.
        l_val = base.linearity_L(dfp)
        dfp["L"] = l_val
        dfp["True_I"] = ";".join(str(ch.t_low) for ch in CASE8.chords)

        n_total = len(dfp)
        n_conc = int(dfp["is_concordant"].sum())
        c_pct = int(round(100 * n_conc / n_total))
        l_pct = int(round(100 * l_val))
        dfp["CL_code"] = f"C{c_pct:02d}L{l_pct:02d}"
        panels.append(dfp)
    return panels


def plot_case8_grid(panels: List[pd.DataFrame]) -> plt.Figure:
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.5, 3.4),
        sharex=True,
        sharey=True,
        gridspec_kw={"wspace": 0.03, "hspace": 0.03},
    )

    tvals = np.linspace(0, 4500, 400)
    cx, cy = zip(*(base.wetherill_xy(tt) for tt in tvals))

    for df_panel in panels:
        tier = df_panel["Tier"].iat[0]
        c = ["A", "B", "C"].index(tier)
        ax = axes[c]

        # Concordia and guide chords
        ax.plot(cx, cy, color="slategray", lw=1.0, zorder=0)
        for ch in CASE8.chords:
            x_u, y_u = base.wetherill_xy(ch.t_up)
            x_l, y_l = base.wetherill_xy(ch.t_low)
            ax.plot([x_u, x_l], [y_u, y_l], ls="--", lw=0.8, color="lightslategray", zorder=1)

        for x, y, sx, sy, is_c in df_panel[["x", "y", "x_err", "y_err", "is_concordant"]].itertuples(index=False):
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            patch = base.ellipse_patch(x, y, sx, sy, is_conc=bool(is_c), rho=base.RHO_CONST)
            if not bool(is_c):
                # Keep the same style language (ellipses + center dots), but
                # slightly darken discordant fills for readability in Tier A.
                patch.set_facecolor((0.72, 0.58, 0.86, 0.90))
                patch.set_edgecolor("black")
                patch.set_linewidth(0.20)
                patch.set_zorder(4.5)
            ax.add_patch(patch)
            ax.plot(x, y, marker="o", ms=1.3, mfc="black", mec="none", zorder=6)

        ax.set_xlim(0, 25)
        ax.set_ylim(0, 0.8)
        ax.tick_params(direction="in", labelsize=7)
        ax.set_title(f"Case 8, Tier {tier}", fontweight="bold", fontsize=10)

        cl = df_panel["CL_code"].iat[0]
        ax.text(
            0.96,
            0.04,
            cl,
            transform=ax.transAxes,
            fontsize=6,
            ha="right",
            va="bottom",
            color="0.3",
            bbox=dict(facecolor="white", edgecolor="0.5", boxstyle="round,pad=0.15", alpha=0.7),
        )

    handles = [
        plt.Line2D([], [], ls="", marker="s", markersize=6, markerfacecolor=base.COL_CONC, markeredgecolor="black", label="Concordant"),
        plt.Line2D([], [], ls="", marker="s", markersize=6, markerfacecolor=(0.72, 0.58, 0.86, 0.90), markeredgecolor="black", label="Discordant"),
    ]
    axes[0].legend(handles=handles, loc="upper left", frameon=False, ncol=1, borderaxespad=0.3, handlelength=1.0, handletextpad=0.4)

    fig.text(0.5, 0.06, r"$^{207}\mathrm{Pb}/^{235}\mathrm{U}$", ha="center", va="center", fontsize=9)
    fig.text(0.04, 0.5, r"$^{206}\mathrm{Pb}/^{238}\mathrm{U}$", ha="center", va="center", rotation="vertical", fontsize=9)
    fig.tight_layout(rect=[0.06, 0.12, 1.0, 0.96])
    return fig


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate Case 8 fan-to-zero synthetic dataset.")
    ap.add_argument("--no-show", action="store_true", help="Do not open matplotlib windows.")
    args = ap.parse_args()

    # Re-seed so repeated invocations are deterministic.
    base.rng = np.random.default_rng(base.SEED)

    panels = build_case8_panels()
    master = pd.concat(panels, ignore_index=True)
    master["Sample"] = master["Case"].astype(str) + master["Tier"]

    fig = plot_case8_grid(panels)
    fig.savefig(base.FIG_DIR / "case8_fan_to_zero_Wetherill_grid.svg")
    fig.savefig(base.FIG_DIR / "case8_fan_to_zero_Wetherill_grid.tiff")
    fig.savefig(base.FIG_DIR / "case8_fan_to_zero_Wetherill_grid.png")
    if not args.no_show:
        plt.show()
    plt.close(fig)

    path_tw = base.DATA_DIR / "case8_fan_to_zero_synth_TW.csv"
    path_reim = base.DATA_DIR / "case8_fan_to_zero_synth_reim.csv"
    path_weth = base.DATA_DIR / "case8_fan_to_zero_synth_weth.csv"

    base.to_teraW(master).to_csv(path_tw, index=False)
    base.to_reimink(master).to_csv(path_reim, index=False)
    master.to_csv(path_weth, index=False)

    print(f"Wrote {len(master)} rows to:")
    print("  ", path_tw)
    print("  ", path_reim)
    print("  ", path_weth)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
