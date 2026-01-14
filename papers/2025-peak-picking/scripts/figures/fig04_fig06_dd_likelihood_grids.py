#!/usr/bin/env python3
"""fig04_fig06_dd_likelihood_grids.py

Figures 4 & 6: Discordance-dating (DD; Reimink et al. 2016) likelihood grids.

Each panel shows:
  • all bootstrap likelihood curves (light grey)
  • the aggregate/median likelihood curve (black)
  • a shaded band for the DD summary statistic: median of per-bootstrap maxima (95% CI)
  • dashed verticals at the true lower-intercept ages (synthetic truth)

Defaults (repo-relative)
------------------------
Reads from:
  <paper>/data/derived/reimink_discordance_dating/

Run
---
  python scripts/figures/fig04_fig06_dd_likelihood_grids.py --save

"""


from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


# manuscript house-style
mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "mathtext.fontset": "stix",
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "legend.fontsize": 8,
        "lines.linewidth": 1.2,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

CI_COLOR = "skyblue"
TRUE_COLOR = "crimson"


# synthetic truth (Ma)
TRUE_CASES: Dict[str, List[float]] = {
    "1": [700],
    "2": [300, 1800],
    "3": [400],
    "4": [500, 1800],
    "5": [500, 1500],
    "6": [500, 1500],
    "7": [500, 1500],
}

TIERS = ["a", "b", "c"]


def _default_paper_dir() -> Path:
    p = Path(__file__).resolve()
    cand = p.parents[2]  # expected: <paper>/scripts/figures/<this_file>
    if (cand / "data").is_dir() and (cand / "scripts").is_dir():
        return cand
    for parent in p.parents:
        if (parent / "data").is_dir() and (parent / "scripts").is_dir():
            return parent
    return cand


def _load_dd_panel(dd_dir: Path, case: str, tier: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a single DD panel.

    Returns
    -------
    x_ma  : (n_grid,) grid in Ma
    boot  : (n_boot, n_grid) bootstrap curves, each normalised to max=1
    y_med : (n_grid,) aggregate curve normalised to max=1, interpolated onto x_ma if required
    """
    dd_dir = Path(dd_dir)

    boot_path = dd_dir / f"{case}{tier}_bootstrap_curves_boot200.csv"
    agg_path = dd_dir / f"{case}{tier}_lowerdisc_curve_boot200.csv"

    boot_df = pd.read_csv(boot_path)
    agg_df = pd.read_csv(agg_path)

    piv = (
        boot_df.pivot(index="Lower Intercept", columns="run.number", values="normalized.sum.likelihood")
        .sort_index()
    )

    x_ma = (piv.index.to_numpy(float) / 1e6).astype(float)
    boot = piv.to_numpy(float).T  # (n_boot, n_grid)

    # normalise each bootstrap curve to peak=1
    with np.errstate(invalid="ignore", divide="ignore"):
        peak = np.nanmax(boot, axis=1, keepdims=True)
        peak[~np.isfinite(peak)] = 1.0
        peak[peak <= 0] = 1.0
        boot = boot / peak

    # aggregate curve
    x_agg = (agg_df["Lower Intercept"].to_numpy(float) / 1e6).astype(float)
    y_agg = agg_df["normalized.sum.likelihood"].to_numpy(float)

    # normalise aggregate to peak=1
    ymax = float(np.nanmax(y_agg)) if np.isfinite(y_agg).any() else 1.0
    if ymax > 0:
        y_agg = y_agg / ymax

    if x_agg.shape == x_ma.shape and np.allclose(x_agg, x_ma):
        y_med = y_agg
    else:
        y_med = np.interp(x_ma, x_agg, y_agg, left=np.nan, right=np.nan)
        # fill any edge NaNs from bootstrap median
        if np.isnan(y_med).any():
            alt = np.nanmedian(boot, axis=0)
            m = np.isnan(y_med)
            y_med[m] = alt[m]

    return x_ma, boot, y_med


def _bootstrap_maxima_ci(x_ma: np.ndarray, boot: np.ndarray) -> Tuple[float, float, float, np.ndarray]:
    """Median + 95% CI of per-bootstrap maxima, returned in Ma."""
    if boot.size == 0:
        raise ValueError("Empty bootstrap array")

    boot_safe = np.where(np.isfinite(boot), boot, -np.inf)
    idx = np.argmax(boot_safe, axis=1)
    x_max = x_ma[idx]
    x_max = x_max[np.isfinite(x_max)]
    if x_max.size == 0:
        raise ValueError("No finite maxima")

    med = float(np.median(x_max))
    lo, hi = np.percentile(x_max, [2.5, 97.5]) if x_max.size >= 3 else (med, med)
    return med, float(lo), float(hi), x_max


def _draw_ci_band(ax, med: float, lo: float, hi: float, *, alpha: float = 0.18) -> None:
    ax.axvspan(lo, hi, ymin=0.0, ymax=1.0, facecolor=CI_COLOR, alpha=alpha, edgecolor="none", zorder=0)
    ax.text(
        med,
        1.02,
        f"{med:.0f} Ma",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=7,
        color="k",
        clip_on=False,
        zorder=5,
    )


def plot_dd_grid(
    dd_dir: Path,
    cases_to_plot: List[str],
    *,
    fig_dir: Path,
    out_stub: str,
    formats: List[str],
    save: bool,
    show: bool,
) -> None:
    nrows, ncols = len(cases_to_plot), len(TIERS)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(3.2 * ncols, 3.0 * nrows),
        sharex=True,
        sharey=True,
    )

    # ensure 2D axes
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, ncols)
    elif ncols == 1:
        axes = axes.reshape(nrows, 1)

    for r, case in enumerate(cases_to_plot):
        for c, tier in enumerate(TIERS):
            ax = axes[r, c]
            ax.set_xlim(0, 2000)
            ax.set_ylim(0, 1.10)
            ax.set_autoscale_on(False)

            try:
                x, boot, y_med = _load_dd_panel(dd_dir, case, tier)
            except FileNotFoundError as e:
                ax.text(0.5, 0.5, f"Missing data {case}{tier}\n{e}", ha="center", va="center", transform=ax.transAxes, fontsize=8)
                ax.axis("off")
                continue

            # background: all bootstrap curves
            for yb in boot:
                ax.plot(x, yb, color="0.75", lw=0.4, alpha=0.6, zorder=1)

            # median curve
            ax.plot(x, y_med, color="k", lw=1.5, zorder=2)

            # dashed verticals at true ages
            for age in TRUE_CASES.get(case, []):
                ax.axvline(age, ls="--", lw=0.9, color=TRUE_COLOR, zorder=0)

            # per-bootstrap maxima summary
            try:
                x_med, lo, hi, x_max = _bootstrap_maxima_ci(x, boot)
                # rug of per-bootstrap maxima along the top edge
                ax.scatter(
                    x_max,
                    np.full_like(x_max, 1.03),
                    marker="|",
                    s=18,
                    color="0.5",
                    alpha=0.5,
                    transform=ax.get_xaxis_transform(),
                    zorder=3,
                )
                _draw_ci_band(ax, x_med, lo, hi, alpha=0.18)
            except Exception:
                pass

            # cosmetics
            if r == 0:
                ax.set_title(f"Tier {tier.upper()}", fontsize=10, fontweight="bold")
            if c == ncols - 1:
                ax.text(
                    1.04,
                    0.5,
                    f"Case {case}",
                    transform=ax.transAxes,
                    rotation=-90,
                    va="center",
                    ha="left",
                    fontsize=9,
                    fontweight="bold",
                )

            show_xlabels = r == nrows - 1
            show_ylabels = c == 0
            ax.tick_params(direction="in", labelsize=7, pad=2, labelbottom=show_xlabels, labelleft=show_ylabels)

    # axis labels
    axes[-1, ncols // 2].set_xlabel("Lower-intercept age (Ma)", fontsize=9)
    axes[nrows // 2, 0].set_ylabel("Normalised likelihood", fontsize=9)

    fig.tight_layout()

    if save:
        fig_dir = Path(fig_dir)
        fig_dir.mkdir(parents=True, exist_ok=True)
        outs = []
        for ext in formats:
            out_path = fig_dir / f"{out_stub}.{ext}"
            fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
            outs.append(out_path)
        print("[DD grid] wrote:")
        for p in outs:
            print("  ", p)

    if show:
        plt.show()

    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    paper_dir = _default_paper_dir()
    dd_dir = paper_dir / "data" / "derived" / "reimink_discordance_dating"
    fig_dir = paper_dir / "outputs" / "figures"

    ap = argparse.ArgumentParser(description="DD likelihood grids (Figures 4 & 6).")

    ap.add_argument("--paper-dir", type=Path, default=paper_dir, help="Paper/repo root directory.")
    ap.add_argument("--dd-dir", type=Path, default=dd_dir, help="Directory containing DD bootstrap/aggregate CSVs.")
    ap.add_argument("--fig-dir", type=Path, default=fig_dir, help="Output directory for figure files.")
    ap.add_argument("--formats", type=str, default="png,pdf,svg", help="Comma-separated output formats.")
    ap.add_argument("--no-save", action="store_true", help="Do not write figure files.")
    ap.add_argument("--no-show", action="store_true", help="Do not display figures interactively.")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()

    paper_dir = args.paper_dir.expanduser().resolve()
    dd_dir = args.dd_dir.expanduser().resolve()
    fig_dir = args.fig_dir.expanduser().resolve()
    formats = [s.strip().lower() for s in str(args.formats).split(",") if s.strip()]
    save = not args.no_save
    show = not args.no_show

    print(f"[paths] paper_dir={paper_dir}")
    print(f"[paths] dd_dir={dd_dir} (exists={dd_dir.exists()})")
    print(f"[paths] fig_dir={fig_dir}")

    plot_dd_grid(
        dd_dir,
        cases_to_plot=["1", "2", "3", "4"],
        fig_dir=fig_dir,
        out_stub="fig04_dd_likelihood_grid_cases1to4",
        formats=formats,
        save=save,
        show=show,
    )

    plot_dd_grid(
        dd_dir,
        cases_to_plot=["5", "6", "7"],
        fig_dir=fig_dir,
        out_stub="fig06_dd_likelihood_grid_cases5to7",
        formats=formats,
        save=save,
        show=show,
    )


if __name__ == "__main__":
    main()
