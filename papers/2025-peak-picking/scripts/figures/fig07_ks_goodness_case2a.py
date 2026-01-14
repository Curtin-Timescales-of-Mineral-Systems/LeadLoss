#!/usr/bin/env python3
"""
Figure 7 — K–S goodness example (case 2A).

Writes by default to:
  papers/2025-peak-picking/outputs/figures/ks_failure.(pdf|svg|png)

Run:
  python papers/2025-peak-picking/scripts/figures/fig07_ks_goodness_case2a.py --no-show
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# manuscript house-style
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


def _default_paper_dir() -> Path:
    # expected: <paper>/scripts/figures/<this_file>
    p = Path(__file__).resolve()
    cand = p.parents[2]
    if (cand / "data").is_dir() and (cand / "scripts").is_dir():
        return cand
    for parent in p.parents:
        if (parent / "data").is_dir() and (parent / "scripts").is_dir():
            return parent
    return cand


def _parse_args() -> argparse.Namespace:
    paper_dir = _default_paper_dir()
    ap = argparse.ArgumentParser(description="Figure 7: K–S goodness example (case 2A).")
    ap.add_argument("--paper-dir", type=Path, default=paper_dir, help="Path to papers/2025-peak-picking directory.")
    ap.add_argument("--ks-dir", type=Path, default=None,
                    help="Directory containing ks_exports CSVs (default: <paper>/data/derived/ks_exports).")
    ap.add_argument("--fig-dir", type=Path, default=None,
                    help="Output directory for figures (default: <paper>/outputs/figures).")
    ap.add_argument("--outfile-stub", type=str, default="ks_failure",
                    help="Base name (no extension) for output files.")
    ap.add_argument("--formats", type=str, default="png,pdf,svg",
                    help="Comma-separated output formats.")
    ap.add_argument("--no-save", action="store_true", help="Do not write figure files.")
    ap.add_argument("--no-show", action="store_true", help="Do not display the figure.")
    return ap.parse_args()


def _save_figure(fig, outdir: Path, stub: str, formats: list[str]) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        ext = ext.strip().lstrip(".")
        if not ext:
            continue
        path = outdir / f"{stub}.{ext}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
        print(f"[saved] {path}")


def main() -> None:
    args = _parse_args()

    paper_dir = args.paper_dir.expanduser().resolve()
    ks_dir = (args.ks_dir.expanduser().resolve() if args.ks_dir else (paper_dir / "data" / "derived" / "ks_exports"))
    fig_dir = (args.fig_dir.expanduser().resolve() if args.fig_dir else (paper_dir / "outputs" / "figures"))
    formats = [x.strip() for x in str(args.formats).split(",") if x.strip()]

    pen_path = ks_dir / "KS_goodness_pen.csv"
    raw_path = ks_dir / "KS_goodness_raw.csv"
    opt_pen_path = ks_dir / "KS_run_optima_pen.csv"
    opt_raw_path = ks_dir / "KS_run_optima_raw.csv"

    for p in [pen_path, raw_path, opt_pen_path, opt_raw_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required input file: {p}")

    # ---- load ----
    pen = pd.read_csv(pen_path)
    raw = pd.read_csv(raw_path)
    pen.columns = pen.columns.str.strip()
    raw.columns = raw.columns.str.strip()

    df = pen.copy()

    need_cols = [c for c in ["D_raw", "D_pen"] if (c not in df.columns and c in raw.columns)]
    if need_cols:
        df = df.merge(raw[["age_Ma"] + need_cols], on="age_Ma", how="inner")

    missing = [c for c in ["D_raw", "D_pen"] if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing required columns after merge: {missing}. "
            f"pen cols={list(pen.columns)}, raw cols={list(raw.columns)}"
        )

    df["S_raw"] = 1.0 - df["D_raw"].astype(float)
    df["S_pen"] = 1.0 - df["D_pen"].astype(float)

    t     = df["age_Ma"].to_numpy(float)
    S_raw = df["S_raw"].to_numpy(float)
    S_pen = df["S_pen"].to_numpy(float)

    opt_pen_runs = pd.read_csv(opt_pen_path)["opt_age_Ma"].to_numpy(float)
    opt_raw_runs = pd.read_csv(opt_raw_path)["opt_age_Ma"].to_numpy(float)

    opt_pen_runs = opt_pen_runs[np.isfinite(opt_pen_runs)]
    opt_raw_runs = opt_raw_runs[np.isfinite(opt_raw_runs)]

    ci_pen = np.quantile(opt_pen_runs, [0.025, 0.975])
    ci_raw = np.quantile(opt_raw_runs, [0.025, 0.975])

    opt_pen_ui = float(np.median(opt_pen_runs))
    opt_raw_ui = float(np.median(opt_raw_runs))

    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", alpha=0.15, linewidth=0.5)
    ax.tick_params(which="both", direction="in", top=True, right=True)

    ax.axvspan(ci_raw[0], ci_raw[1], alpha=0.4, color="0.75",
               label="raw 95% run-optima interval", zorder=0)
    ax.axvspan(ci_pen[0], ci_pen[1], alpha=0.18, color="tab:red",
               label="penalised 95% run-optima interval", zorder=0)

    ax.plot(t, S_raw, ls="-", color="0.25", lw=1.2, label="raw (no penalty)", zorder=2)
    ax.plot(t, S_pen, ls="-", color="tab:red", lw=1.6, label="penalised", zorder=3)

    for x, lab, ha in [(300, "true 300 Ma", "right"), (1800, "true 1800 Ma", "right")]:
        ax.axvline(x, color="darkblue", ls=":", lw=1.2, zorder=4)
        ax.text(x, 0.98, lab, rotation=90, transform=ax.get_xaxis_transform(),
                va="top", ha=ha, color="black", fontsize=8)

    raw_ls = (0, (3, 2))
    ax.axvline(opt_raw_ui, color="0.15", ls=raw_ls, lw=1.4, zorder=5)
    ax.text(opt_raw_ui, 0.98, f"raw median {opt_raw_ui:.0f} Ma",
            rotation=90, transform=ax.get_xaxis_transform(),
            va="top", ha="left", color="k", fontsize=8)

    ax.axvline(opt_pen_ui, color="tab:red", ls="--", lw=1.6, zorder=6)
    ax.text(opt_pen_ui, 0.98, f"pen median {opt_pen_ui:.0f} Ma",
            rotation=90, transform=ax.get_xaxis_transform(),
            va="top", ha="left", color="k", fontsize=8)

    ax.set_xlabel("Pb-loss age (Ma)")
    ax.set_ylabel(r"Goodness $S(t)=1-D^{*}(t)$")
    ax.set_ylim(0, 0.75)
    ax.set_xlim(0, 2000)

    x0, x1 = 300, 450
    mask = (t >= x0) & (t <= x1)
    if mask.any():
        y_plateau = float(np.nanmax(S_pen[mask]))
        dS = float(np.nanmax(S_pen[mask]) - np.nanmin(S_pen[mask]))
        ax.axvspan(x0, x1, alpha=0.04, zorder=0)
        ax.annotate(
            f"broad plateau (weak identifiability)\nΔS ≈ {dS:.3f} over {x0}–{x1} Ma",
            xy=((x0 + x1) / 2, y_plateau),
            xytext=(520, y_plateau + 0.10),
            arrowprops=dict(arrowstyle="->", lw=0.8),
            fontsize=9,
            ha="left",
            va="bottom",
        )

    x_peak = 1779
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

    if not args.no_save:
        _save_figure(fig, fig_dir, args.outfile_stub, formats)

    if not args.no_show:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    main()
