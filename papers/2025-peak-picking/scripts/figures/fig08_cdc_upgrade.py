#!/usr/bin/env python3
"""fig08_cdc_upgrade.py

Figure 8: "CDC upgrade" diagnostic for one dataset (default: synthetic Case 2A).

This script compares:
  A) legacy "per-run optimum" histogram (from penalised run surfaces)
  B) the updated CDC ensemble catalogue overlay on the median penalised goodness curve

Path handling
-------------
You can run it inside the repository and point it at defaults:

  - Penalised ensemble catalogue:
      <paper>/data/derived/ensemble_catalogue.csv

  - Run-surfaces NPZ:
      *NOT* shipped in the public repo (often large). Provide via --runs-npz,
      or place a file named "<SAMPLE>_runs_S.npz" somewhere under <paper> and
      the script will try to auto-discover it.

Outputs
---------------------
Writes to:
  <paper>/outputs/figures/fig08_cdc_upgrade_<SAMPLE>.(png|pdf|svg)

Run
---
  python scripts/figures/fig08_cdc_upgrade.py --sample-id 2A --runs-npz /path/to/2A_runs_S.npz --save

Author: Lucy Mathieson
Date: 30/12/2025
"""


from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


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


def _default_paper_dir() -> Path:
    p = Path(__file__).resolve()
    cand = p.parents[2]  # expected: <paper>/scripts/figures/<this_file>
    if (cand / "data").is_dir() and (cand / "scripts").is_dir():
        return cand
    for parent in p.parents:
        if (parent / "data").is_dir() and (parent / "scripts").is_dir():
            return parent
    return cand


def _discover_runs_npz(paper_dir: Path, sample_id: str) -> Optional[Path]:
    """Best-effort search for '<sample>_runs_S.npz' under paper_dir."""
    patt = f"{sample_id}_runs_S.npz"
    hits = list(paper_dir.rglob(patt))
    if not hits:
        return None
    # Prefer the shallowest path (often <derived>/<diag>/<file>)
    hits.sort(key=lambda p: (len(p.parts), str(p)))
    return hits[0]


# =========================
# HELPERS (unchanged logic)
# =========================
def _col(df: pd.DataFrame, include: Sequence[str], exclude: Sequence[str] = ()):
    """Find first column whose lowercase name contains all strings in `include`
    and none in `exclude`."""
    for c in df.columns:
        cl = c.lower()
        if all(k in cl for k in include) and not any(e in cl for e in exclude):
            return c
    return None


def _to_ma(x):
    x = np.asarray(x, float)
    m = np.isfinite(x)
    if m.any() and np.nanmedian(x[m]) > 1e6:
        return x / 1e6
    return x


def _standardise_runs_matrix(S, n_ages):
    """Return shape (n_runs, n_ages)."""
    S = np.asarray(S, float)
    if S.ndim != 2:
        raise ValueError(f"S_runs_pen expected 2D, got shape {S.shape}")
    if S.shape[1] == n_ages:
        return S
    if S.shape[0] == n_ages:
        return S.T
    raise ValueError(f"Cannot align S_runs_pen with age grid. age={n_ages}, S={S.shape}")


def _fill_nan_linear(y):
    y = np.asarray(y, float)
    x = np.arange(y.size)
    m = np.isfinite(y)
    if m.sum() >= 2:
        return np.interp(x, x[m], y[m])
    if m.sum() == 1:
        return np.full_like(y, y[m][0])
    return np.zeros_like(y)


def _plateau_midpoint_argmax(row):
    """Argmax tie-break: midpoint of any flat max plateau."""
    row = np.asarray(row, float)
    if not np.isfinite(row).any():
        return None
    maxv = np.nanmax(row)
    idxs = np.where(row == maxv)[0]
    if idxs.size == 0:
        return int(np.nanargmax(row))
    return int((idxs.min() + idxs.max()) // 2)


def _parse_args() -> argparse.Namespace:
    paper_dir = _default_paper_dir()

    ap = argparse.ArgumentParser(description="Figure 8 CDC upgrade diagnostic.")

    ap.add_argument("--paper-dir", type=Path, default=paper_dir, help="Paper/repo root directory.")
    ap.add_argument("--sample-id", type=str, default="2A", help="Dataset tag (e.g. '2A').")
    ap.add_argument(
        "--runs-npz",
        type=Path,
        default=None,
        help="Path to '<SAMPLE>_runs_S.npz'. If omitted, the script will try to auto-discover.",
    )
    ap.add_argument(
        "--catalogue-csv",
        type=Path,
        default=None,
        help="Ensemble catalogue CSV (default: <paper>/data/derived/ensemble_catalogue.csv).",

    )
    ap.add_argument(
        "--fig-dir",
        type=Path,
        default=None,
        help="Output directory for figures (default: <paper>/Figures).",

    )
    ap.add_argument(
        "--formats",
        type=str,
        default="png,pdf,svg",
        help="Comma-separated output formats (used with --save).",

    )
    ap.add_argument("--no-save", action="store_true", help="Do not write figure files.")
    ap.add_argument("--no-show", action="store_true", help="Do not display the figure.")
    ap.add_argument(
        "--true-episodes",
        type=str,
        default="300,1800",
        help="Comma-separated true episode ages in Ma (use empty string for none).",

    )

    return ap.parse_args()


def main() -> None:
    args = _parse_args()

    paper_dir = args.paper_dir.expanduser().resolve()
    fig_dir = (args.fig_dir.expanduser().resolve() if args.fig_dir else (paper_dir / "outputs" / "figures"))
    catalogue_csv = (
        args.catalogue_csv.expanduser().resolve()
        if args.catalogue_csv
        else (paper_dir / "data" / "derived" / "ensemble_catalogue.csv")
    )

    sample_id = str(args.sample_id).strip()
    if not sample_id:
        raise ValueError("--sample-id cannot be empty")

    # Parse true episodes
    if str(args.true_episodes).strip() == "":
        true_episodes_ma = []
    else:
        true_episodes_ma = [float(x) for x in str(args.true_episodes).split(",") if x.strip()]

    runs_npz = args.runs_npz.expanduser().resolve() if args.runs_npz else _discover_runs_npz(paper_dir, sample_id)
    if runs_npz is None or not runs_npz.exists():
        raise FileNotFoundError(
            "Could not find a runs NPZ for this sample. "

            f"Either pass --runs-npz PATH, or place '{sample_id}_runs_S.npz' somewhere under {paper_dir}."
        )

    if not catalogue_csv.exists():
        raise FileNotFoundError(f"Catalogue CSV not found: {catalogue_csv}")

    # =========================
    # LOAD RUN SURFACES (penalised)
    # =========================
    z = np.load(runs_npz, allow_pickle=True)
    if "age_Ma" in z.files:
        age = np.asarray(z["age_Ma"], float)
    elif "age_ma" in z.files:
        age = np.asarray(z["age_ma"], float)
    else:
        raise KeyError(f"{runs_npz.name}: missing age_Ma/age_ma")

    age = _to_ma(age)

    if "S_runs_pen" not in z.files:
        raise KeyError(f"{runs_npz.name}: missing S_runs_pen")
    S_pen = z["S_runs_pen"]
    S_pen = _standardise_runs_matrix(S_pen, n_ages=age.size)
    S_pen = np.where(np.isfinite(S_pen), S_pen, np.nan)

    # Ensemble summary (median + 95% envelope) from penalised runs
    S_med = np.nanmedian(S_pen, axis=0)
    S_lo = np.nanquantile(S_pen, 0.025, axis=0)
    S_hi = np.nanquantile(S_pen, 0.975, axis=0)

    # Fill any all-NaN columns so curves don't break visually
    S_med = _fill_nan_linear(S_med)
    S_lo = _fill_nan_linear(S_lo)
    S_hi = _fill_nan_linear(S_hi)

    # Penalised per-run optima (recomputed from S_runs_pen for consistency)
    opt = []
    for r in range(S_pen.shape[0]):
        j = _plateau_midpoint_argmax(S_pen[r, :])
        if j is not None:
            opt.append(age[j])
    opt = np.asarray(opt, float)
    opt = opt[np.isfinite(opt)]

    opt_med = float(np.median(opt)) if opt.size else float("nan")
    opt_ci = np.quantile(opt, [0.025, 0.975]) if opt.size else np.array([np.nan, np.nan])

    # =========================
    # LOAD PENALISED CATALOGUE + FILTER TO SAMPLE
    # =========================
    cat = pd.read_csv(catalogue_csv)
    cat.columns = cat.columns.str.strip()

    # If file contains multiple samples, filter
    sample_col = _col(cat, ["sample"]) or _col(cat, ["case"]) or _col(cat, ["name"])
    if sample_col is not None:
        s = cat[sample_col].astype(str).str.strip()
        cat = cat[s.eq(sample_id)].copy()
        if cat.empty:
            cat = cat[s.str.contains(sample_id, na=False)].copy()

    # Detect key columns (robust)
    apex_col = _col(cat, ["apex"]) or _col(cat, ["peak", "age"]) or _col(cat, ["age"])
    lo_col = _col(cat, ["lo"]) or _col(cat, ["lower"]) or _col(cat, ["left"])
    hi_col = _col(cat, ["hi"]) or _col(cat, ["upper"]) or _col(cat, ["right"])
    sup_col = _col(cat, ["support"]) or _col(cat, ["vote"])

    if apex_col is None:
        raise KeyError(f"Cannot find apex column in catalogue. Columns={list(cat.columns)}")

    apex = _to_ma(pd.to_numeric(cat[apex_col], errors="coerce").to_numpy(float))
    lo = _to_ma(pd.to_numeric(cat[lo_col], errors="coerce").to_numpy(float)) if lo_col else np.full_like(apex, np.nan)
    hi = _to_ma(pd.to_numeric(cat[hi_col], errors="coerce").to_numpy(float)) if hi_col else np.full_like(apex, np.nan)
    sup = pd.to_numeric(cat[sup_col], errors="coerce").to_numpy(float) if sup_col else np.full_like(apex, np.nan)

    m = np.isfinite(apex)
    apex, lo, hi, sup = apex[m], lo[m], hi[m], sup[m]

    # =========================
    # PLOT
    # =========================
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.2, 3.6), constrained_layout=True)

    # ---- Panel A: Legacy run-optima summary ----
    axA.set_title("Legacy summary: per-run optimum (counts)")
    bins = np.linspace(np.nanmin(age), np.nanmax(age), 55)

    counts, edges, _ = axA.hist(
        opt,
        bins=bins,
        density=False,
        color="0.85",
        edgecolor="0.65",
        linewidth=0.8,
        alpha=1.0,
        label="run optima (counts)",
    )

    # KDE (scaled to counts/bin) with slightly tighter bandwidth so minor modes survive
    try:
        from scipy.stats import gaussian_kde

        bw_scale = 0.55  # < 1.0 = tighter than default Scott
        kde_fun = gaussian_kde(opt, bw_method=lambda s: s.scotts_factor() * bw_scale)
        grid = np.linspace(edges[0], edges[-1], 1000)
        binw = edges[1] - edges[0]
        kde_counts = kde_fun(grid) * len(opt) * binw
        axA.plot(grid, kde_counts, color="0.20", lw=2.0, label="KDE (scaled)")
    except Exception:
        grid = None
        kde_counts = None

    # Median + 95% interval
    if np.isfinite(opt_med):
        axA.axvline(opt_med, color="0.15", lw=2.0)
    if np.isfinite(opt_ci).all():
        axA.axvspan(opt_ci[0], opt_ci[1], color="0.15", alpha=0.12, label="95% interval")

    # True episodes
    true_ls = (0, (6, 3))
    for x in true_episodes_ma:
        axA.axvline(x, color="0.35", ls=true_ls, lw=1.2)
        axB.axvline(x, color="0.35", ls=true_ls, lw=1.2)

    axA.set_xlabel("Pb-loss age (Ma)")
    axA.set_ylabel("run count")
    axA.spines["top"].set_visible(False)
    axA.spines["right"].set_visible(False)
    axA.grid(True, axis="y", alpha=0.15)
    axA.text(0.01, 0.98, "A", transform=axA.transAxes, va="top", ha="left", fontweight="bold")

    axA.set_ylim(0, max(5, counts.max() * 1.15) if len(counts) else 5)

    # -------- zoom inset so the 1800 Ma mode is visible --------
    x0, x1 = 1600, 1950
    axins = axA.inset_axes([0.40, 0.20, 0.35, 0.35])  # [left, bottom, width, height] in axes fraction
    axins.hist(opt, bins=bins, density=False, color="0.75", edgecolor="0.55", alpha=0.8)

    # Replot KDE in inset (if it exists)
    if grid is not None and kde_counts is not None:
        axins.plot(grid, kde_counts, color="0.20", lw=1.6)

    for x in true_episodes_ma:
        axins.axvline(x, color="0.35", ls=(0, (6, 3)), lw=1.2)

    axins.set_xlim(x0, x1)

    n_old = int(np.sum((opt >= x0) & (opt <= x1))) if opt.size else 0
    axins.set_ylim(0, max(1, n_old) + 1)

    axins.set_title("zoom: 1600–1950 Ma", fontsize=8, pad=2)
    axins.grid(True, axis="y", alpha=0.15)
    axins.tick_params(which="both", direction="in", top=True, right=False)
    axins.spines["top"].set_visible(False)
    axins.spines["right"].set_visible(False)

    mark_inset(axA, axins, loc1=2, loc2=4, fc="none", ec="0.55", lw=0.8)

    axA.legend(frameon=False, fontsize=8, loc="upper left")

    # ---- Panel B: New CDC ensemble catalogue (penalised) ----
    axB.set_title("Updated CDC: penalised ensemble catalogue")
    axB.fill_between(age, S_lo, S_hi, alpha=0.18, label="95% run-surface envelope")
    axB.plot(age, S_med, lw=2.4, color="tab:red", label=r"median $\tilde S(t)$")

    # catalogue peaks: marker + interval + support label
    for a, l, h, s in zip(apex, lo, hi, sup):
        y = np.interp(a, age, S_med)
        axB.plot([a], [y], marker="^", ms=6, color="0.15", zorder=4)
        if np.isfinite(l) and np.isfinite(h):
            axB.hlines(y, l, h, color="0.15", lw=1.4, zorder=3)
        axB.text(
            a,
            y + 0.03,
            f"{a:.0f} Ma",
            ha="center",
            va="bottom",
            fontsize=8,
            color="0.15",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.0),
        )

    for x in true_episodes_ma:
        axB.axvline(x, color="0.35", ls=(0, (6, 3)), lw=1.4)

    axB.set_xlabel("Pb-loss age (Ma)")
    axB.set_ylabel(r"Goodness $\tilde S(t)$")
    axB.set_ylim(0.0, 1.02)
    axB.spines["top"].set_visible(False)
    axB.spines["right"].set_visible(False)
    axB.grid(True, axis="y", alpha=0.15)
    axB.legend(frameon=False, fontsize=8, loc="upper right")
    axB.text(0.01, 0.98, "B", transform=axB.transAxes, va="top", ha="left", fontweight="bold")

    xmin, xmax = float(np.nanmin(age)), float(np.nanmax(age))
    axA.set_xlim(xmin, xmax)
    axB.set_xlim(xmin, xmax)

    do_save = not args.no_save
    do_show = not args.no_show

    if do_save:
        fig_dir.mkdir(parents=True, exist_ok=True)
        fmts = [s.strip().lower() for s in str(args.formats).split(",") if s.strip()]
        out_paths = []
        for ext in fmts:
            out_path = fig_dir / f"fig08_cdc_upgrade_{sample_id}.{ext}"
            fig.savefig(out_path, bbox_inches="tight")
            out_paths.append(out_path)
        print("[Fig08] wrote:")
        for p in out_paths:
            print("  ", p)

    if do_show:
        plt.show()

    plt.close(fig)



if __name__ == "__main__":
    main()
