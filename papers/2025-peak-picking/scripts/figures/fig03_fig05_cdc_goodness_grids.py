#!/usr/bin/env python3
"""fig03_fig05_cdc_goodness_grids.py

Figures 3 & 5: CDC goodness-surface grids with ensemble-catalogue overlay.

Each panel can show:
  • all runs' goodness curves S(t) (grey)   [requires per-run NPZ surfaces]
  • the median goodness curve (black)
  • true ages (dashed verticals)
  • ensemble catalogue ages (as 95% CI bands, labelled)

Defaults (repo-relative)
------------------------
- Ensemble catalogue CSV:
    <paper>/data/derived/ensemble_catalogue.csv

- NPZ surfaces directory (KS/CDC diagnostics):
    If you have per-run NPZ files, point --ks-dir at the folder containing them.

    This repository does not ship the (potentially large) NPZ surfaces, so by default
    the script will likely report "no NPZ" for each panel until you supply --ks-dir.

Outputs
---------------------
Writes to:
  <paper>/outputs/figures/fig03_cdc_goodness_grid_cases1to4.(png|pdf|svg)
  <paper>/outputs/figures/fig05_cdc_goodness_grid_cases5to7.(png|pdf|svg)

Run
---
  python scripts/figures/fig03_fig05_cdc_goodness_grids.py --ks-dir /path/to/ks_diagnostics --save

"""


from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Patch
from scipy.signal import find_peaks


# ── style ────────────────────────────────────────────────────────────────────
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

ORANGE = "#d95f02"  # median-curve peaks (illustrative)
GREEN = "#1b9e77"   # ensemble-catalogue overlay (triangles, if used)


# ── truth & layout ───────────────────────────────────────────────────────────
TRUE = {
    "1": [700],
    "2": [300, 1800],
    "3": [400],
    "4": [500, 1800],
    "5": [500, 1500],
    "6": [500, 1500],
    "7": [500, 1500],
}
TIERS = ["A", "B", "C"]       # columns
CASES_1_TO_4 = ["1", "2", "3", "4"]
CASES_5_TO_7 = ["5", "6", "7"]


def _default_paper_dir() -> Path:
    p = Path(__file__).resolve()
    cand = p.parents[2]  # expected: <paper>/scripts/figures/<this_file>
    if (cand / "data").is_dir() and (cand / "scripts").is_dir():
        return cand
    for parent in p.parents:
        if (parent / "data").is_dir() and (parent / "scripts").is_dir():
            return parent
    return cand


def _discover_tags(ks_dir: Path) -> List[str]:
    """Collect tags like '4C' from filenames '4C_123.npz' or '4C_runs_S.npz'."""
    ks_dir = Path(ks_dir)
    if not ks_dir.exists():
        return []
    tags = sorted({p.stem.split("_")[0].upper() for p in ks_dir.rglob("*.npz")})
    return tags


def _tri_down_on_line(
    ax,
    x,
    y,
    *,
    step,
    width_steps=1.6,
    height_data=0.035,
    facecolor=GREEN,
    edgecolor="k",
    linewidth=0.6,
    zorder=6,
):
    """Downward triangle whose apex is exactly at (x, y)."""
    w = width_steps * float(step)
    h = float(height_data)
    tri = Polygon(
        [(x, y), (x - 0.5 * w, y + h), (x + 0.5 * w, y + h)],
        closed=True,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
        clip_on=False,
    )
    ax.add_patch(tri)


def draw_catalogue_bands(ax, entries, *, label_peaks=True, alpha=0.2):
    """Draw vertical shaded bands for catalogue 95% CI intervals."""
    for e in entries:
        lo = float(e["lo"])
        hi = float(e["hi"])
        age = float(e["age"])

        ax.axvspan(lo, hi, ymin=0.0, ymax=1.0, facecolor="skyblue", alpha=alpha, edgecolor="none", zorder=0)

        if label_peaks and e.get("_label", True):
            ax.text(
                age,
                1.02,
                f"{age:.0f} Ma",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=7,
                color="k",
                clip_on=False,
                zorder=5,
            )


def _grid_step(x):
    dif = np.diff(np.asarray(x, float))
    return float(np.median(dif)) if dif.size else 10.0


def support_pct(raw, n_runs=None):
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return np.nan
    v = float(raw)
    if v <= 1.0 + 1e-12:  # fraction
        p = 100.0 * v
    elif v <= 100.0 + 1e-9:  # percent
        p = v
    elif n_runs:
        p = 100.0 * v / float(n_runs)
    else:
        p = v
    return float(np.clip(p, 0.0, 100.0))


def condense_catalogue(
    entries,
    *,
    step=10.0,
    merge_within=None,
    prefer="support",
    top_k_label=2,
    ci_overlap_frac=0.50,
):
    """Merge nearby/overlapping peaks and keep one representative per group."""
    if not entries:
        return []

    if merge_within is None:
        merge_within = 3.0 * float(step)

    ents = sorted(entries, key=lambda e: float(e["age"]))

    def sup_val(e):
        s = support_pct(e.get("support", np.nan))
        return -np.inf if np.isnan(s) else float(s)

    def ci_width(e):
        return float(e["hi"]) - float(e["lo"])

    def ci_overlap(e1, e2):
        lo = max(float(e1["lo"]), float(e2["lo"]))
        hi = min(float(e1["hi"]), float(e2["hi"]))
        if hi <= lo:
            return False
        span1 = max(ci_width(e1), 1e-9)
        span2 = max(ci_width(e2), 1e-9)
        frac = (hi - lo) / min(span1, span2)
        return frac >= ci_overlap_frac

    groups, cur = [], [ents[0]]
    for e in ents[1:]:
        if (abs(float(e["age"]) - float(cur[-1]["age"])) <= merge_within) or ci_overlap(e, cur[-1]):
            cur.append(e)
        else:
            groups.append(cur)
            cur = [e]
    groups.append(cur)

    reps = []
    for g in groups:
        if prefer == "support":
            rep = max(g, key=lambda e: (sup_val(e), -ci_width(e)))
        else:
            rep = min(g, key=lambda e: (ci_width(e), -sup_val(e)))
        reps.append(dict(rep))

    # which reps get text labels
    if top_k_label is None or top_k_label < 0 or top_k_label >= len(reps):
        idx_labeled = set(range(len(reps)))
    else:
        order = sorted(range(len(reps)), key=lambda i: (sup_val(reps[i]), -ci_width(reps[i])), reverse=True)
        idx_labeled = set(order[: int(top_k_label)])

    for i, r in enumerate(reps):
        r["_label"] = i in idx_labeled

    return reps


def _norm_tag(s: str) -> str:
    return str(s).strip().upper()


def load_catalogue_table(path: Path) -> Dict[str, List[dict]]:
    """Load ensemble catalogue CSV into a dict keyed by sample tag."""
    import io
    import re

    path = Path(path)
    if not path.exists():
        print(f"[catalogue] not found: {path}")
        return {}

    reqcols = {"sample", "peak_no", "age_ma", "ci_low", "ci_high", "support"}

    def _read_csv_like(obj):
        common = dict(
            comment="#",
            skip_blank_lines=True,
            encoding="utf-8-sig",
            engine="python",
            sep=",",
            skipinitialspace=True,
            usecols=lambda c: str(c).strip().lower() in reqcols,
        )
        try:
            return pd.read_csv(obj, on_bad_lines="warn", **common)
        except TypeError:
            return pd.read_csv(obj, error_bad_lines=False, warn_bad_lines=True, **common)

    try:
        df = _read_csv_like(path)
    except pd.errors.ParserError:
        text = path.read_text(encoding="utf-8-sig")
        text = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", text)  # remove thousands separators
        df = _read_csv_like(io.StringIO(text))

    df.columns = [str(c).strip().lower() for c in df.columns]
    for col in ["age_ma", "ci_low", "ci_high", "support"]:
        df[col] = pd.to_numeric(
            df[col]
            .astype(str)
            .str.replace("%", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip(),
            errors="coerce",
        )

    out: Dict[str, List[dict]] = {}
    keep = df.dropna(subset=["sample", "age_ma", "ci_low", "ci_high"])
    for _, r in keep.iterrows():
        samp = _norm_tag(r["sample"])
        out.setdefault(samp, []).append(
            dict(
                age=float(r["age_ma"]),
                lo=float(r["ci_low"]),
                hi=float(r["ci_high"]),
                support=(float(r["support"]) if pd.notna(r["support"]) else np.nan),
            )
        )
    for k in out:
        out[k].sort(key=lambda d: d["age"])

    print(f"[catalogue] samples: {', '.join(sorted(out.keys()))}")
    return out




def draw_catalogue_above(ax, entries, n_runs, step, y_base=1.035, dy=0.055):
    """Draw catalogue CI bars above the axes (no y-axis rescaling of curves)."""
    for i, e in enumerate(entries):
        y = y_base + i * dy
        # CI bar
        ax.plot([e["lo"], e["hi"]], [y, y], color="skyblue", lw=1.6, zorder=5, clip_on=False)
        # triangle whose tip sits on the CI bar
        _tri_down_on_line(ax, e["age"], y, step=step, facecolor=GREEN, edgecolor="k", linewidth=0.5, zorder=6)
        # optional support label
        if e.get("_label", True):
            pct = support_pct(e.get("support"), n_runs=n_runs)
            label = f'{e["age"]:.0f}' if not np.isfinite(pct) else f'{e["age"]:.0f} ({int(round(pct))}%)'
            ax.text(
                e["age"],
                y + 0.012,
                label,
                ha="center",
                va="bottom",
                fontsize=7,
                color="k",
                zorder=7,
                clip_on=False,
            )

def load_npz_both(sample_tag: str, ks_dir: Path):
    """Load MC goodness surfaces for a given sample tag.

    Returns:
      x_ma       : (n_grid,) float   age grid in Ma
      S_raw_runs : (n_runs, n_grid)  S = 1 - D_raw (or 1 - D)
      S_pen_runs : (n_runs, n_grid)  S = 1 - D_pen (or 1 - D*)
    """
    ks_dir = Path(ks_dir)

    # 1) Prefer pre-bundled S_runs_* files if present.
    runs_files = sorted(ks_dir.rglob(f"{sample_tag}_runs_S.npz"))
    if runs_files:
        f = runs_files[0]
        dat = np.load(f)

        # age grid
        if "age_Ma" in dat.files:
            x_ma = dat["age_Ma"].astype(float)
        elif "age_ma" in dat.files:
            x_ma = dat["age_ma"].astype(float)
        else:
            raise KeyError(f"{f.name}: no 'age_Ma'/'age_ma' key")

        def _orient(arr, name):
            if arr is None:
                return None
            arr = np.asarray(arr, float)
            if arr.ndim != 2:
                raise ValueError(f"{f.name}: {name} must be 2-D, got shape {arr.shape}")
            n0, n1 = arr.shape
            g = x_ma.size
            if n1 == g:
                return arr
            if n0 == g:
                return arr.T
            raise ValueError(f"{f.name}: {name} has incompatible shape {arr.shape} for grid {g}")

        S_raw = _orient(dat["S_runs_raw"], "S_runs_raw") if "S_runs_raw" in dat.files else None
        S_pen = _orient(dat["S_runs_pen"], "S_runs_pen") if "S_runs_pen" in dat.files else None

        if S_raw is None and S_pen is None:
            raise KeyError(f"{f.name}: missing S_runs_raw/S_runs_pen")

        if S_raw is None:
            S_raw = S_pen.copy()
        if S_pen is None:
            S_pen = S_raw.copy()

        return x_ma, S_raw, S_pen

    # 2) Fall back to per-run D_* surfaces.
    files = sorted(
        f
        for f in ks_dir.rglob(f"{sample_tag}_*.npz")
        if not any(s in f.name for s in ("_runs_S", "_ensemble_surfaces"))
    )
    if not files:
        raise FileNotFoundError(f"No MC surfaces for {sample_tag} in {ks_dir}")

    x_ma = None
    S_raw_list, S_pen_list = [], []

    for f in files:
        dat = np.load(f)

        if "age_Ma" in dat.files:
            x = dat["age_Ma"].astype(float)
        elif "age_ma" in dat.files:
            x = dat["age_ma"].astype(float)
        else:
            continue

        if x_ma is None:
            x_ma = x
        else:
            if x.shape != x_ma.shape or not np.allclose(x, x_ma):
                raise ValueError(f"Inconsistent grids in {sample_tag}")

        if "D_raw" in dat.files:
            D_raw = dat["D_raw"].astype(float)
        else:
            D_fallback = (
                dat["D_pen"]
                if "D_pen" in dat.files
                else dat["D_star"]
                if "D_star" in dat.files
                else dat["D"]
                if "D" in dat.files
                else None
            )
            if D_fallback is None:
                continue
            D_raw = D_fallback.astype(float)

        if "D_pen" in dat.files:
            D_pen = dat["D_pen"].astype(float)
        elif "D_star" in dat.files:
            D_pen = dat["D_star"].astype(float)
        elif "D" in dat.files:
            D_pen = dat["D"].astype(float)
        else:
            D_pen = D_raw.copy()

        S_raw_list.append(1.0 - D_raw)
        S_pen_list.append(1.0 - D_pen)

    if not S_raw_list:
        raise FileNotFoundError(f"No usable MC surfaces for {sample_tag}")

    return x_ma, np.vstack(S_raw_list), np.vstack(S_pen_list)


def _legend_handles(include_median_peaks=True):
    handles = [
        Line2D([0], [0], color="0.75", lw=1.2, label="all runs $S(t)$"),
        Line2D([0], [0], color="k", lw=1.5, label="median $S(t)$"),
        Line2D([0], [0], color="0.6", lw=0.6, ls="--", label="true age"),
        Patch(facecolor="skyblue", alpha=0.18, edgecolor="none", label="95% CI band"),
    ]
    if include_median_peaks:
        handles.insert(
            2,
            Line2D([0], [0], marker="o", markersize=5, markerfacecolor=ORANGE, markeredgecolor=ORANGE, lw=0, label="peaks (median)"),
        )
    return handles


def save_figure(fig, outdir: Path, stub: str, formats: List[str]):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in formats:
        p = outdir / f"{stub}.{ext}"
        fig.savefig(p, bbox_inches="tight", pad_inches=0.02)
        paths.append(p)
    print("[saved]")
    for p in paths:
        print("  ", p)


def make_grid(
    *,
    cases: List[str],
    ks_dir: Path,
    curve_surface: str,
    triangle_surface: str,
    title: str,
    outfile_stub: str,
    catalogue_map: Dict[str, List[dict]],
    outdir: Path,
    formats: List[str],
    overlay_mode: str = "curve",
    legend_compact: bool = True,
    show_median_peaks: bool = False,
    y_max: Optional[float] = None,
    save: bool = False,
    show: bool = True,
) -> None:
    assert overlay_mode in {"above", "curve"}
    curve_surface = str(curve_surface).strip().lower()
    triangle_surface = str(triangle_surface).strip().lower()

    n_row, n_col = len(cases), len(TIERS)
    fig, axes = plt.subplots(
        n_row,
        n_col,
        figsize=(7.5, 3.0 * n_row),
        sharex=True,
        sharey=True,
        gridspec_kw={"wspace": 0.16, "hspace": 0.08},
    )

    # Ensure 2D axes
    if n_row == 1 and n_col == 1:
        axes = np.array([[axes]])
    elif n_row == 1:
        axes = axes.reshape(1, n_col)
    elif n_col == 1:
        axes = axes.reshape(n_row, 1)

    any_overlay = False
    max_entries = 0

    for r, case in enumerate(cases):
        for c, tier in enumerate(TIERS):
            ax = axes[r, c]
            tag = f"{case}{tier}".upper()

            try:
                x, S_raw, S_pen = load_npz_both(tag, ks_dir)
            except FileNotFoundError:
                ax.set_xlim(0, 2000)
                ax.set_ylim(0, 1)
                ax.set_title(f"Tier {tier} (no NPZ for {tag})", color="crimson", fontsize=9)
                for age in TRUE.get(case, []):
                    ax.axvline(age, ls="--", lw=0.8, color="0.6", zorder=0)
                ax.axis("on")
                continue

            boot_curve = S_pen if curve_surface == "pen" else S_raw
            # triangle_surface retained for parity; currently bands do not depend on it
            y_med = np.nanmedian(boot_curve, axis=0)

            # optional: median-curve peak markers (illustrative)
            if show_median_peaks:
                pk, _ = find_peaks(y_med, prominence=0.02, width=3)
                if pk.size:
                    ax.plot(x[pk], y_med[pk], "o", ms=4, mfc=ORANGE, mec=ORANGE, lw=0, zorder=3)
                    for j in pk:
                        ax.text(x[j], y_med[j] + 0.02, f"{x[j]:.0f}", ha="center", va="bottom", fontsize=7, color=ORANGE)

            # curves
            for y in boot_curve:
                ax.plot(x, y, color="0.75", lw=0.4, alpha=0.6, zorder=1)
            ax.plot(x, y_med, color="k", lw=1.5, zorder=2)

            # truth
            for age in TRUE.get(case, []):
                ax.axvline(age, ls="--", lw=1, color="crimson", zorder=0)

            # overlay from catalogue (penalised by default)
            entries_raw = catalogue_map.get(_norm_tag(tag), [])
            n_runs = boot_curve.shape[0]
            step = _grid_step(x)
            entries = []
            if entries_raw:
                any_overlay = True
                entries = condense_catalogue(
                    entries_raw,
                    step=step,
                    merge_within=3.0 * step,
                    prefer="support",
                    top_k_label=2,
                    ci_overlap_frac=0.50,
                )

            if overlay_mode == "above":
                max_entries = max(max_entries, len(entries))
                draw_catalogue_above(ax, entries, n_runs, step)
            else:
                draw_catalogue_bands(ax, entries, label_peaks=True)

            ax.set_xlim(0, 2000)

            show_xlabels = r == n_row - 1
            show_ylabels = c == 0
            ax.tick_params(direction="in", labelsize=7, pad=2, labelbottom=show_xlabels, labelleft=show_ylabels)

            if r == 0:
                ax.set_title(f"Tier {tier}", fontweight="bold")
            if c == n_col - 1:
                ax.text(1.04, 0.5, f"Case {case}", transform=ax.transAxes, rotation=-90, va="center", ha="left", fontsize=9, fontweight="bold")

    axes[-1, n_col // 2].set_xlabel("Pb-loss age (Ma)", fontsize=9)
    axes[n_row // 2, 0].set_ylabel(r"Normalised goodness, $S$", fontsize=9)

    if any_overlay:
        if overlay_mode == "above":
            y_base, dy = 1.035, 0.055
            needed_top = max(1.12, y_base + (max_entries - 1) * dy + 0.06) if max_entries else 1.12
        else:
            needed_top = 1.12
        for ax in np.ravel(axes):
            if ax.has_data():
                ax.set_ylim(0, needed_top)
    else:
        for ax in np.ravel(axes):
            if ax.has_data():
                ax.set_ylim(0, 1.0)

    if y_max is not None:
        for ax in np.ravel(axes):
            if ax.has_data():
                ax.set_ylim(0, y_max)

    if legend_compact:
        fig.legend(
            handles=_legend_handles(include_median_peaks=show_median_peaks),
            loc="upper center",
            ncol=6,
            frameon=False,
            bbox_to_anchor=(0.5, 1.00),
            borderaxespad=0.6,
        )

    fig.tight_layout(rect=[0.07, 0.06, 1.0, 0.96])

    if save:
        save_figure(fig, outdir, outfile_stub, formats)

    if show:
        plt.show()
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    paper_dir = _default_paper_dir()
    default_catalogue = paper_dir / "data" / "derived" / "ensemble_catalogue.csv"

    default_ks_dir = paper_dir / "data" / "derived" / "ks_diagnostics"

    ap = argparse.ArgumentParser(description="CDC goodness-surface grids with catalogue overlay (Figures 3 & 5).")

    ap.add_argument("--paper-dir", type=Path, default=paper_dir, help="Paper/repo root directory.")
    ap.add_argument("--ks-dir", type=Path, default=default_ks_dir, help="Directory containing per-run NPZ surfaces.")
    ap.add_argument("--catalogue-csv", type=Path, default=default_catalogue, help="Ensemble catalogue CSV.")
    ap.add_argument("--fig-dir", type=Path, default=(paper_dir / "outputs" / "figures"), help="Output directory for figures.")
    ap.add_argument("--curve-surface", type=str, default="pen", choices=["pen", "raw"], help="Which surface to plot as curves.")
    ap.add_argument("--triangle-surface", type=str, default=None, choices=["pen", "raw"], help="Which catalogue to use (default: same as curve-surface).")

    ap.add_argument("--overlay-mode", type=str, default="curve", choices=["curve", "above"], help="Catalogue overlay mode.")
    ap.add_argument("--no-save", action="store_true", help="Do not write figure files.")
    ap.add_argument("--formats", type=str, default="png,pdf,svg", help="Comma-separated output formats.")
    ap.add_argument("--no-show", action="store_true", help="Do not display figures interactively.")

    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    save = not args.no_save
    show = not args.no_show

    paper_dir = args.paper_dir.expanduser().resolve()
    ks_dir = args.ks_dir.expanduser().resolve()
    catalogue_csv = args.catalogue_csv.expanduser().resolve()
    fig_dir = args.fig_dir.expanduser().resolve()
    formats = [s.strip().lower() for s in str(args.formats).split(",") if s.strip()]

    curve_surface = str(args.curve_surface).strip().lower()
    triangle_surface = str(args.triangle_surface).strip().lower() if args.triangle_surface else curve_surface

    print(f"[paths] paper_dir={paper_dir}")
    print(f"[paths] ks_dir={ks_dir} (exists={ks_dir.exists()})")
    print(f"[paths] catalogue_csv={catalogue_csv} (exists={catalogue_csv.exists()})")

    tags = _discover_tags(ks_dir)
    print(f"[paths] Found {len(tags)} NPZ tag(s): {', '.join(tags) if tags else '(none)'}")

    catalogue_map = load_catalogue_table(catalogue_csv)

    # Fig03 (Cases 1–4)
    make_grid(
        cases=CASES_1_TO_4,
        ks_dir=ks_dir,
        curve_surface=curve_surface,
        triangle_surface=triangle_surface,
        title="CDC goodness surfaces with ensemble catalogue — Cases 1–4 × Tiers A–C",
        outfile_stub="fig03_cdc_goodness_grid_cases1to4",
        catalogue_map=catalogue_map,
        outdir=fig_dir,
        formats=formats,
        overlay_mode=args.overlay_mode,
        legend_compact=True,
        show_median_peaks=False,
        y_max=1.0,
        save = not args.no_save,
        show = not args.no_show,
    )

    # Fig05 (Cases 5–7)
    make_grid(
        cases=CASES_5_TO_7,
        ks_dir=ks_dir,
        curve_surface=curve_surface,
        triangle_surface=triangle_surface,
        title="CDC goodness surfaces with ensemble catalogue — Cases 5–7 × Tiers A–C",
        outfile_stub="fig05_cdc_goodness_grid_cases5to7",
        catalogue_map=catalogue_map,
        outdir=fig_dir,
        formats=formats,
        overlay_mode=args.overlay_mode,
        legend_compact=True,
        show_median_peaks=False,
        y_max=0.8,
        save=save,
        show=show,
    )


if __name__ == "__main__":
    main()
