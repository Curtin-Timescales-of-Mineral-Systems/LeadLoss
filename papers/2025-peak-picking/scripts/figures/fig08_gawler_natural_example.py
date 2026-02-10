#!/usr/bin/env python3
"""
fig08_gawler_natural_example.py

Figure 08: Illustrative natural example (Gawler Craton)
- Main panel: CDC penalised goodness ensemble curve + MC runs + IQR band + catalogue 95% CI bands.
- Inset: Wetherill concordia with ellipses coloured by discordance threshold and illustrative chords.

Defaults assume:
- GA filtered spot-level CSV in:
    papers/2025-peak-picking/data/inputs/ga_gawler_fig08/
  (auto-detected if --spots-csv not provided)

- Gawler diagnostics NPZs in:
    papers/2025-peak-picking/data/derived/ks_diagnostics_gawler/
  with filenames:
    sample_runs_S.npz
    sample_ensemble_surfaces.npz
  (tag default: "sample")

Outputs (by default) into:
    papers/2025-peak-picking/outputs/figures/
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Patch


# ── style (match your existing figure scripts) ───────────────────────────────
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

# colours (kept consistent with what you already used)
RUN_GREY = "0.75"
IQR_GREY = "0.85"
CI_BLUE  = "skyblue"
BLACK    = "k"
PEAK_COL = "tab:orange"

COL_CONC = "mediumseagreen"
COL_DISC = "thistle"
COL_CONC_CURVE = "slategray"

CHI2_95 = 5.991  # 95% ellipse scale for 2 dof

# decay constants (1/Ma)
L235 = 9.8485e-4
L238 = 1.55125e-4


# ── path helpers ─────────────────────────────────────────────────────────────
def _default_paper_dir() -> Path:
    p = Path(__file__).resolve()
    # <paper>/scripts/figures/<this file> -> paper dir is parents[2]
    cand = p.parents[2]
    if (cand / "data").is_dir() and (cand / "scripts").is_dir():
        return cand
    for parent in p.parents:
        if (parent / "data").is_dir() and (parent / "scripts").is_dir():
            return parent
    return cand


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def _parse_pair(s: str) -> Tuple[float, float]:
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError("Expected 'a,b'")
    return float(parts[0]), float(parts[1])


def _parse_list_of_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]


def save_figure(fig, outdir: Path, stub: str, formats: List[str]) -> None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        p = outdir / f"{stub}.{ext}"
        fig.savefig(p, bbox_inches="tight", pad_inches=0.02)
        print("[saved]", p)


# ── Wetherill / discordance helpers ──────────────────────────────────────────
def wetherill_xy(t_ma: float) -> Tuple[float, float]:
    if t_ma <= 0:
        return (0.0, 0.0)
    return (math.expm1(L235 * t_ma), math.expm1(L238 * t_ma))


def age_207_235(x: float) -> float:
    return math.log1p(x) / L235 if x > 0 else np.nan


def age_206_238(y: float) -> float:
    return math.log1p(y) / L238 if y > 0 else np.nan


def fractional_discordance(x: float, y: float) -> float:
    t76 = age_207_235(x)
    t68 = age_206_238(y)
    if not (np.isfinite(t76) and np.isfinite(t68) and t76 > 0 and t68 > 0):
        return np.nan
    return 1.0 - min(t76, t68) / max(t76, t68)


def ellipse_patch(x, y, sx, sy, rho, *, facecolor, alpha=0.80, lw=0.15) -> Ellipse:
    sx = float(sx)
    sy = float(sy)
    rho = float(rho) if np.isfinite(rho) else 0.0
    rho = float(np.clip(rho, -0.999, 0.999))

    cov = np.array([[sx * sx, rho * sx * sy], [rho * sx * sy, sy * sy]], float)
    lam, vec = np.linalg.eigh(cov)
    lam, vec = lam[::-1], vec[:, ::-1]

    w, h = 2.0 * np.sqrt(np.maximum(lam, 0.0) * CHI2_95)
    ang = np.degrees(np.arctan2(vec[1, 0], vec[0, 0]))

    return Ellipse(
        (x, y),
        w,
        h,
        angle=ang,
        facecolor=facecolor,
        alpha=alpha,
        edgecolor="black",
        lw=lw,
        zorder=3,
    )


def _find_col(df: pd.DataFrame, *tokens: str) -> Optional[str]:
    toks = [t.lower() for t in tokens]
    for c in df.columns:
        cl = str(c).lower()
        if all(t in cl for t in toks):
            return c
    return None


def load_ga_wetherill_csv(path: Path) -> pd.DataFrame:
    """
    Load GA-style CSV and return x,y,x_err,y_err,rho where:
      x = 207Pb/235U
      y = 206Pb/238U
      x_err, y_err are absolute 1σ (derived from 1σ % columns if present)
      rho is error correlation
    """
    df = pd.read_csv(path, sep=None, engine="python")

    x_col = _find_col(df, "207pb/235u")
    y_col = _find_col(df, "206pb/238u")
    x_pct = _find_col(df, "207pb/235u", "1sigma")
    y_pct = _find_col(df, "206pb/238u", "1sigma")
    rho_c = _find_col(df, "correl") or _find_col(df, "rho")

    if x_col is None or y_col is None:
        raise ValueError("Could not find required columns containing '207Pb/235U' and '206Pb/238U'.")

    x = pd.to_numeric(df[x_col], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")

    if x_pct is not None:
        xp = pd.to_numeric(df[x_pct], errors="coerce")
        x_err = x * (xp / 100.0)
    else:
        x_err = pd.Series(np.nan, index=df.index)

    if y_pct is not None:
        yp = pd.to_numeric(df[y_pct], errors="coerce")
        y_err = y * (yp / 100.0)
    else:
        y_err = pd.Series(np.nan, index=df.index)

    if rho_c is not None:
        rho = pd.to_numeric(df[rho_c], errors="coerce").fillna(0.0)
        # If rho looks like percent (e.g., 85 not 0.85), convert.
        if np.nanmax(np.abs(rho.to_numpy(float))) > 1.5:
            rho = rho / 100.0
        rho = rho.clip(-0.999, 0.999)
    else:
        rho = pd.Series(0.0, index=df.index)

    out = pd.DataFrame({"x": x, "y": y, "x_err": x_err, "y_err": y_err, "rho": rho})
    out = out[np.isfinite(out["x"]) & np.isfinite(out["y"]) & (out["x"] > 0) & (out["y"] > 0)]
    return out


def find_default_spots_csv(inputs_dir: Path) -> Path:
    """
    Find a sensible default GA spots CSV in the ga_gawler_fig08 input folder.
    """
    inputs_dir = Path(inputs_dir)
    if not inputs_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {inputs_dir}")

    # Prefer analyses extracts
    cands = sorted(inputs_dir.glob("*shrimp_analyses*.csv"))
    if not cands:
        cands = sorted(inputs_dir.glob("*.csv"))
    if not cands:
        raise FileNotFoundError(f"No CSV files found in: {inputs_dir}")

    # If multiple exist, pick the shortest name (tends to be the intended one) deterministically.
    cands.sort(key=lambda p: (len(p.name), p.name))
    return cands[0]


# ── panel plotting ───────────────────────────────────────────────────────────
def plot_main_ensemble_panel(
    ax: plt.Axes,
    *,
    ks_dir: Path,
    tag: str,
    xlim: Tuple[float, float],
    qband: Tuple[float, float] = (25.0, 75.0),
    plot_all_runs: bool = True,
) -> None:
    ks_dir = Path(ks_dir)
    runs_path = ks_dir / f"{tag}_runs_S.npz"
    ens_path  = ks_dir / f"{tag}_ensemble_surfaces.npz"

    D = _load_npz(runs_path)
    E = _load_npz(ens_path)

    x = np.asarray(D["age_Ma"], float).ravel()
    S_pen = np.asarray(D["S_runs_pen"], float)

    # Ensure shape is (R, N)
    if S_pen.ndim != 2:
        raise ValueError(f"Expected 2D array for S_runs_pen, got shape {S_pen.shape}")
    if S_pen.shape[1] != x.size and S_pen.shape[0] == x.size:
        S_pen = S_pen.T

    S = S_pen
    R, N = S.shape

    # Crop to xlim for plotting stability
    m = (x >= xlim[0]) & (x <= xlim[1])
    x_plot = x[m]
    S_plot = S[:, m]

    # MC runs (grey)
    if plot_all_runs:
        for j in range(R):
            ax.plot(x_plot, S_plot[j], color=RUN_GREY, lw=0.45, alpha=0.55, zorder=1)

    # IQR band
    qlo, qhi = qband
    ylo = np.nanpercentile(S_plot, qlo, axis=0)
    yhi = np.nanpercentile(S_plot, qhi, axis=0)
    ax.fill_between(x_plot, ylo, yhi, color=IQR_GREY, alpha=0.35, zorder=2)

    # Ensemble median curve
    y_med = np.nanmedian(S_plot, axis=0)
    ax.plot(x_plot, y_med, color=BLACK, lw=1.8, zorder=3)

    # Peaks + 95% CI bands from ensemble file (if present)
    peaks = []
    if all(k in E for k in ("peaks_age_Ma", "peaks_ci_low", "peaks_ci_high")):
        ages = np.asarray(E["peaks_age_Ma"], float).ravel()
        lo   = np.asarray(E["peaks_ci_low"], float).ravel()
        hi   = np.asarray(E["peaks_ci_high"], float).ravel()

        for a, l, h in zip(ages, lo, hi):
            if not (np.isfinite(a) and np.isfinite(l) and np.isfinite(h)):
                continue
            if a < xlim[0] or a > xlim[1]:
                continue
            peaks.append((float(a), float(l), float(h)))

    peaks.sort(key=lambda t: t[0])

    for a, l, h in peaks:
        ax.axvspan(l, h, ymin=0.0, ymax=1.0, facecolor=CI_BLUE, alpha=0.18, edgecolor="none", zorder=0)

        # marker at the median curve
        y_a = float(np.interp(a, x_plot, y_med))
        ax.plot([a], [y_a], marker="o", ms=4.6, color=PEAK_COL, mec="none", zorder=4)

        # label above axis (matches your existing style well)
        ax.text(
            a, 1.02, f"{a:.0f} Ma",
            transform=ax.get_xaxis_transform(),
            ha="center", va="bottom",
            fontsize=9, color="k",
            clip_on=False,
        )

    ax.set_xlim(*xlim)
    ax.set_xlabel("Pb-loss age (Ma)")
    ax.set_ylabel(r"Normalised goodness, $S$")
    ax.tick_params(direction="in")

    # Legend (match your Fig. 8 screenshot)
    handles = [
        Line2D([0], [0], color=RUN_GREY, lw=1.2, label=r"MC runs $S(t)$"),
        Patch(facecolor=IQR_GREY, edgecolor="none", alpha=0.35, label="MC quantile band"),
        Line2D([0], [0], color=BLACK, lw=1.8, label=r"ensemble curve $\tilde{S}(t)$"),
        Patch(facecolor=CI_BLUE, edgecolor="none", alpha=0.18, label="catalogue 95% CI"),
    ]
    ax.legend(handles=handles, loc="lower left", frameon=False)


def plot_wetherill_inset(
    ax: plt.Axes,
    *,
    spots_csv: Path,
    disc_cut: float,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    tmax: float,
    li_ages: List[float],
    draw_chords: bool = True,
    ui_age: Optional[float] = None,
) -> None:
    df = load_ga_wetherill_csv(Path(spots_csv))

    disc = np.array([fractional_discordance(x, y) for x, y in zip(df["x"].to_numpy(), df["y"].to_numpy())], float)
    is_conc = np.isfinite(disc) & (disc < disc_cut)

    # concordia curve
    t = np.linspace(0.0, float(tmax), 600)
    cx = np.expm1(L235 * t)
    cy = np.expm1(L238 * t)
    ax.plot(cx, cy, color=COL_CONC_CURVE, lw=1.0, zorder=0)

    # age markers (keep sparse like your screenshot)
    for tt in (2000.0, 3000.0):
        xt, yt = wetherill_xy(tt)
        if (xlim[0] <= xt <= xlim[1]) and (ylim[0] <= yt <= ylim[1]):
            ax.plot([xt], [yt], marker="o", ms=3.2, mfc="white", mec="0.45", mew=0.8, zorder=1)
            ax.annotate(
                f"{int(tt)} Ma",
                (xt, yt),
                textcoords="offset points",
                xytext=(4, 4),
                ha="left",
                va="bottom",
                fontsize=7,
                color="0.45",
                zorder=2,
            )

    # determine UI age (median concordant) if not provided
    if ui_age is None:
        t76_all = np.array([age_207_235(v) for v in df["x"].to_numpy()], float)
        t68_all = np.array([age_206_238(v) for v in df["y"].to_numpy()], float)
        t_conc = 0.5 * (t76_all + t68_all)
        t_conc = t_conc[is_conc & np.isfinite(t_conc)]
        ui_age = float(np.median(t_conc)) if t_conc.size else np.nan

    # illustrative chords
    if draw_chords and np.isfinite(ui_age):
        x_ui, y_ui = wetherill_xy(float(ui_age))
        for t_li in li_ages:
            x_li, y_li = wetherill_xy(float(t_li))
            ax.plot([x_ui, x_li], [y_ui, y_li], ls="--", lw=1.0, color="k", zorder=2)

    # ellipses (95% from 1σ + rho)
    has_err = np.isfinite(df["x_err"].to_numpy()).any() and np.isfinite(df["y_err"].to_numpy()).any()
    if has_err:
        for (x, y, sx, sy, rho, conc) in df.assign(conc=is_conc)[["x", "y", "x_err", "y_err", "rho", "conc"]].itertuples(index=False, name=None):
            if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(sx) and np.isfinite(sy)):
                continue
            fc = COL_CONC if conc else COL_DISC
            ax.add_patch(ellipse_patch(x, y, sx, sy, rho, facecolor=fc, alpha=0.80, lw=0.15))
    else:
        # fallback scatter
        ax.scatter(df.loc[~is_conc, "x"], df.loc[~is_conc, "y"], s=10, facecolors="none", edgecolors=COL_DISC, linewidths=0.6)
        ax.scatter(df.loc[is_conc, "x"], df.loc[is_conc, "y"], s=10, facecolors="none", edgecolors=COL_CONC, linewidths=0.6)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(r"$^{207}\mathrm{Pb}/^{235}\mathrm{U}$")
    ax.set_ylabel(r"$^{206}\mathrm{Pb}/^{238}\mathrm{U}$")
    ax.tick_params(direction="in")

    # legend (inset)
    handles = [
        Line2D([], [], ls="", marker="s", markersize=6, markerfacecolor=COL_CONC, markeredgecolor="black",
               label=f"Concordant (<{int(round(100*disc_cut))}%)"),
        Line2D([], [], ls="", marker="s", markersize=6, markerfacecolor=COL_DISC, markeredgecolor="black",
               label=f"Discordant (≥{int(round(100*disc_cut))}%)"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, borderaxespad=0.3, handlelength=1.0, handletextpad=0.4)


def main() -> None:
    paper_dir = _default_paper_dir()

    ap = argparse.ArgumentParser(description="Figure 08: Gawler natural example (main + inset).")

    ap.add_argument("--paper-dir", type=Path, default=paper_dir)

    ap.add_argument(
        "--inputs-dir",
        type=Path,
        default=(paper_dir / "data" / "inputs" / "ga_gawler_fig08"),
        help="Folder containing GA spot-level CSV extract(s) for Fig. 08.",
    )
    ap.add_argument(
        "--spots-csv",
        type=Path,
        default=None,
        help="Spot-level CSV (GA SHRIMP analyses extract). If omitted, auto-detected in --inputs-dir.",
    )

    ap.add_argument(
        "--ks-dir",
        type=Path,
        default=(paper_dir / "data" / "derived" / "ks_diagnostics_gawler"),
        help="Folder containing *_runs_S.npz and *_ensemble_surfaces.npz for the Gawler example.",
    )
    ap.add_argument("--tag", type=str, default="sample", help="NPZ file prefix tag (default: sample).")

    ap.add_argument("--disc-cut", type=float, default=0.10, help="Discordance threshold (fraction). Default 0.10 (=10%).")
    ap.add_argument("--li-ages", type=str, default="269,815", help="Comma-separated Pb-loss ages (Ma) for chords.")
    ap.add_argument("--no-chords", action="store_true", help="Disable dashed chords in inset.")

    ap.add_argument("--xlim-main", type=str, default="0,1500", help="Main panel x limits (Ma) as 'min,max'.")
    ap.add_argument("--qband", type=str, default="25,75", help="Quantile band (percentiles) as 'lo,hi' (default 25,75).")

    ap.add_argument("--xlim-inset", type=str, default="0,25", help="Inset x limits (207/235U) as 'min,max'.")
    ap.add_argument("--ylim-inset", type=str, default="0,0.8", help="Inset y limits (206/238U) as 'min,max'.")
    ap.add_argument("--tmax-inset", type=float, default=3500.0, help="Max concordia age to plot (Ma).")

    ap.add_argument("--fig-dir", type=Path, default=(paper_dir / "outputs" / "figures"))
    ap.add_argument("--outfile", type=str, default="fig08_gawler_natural_example")
    ap.add_argument("--formats", type=str, default="png,pdf,svg")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--no-show", action="store_true")

    args = ap.parse_args()

    xlim_main = _parse_pair(args.xlim_main)
    qlo, qhi = _parse_pair(args.qband)
    xlim_in  = _parse_pair(args.xlim_inset)
    ylim_in  = _parse_pair(args.ylim_inset)
    li_ages  = _parse_list_of_floats(args.li_ages)

    spots_csv = args.spots_csv
    if spots_csv is None:
        spots_csv = find_default_spots_csv(args.inputs_dir)

    # --- figure layout: main axis + inset axis ---
    fig, ax = plt.subplots(figsize=(7.4, 3.6))

    plot_main_ensemble_panel(
        ax,
        ks_dir=args.ks_dir,
        tag=str(args.tag).strip(),
        xlim=xlim_main,
        qband=(float(qlo), float(qhi)),
        plot_all_runs=True,
    )

    # inset placement tuned to match your screenshot (relative to main axis)
    ax_in = ax.inset_axes([0.63, 0.46, 0.35, 0.50])  # [x0, y0, w, h] in axes fraction

    plot_wetherill_inset(
        ax_in,
        spots_csv=Path(spots_csv),
        disc_cut=float(args.disc_cut),
        xlim=xlim_in,
        ylim=ylim_in,
        tmax=float(args.tmax_inset),
        li_ages=li_ages,
        draw_chords=(not args.no_chords),
        ui_age=None,
    )

    fig.tight_layout()

    formats = [s.strip().lower() for s in str(args.formats).split(",") if s.strip()]
    if not args.no_save:
        save_figure(fig, args.fig_dir.expanduser().resolve(), str(args.outfile).strip(), formats)

    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
