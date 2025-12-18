#!/usr/bin/env python3
"""
Discordance-dating (DD; Reimink et al. 2016) likelihood grids.
Rows = tiers (A–C), columns = cases.

Outputs publication-ready figures with:
  • all bootstrap curves (light grey)
  • median curve (black)
  • detected peaks on the median curve (red dots + age labels)
  • dashed verticals at true lower-intercept ages

Author: <you>
Date:   2025-08-07
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.signal import find_peaks
from scipy.signal import find_peaks, peak_widths, peak_prominences

# ───────────────────────────────── configuration ─────────────────────────────
BASE_DIR     = Path("/Users/lucymathieson/Desktop/reimink_discordance_dating") 

CASES = {
    "1": [700],
    "2": [300, 1800],
    "3": [400],
    "4": [500, 1800],
    "5": [500, 1500],
    "6": [500, 1500],
    "7": [500, 1500],
}
TIERS = ["a", "b", "c"]

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

# ---- SIMPLE-PEAK defaults (uniform across all panels) ----
PEAK_SMOOTH_MA    = 8.0     # light smoothing
PEAK_PROM_FRAC    = 0.015   # 1.5% of dynamic range
PEAK_PROM_MIN     = 0.004   # absolute floor
PEAK_MIN_DIST_MA  = 45.0    # allow closer peaks
PEAK_MIN_WIDTH_MA = 15.0    # accept narrower shoulders

# Colour for CI bands (match CDC style)
CI_COLOR = "skyblue"   # blue
TRUE_COLOR = "crimson"
def draw_dd_ci_band(ax, med, lo, hi, *, alpha=0.18):
    """
    Draw a vertical shaded band for a DD summary peak (median of bootstrap maxima).
    Band spans [lo, hi] in age and the full vertical extent of the axes.
    Adds a 'XXX Ma' label above the band.
    """
    # shaded 95% CI band
    ax.axvspan(lo, hi, ymin=0.0, ymax=1.0,
               facecolor=CI_COLOR, alpha=alpha,
               edgecolor="none", zorder=0)

    # age label just above the axes
    ax.text(
        med, 1.02, f"{med:.0f} Ma",
        transform=ax.get_xaxis_transform(),  # x in data, y in axes fraction
        ha="center", va="bottom",
        fontsize=7, color="k",
        clip_on=False, zorder=5,
    )

def _load_dd_panel(case: str, tier: str):
    """
    Returns:
      x      : 1D array (Ma), bootstrap x-grid (authoritative)
      boot   : 2D array [n_boot, n_grid], NORMALISED likelihood curves
      y_med  : 1D array, median of the NORMALISED curves on the same x grid
    """
    boot_path = BASE_DIR / f"{case}{tier}_bootstrap_curves_boot200.csv"
    agg_path  = BASE_DIR / f"{case}{tier}_lowerdisc_curve_boot200.csv"
    boot_df   = pd.read_csv(boot_path)
    agg_df    = pd.read_csv(agg_path)

    # Pivot bootstraps to [x_grid × runs]
    piv = (boot_df
           .pivot(index="Lower Intercept", columns="run.number",
                  values="normalized.sum.likelihood")
           .sort_index())
    x_boot = (piv.index.values / 1e6).astype(float)  # Ma

    boot = piv.values.T  # [n_boot, n_grid], may contain NaNs
    # Normalise each bootstrap curve to peak=1 for consistent visual scale
    with np.errstate(invalid="ignore", divide="ignore"):
        peak = np.nanmax(boot, axis=1, keepdims=True)
        peak[~np.isfinite(peak)] = 1.0
        boot = np.divide(boot, peak, out=np.zeros_like(boot), where=(peak > 0))

    # Aggregate (median) – resample to bootstrap grid if needed
    x_agg = (agg_df["Lower Intercept"].values / 1e6).astype(float)
    y_agg = agg_df["normalized.sum.likelihood"].values.astype(float)
    # Normalise aggregate to peak=1
    ymax = np.nanmax(y_agg) if np.isfinite(y_agg).any() else 1.0
    y_agg = (y_agg / ymax) if ymax > 0 else y_agg

    if (x_agg.shape == x_boot.shape) and np.allclose(x_agg, x_boot):
        y_med = y_agg
    else:
        # conservative 1D interpolation
        y_med = np.interp(x_boot, x_agg, y_agg, left=np.nan, right=np.nan)

    # If any NaNs remain in y_med (edge), fill from bootstrap median
    if np.isnan(y_med).any():
        y_med_alt = np.nanmedian(boot, axis=0)
        mask = np.isnan(y_med)
        y_med[mask] = y_med_alt[mask]

    return x_boot, boot, y_med

from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths
import numpy as np

def _dx(x):
    return float(np.median(np.diff(np.asarray(x, float)))) if len(x) > 1 else 10.0

from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths

from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths  # keep imports consolidated

def _simple_peaks_with_widths(x, y,
                              smooth_ma=10.0,
                              prom_frac=0.02,
                              prom_min_abs=0.006,
                              min_dist_ma=55.0,
                              min_width_ma=20.0):
    """
    Detect peaks on a lightly smoothed median curve and return:
      pk: indices of detected peaks (numpy array, on original grid)
      widths_pts: FWHM widths (in points) for each pk
      y_s: the smoothed curve used for detection (same length as y)
      dx: median grid step (Ma/pt)
      distance_pts: the min separation in points actually used
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    dx = float(np.median(np.diff(x))) if len(x) > 1 else 10.0

    # smooth and thresholds
    sigma_pts   = max(0.5, (smooth_ma / dx) / 2.355)
    y_s         = gaussian_filter1d(y, sigma_pts, mode="nearest")
    dyn         = float(np.nanmax(y_s) - np.nanmin(y_s))
    prom        = max(prom_frac * dyn, prom_min_abs)
    distance_pts= max(1, int(round(min_dist_ma / dx)))
    width_pts   = max(1, int(round(min_width_ma / dx)))

    # detect (allow flat-topped peaks)
    pk, _ = find_peaks(y_s, prominence=prom, distance=distance_pts, width=width_pts, plateau_size=1)

    widths_pts = peak_widths(y_s, pk, rel_height=0.5)[0] if pk.size else np.array([], float)
    return pk.astype(int), widths_pts, y_s, dx, distance_pts

def _format_pm(med, lo, hi):
    med_r = int(round(med))
    pm    = int(round(max(med - lo, hi - med)))
    return f"{med_r} ± {pm}"

# ---- edge params (tweak if needed) ----
EDGE_FORCE_WINDOW_MA   = 180.0   # how far back we look for a right-edge candidate
EDGE_MIN_RISE          = 0.004   # min rise (normalized units) over that window
EDGE_MIN_RELHEIGHT     = 0.10    # edge peak must be at least 10% of main peak height
EDGE_MIN_SEP_MA        = 35.0    # don't duplicate if an interior peak is already this close
EDGE_USE_BOOT_SPREAD   = True    # widen ± using bootstrap spread if available

def _grid_dx(x):
    return float(np.median(np.diff(np.asarray(x, float)))) if len(x) > 1 else 10.0

def _left_hwhm_ma(x, y_s, j, fallback_ma):
    """Left half-width at half-max (HWHM) on smoothed curve; fallback if no crossing."""
    x = np.asarray(x, float); y_s = np.asarray(y_s, float)
    if j <= 0 or j >= len(y_s): return fallback_ma
    ypk = float(y_s[j]);  target = 0.5 * ypk
    k = j
    while k > 0 and y_s[k] > target:
        k -= 1
    if k == j or k == 0:
        return fallback_ma
    y0, y1 = float(y_s[k]), float(y_s[k+1])
    if y1 == y0:
        x_cross = x[k]
    else:
        frac = (target - y0) / (y1 - y0)
        x_cross = x[k] + frac * (x[k+1] - x[k])
    return max(0.0, float(x[j] - x_cross))

def annotate_peaks_with_ci(ax, x, y_med, x_max,
                           smooth_ma=10.0, prom_frac=0.02, prom_min_abs=0.006,
                           min_dist_ma=55.0, min_width_ma=20.0,
                           max_labels=2):
    """
    Draw markers for *true* peaks found on the smoothed curve; label ONLY the
    first `max_labels` peaks (left→right) as 'age ± error'. No numbering.
    """
    pk, widths_pts, y_s, dx, distance_pts = _simple_peaks_with_widths(
        x, y_med, smooth_ma, prom_frac, prom_min_abs, min_dist_ma, min_width_ma
    )

    # Main = tallest detected peak; if none detected, fall back to the apex
    # on the smoothed curve *only if* it is a genuine local max (or a 3-point plateau).
    if pk.size:
        j_main = int(pk[np.argmax(y_s[pk])])
    else:
        j_guess = int(np.nanargmax(y_s)) if y_s.size else None
        if j_guess is not None and 0 < j_guess < len(y_s) - 1:
            is_local = (y_s[j_guess] >= y_s[j_guess-1] and y_s[j_guess] >= y_s[j_guess+1])
        else:
            is_local = False
        j_main = j_guess if is_local else None

    pk_set = set(int(j) for j in pk)
    if j_main is not None:
        pk_set.add(j_main)

    # widths for local windows
    width_map = {int(j): float(w) for j, w in zip(pk, widths_pts)}
    if j_main is not None and j_main not in width_map:
        width_map[j_main] = max(3.0, (min_width_ma / dx))

    # optional forced right-edge candidate (triangle)
    forced_edges = set()
    j_edge = _force_edge_candidate(x, y_s, after_j=j_main, window_ma=220.0)
    if j_edge is not None:
        sep_pts = int(round(30.0 / dx))
        if all(abs(j_edge - j) > sep_pts for j in pk_set):
            pk_set.add(j_edge)
            forced_edges.add(j_edge)
            width_map.setdefault(j_edge, max(3.0, (min_width_ma / dx)))

    # sort and choose which to label
    j_all = sorted(pk_set, key=lambda jj: x[jj])
    j_labeled = set(j_all[:max_labels]) if max_labels is not None else set(j_all)

    # bootstrap maxima for local CIs
    picks_all = np.asarray(x_max, float)
    picks_all = picks_all[np.isfinite(picks_all)]

    for j in j_all:
        x0 = float(x[j]); y0 = float(y_med[j])
        half_ma = 0.5 * width_map.get(j, (min_width_ma / dx)) * dx

        # triangles ONLY for explicitly forced edge
        is_edge = (j in forced_edges)

        # define picks_here *once* and use it in both branches
        picks_here = picks_all[np.abs(picks_all - x0) <= half_ma]

        # markers
        if is_edge:
            ax.plot([x0], [y0], marker=">", mfc="none", mec="firebrick", ms=5, zorder=6)
        else:
            mkwargs = dict(marker="s", mfc=("firebrick" if (j_main is not None and j == j_main) else "none"),
                           mec="firebrick", ms=4)
            ax.plot([x0], [y0], zorder=6, **mkwargs)

        # label subset as '<age> ± <error>'
        if j not in j_labeled:
            continue

        if is_edge:
            pm = _left_hwhm_ma(x, y_s, j, fallback_ma=max(half_ma, 0.5*min_width_ma))
            if EDGE_USE_BOOT_SPREAD and picks_here.size >= 2:
                if picks_here.size >= 5:
                    lo_b, hi_b = np.percentile(picks_here, [2.5, 97.5])
                else:
                    lo_b, hi_b = float(np.min(picks_here)), float(np.max(picks_here))
                pm = max(pm, 0.5 * float(hi_b - lo_b))
            # draw a small horizontal error bar for the peak
            ax.errorbar(med, y0,
                        xerr=[[med - lo], [hi - med]],
                        fmt="s", mfc="none", mec="crimson", ms=4,
                        ecolor="crimson", lw=1.0, capsize=0, zorder=5)
            # and keep the text if you like
            ax.text(med, y0 + 0.05, f"{int(round(med))} ± {int(round(pm))}",
                    ha="center", va="bottom", fontsize=7, color="crimson")
        else:
            if picks_here.size >= 5:
                lo, hi = np.percentile(picks_here, [2.5, 97.5]); med = float(np.median(picks_here))
            elif picks_here.size == 2:
                lo, hi = float(np.min(picks_here)), float(np.max(picks_here)); med = float(np.median(picks_here))
            elif picks_here.size == 1:
                med = float(picks_here[0]); lo, hi = med - half_ma, med + half_ma
            else:
                med = x0; lo, hi = x0 - half_ma, x0 + half_ma

            pm = max(med - lo, hi - med)
            ax.text(x0, y0 + 0.05, f"{int(round(med))} ± {int(round(pm))}",
                    ha="center", va="bottom", fontsize=7, color="crimson")

def _label_limit_for_case(case: str) -> int:
    # Number of peaks to label (left→right)
    return 5 if str(case) in {"5", "6", "7"} else 2

def _force_edge_candidate(x, y_s, after_j=None, window_ma=220.0):
    """
    Accept a right-edge candidate only if the tail shows a clear rise over the window.
    Returns the index if accepted, else None.
    """
    x = np.asarray(x, float); y_s = np.asarray(y_s, float)
    if len(y_s) < 6: 
        return None
    dx = float(np.median(np.diff(x))) if len(x) > 1 else 10.0
    w  = max(8, int(round(window_ma/dx)))

    j0 = max(1, len(y_s) - w)
    if after_j is not None:
        j0 = max(j0, after_j + max(1, int(round(20.0/dx))))
    j1 = len(y_s) - 2
    if j0 >= j1:
        return None

    # Tail must rise by at least EDGE_MIN_RISE
    if float(y_s[j1]) - float(y_s[j0]) < EDGE_MIN_RISE:
        return None

    # Candidate must be reasonably tall relative to the global peak
    j = j0 + int(np.argmax(y_s[j0:j1+1]))
    if float(y_s[j]) < EDGE_MIN_RELHEIGHT * float(np.nanmax(y_s)):
        return None

    return int(j)



def find_peaks_simple(x, y,
                      smooth_ma=10.0,      # lighter smoothing than before
                      prom_frac=0.02,      # 2% of dynamic range (was 4%)
                      prom_min_abs=0.006,  # absolute floor, smaller to catch shoulders
                      min_dist_ma=55.0,    # allow peaks closer together
                      min_width_ma=20.0):  # count narrower bumps
    """Return main index and list of other peak indices (on original grid)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    dx = _dx(x)
    sigma_pts = max(0.5, (smooth_ma / dx) / 2.355)
    y_s = gaussian_filter1d(y, sigma_pts, mode="nearest")

    dyn = float(np.nanmax(y_s) - np.nanmin(y_s))
    prom = max(prom_frac * dyn, prom_min_abs)
    distance_pts = max(1, int(round(min_dist_ma / dx)))
    width_pts    = max(1, int(round(min_width_ma / dx)))

    pk, _ = find_peaks(y_s, prominence=prom, distance=distance_pts, width=width_pts)

    j_main = int(np.nanargmax(y)) if y.size else None
    others = [int(j) for j in pk if j_main is not None and abs(int(j) - j_main) > distance_pts // 2]
    return j_main, others

def mark_peaks_simple(ax, x, y_med, **kw):
    j_main, others = find_peaks_simple(x, y_med, **kw)
    if j_main is None:
        return
    ax.plot([x[j_main]], [y_med[j_main]], marker="s", mfc="firebrick", mec="firebrick", ms=4, zorder=5)
    ax.text(x[j_main], 0.98, f"{x[j_main]:.0f}", ha="center", va="top",
            fontsize=7, color="crimson", transform=ax.get_xaxis_transform())
    for j in others:
        ax.plot([x[j]], [y_med[j]], marker="s", mfc="none", mec="firebrick", ms=4, zorder=5)

def _dd_median_ci_of_maxima(boot, x):
    """Return (med, lo, hi, x_max) for per-bootstrap maxima on grid x."""
    if boot.size == 0:
        return None
    # argmax with NaN guard
    boot_safe = np.where(np.isfinite(boot), boot, -np.inf)
    idx       = np.argmax(boot_safe, axis=1)
    x_max     = x[idx]
    x_max     = x_max[np.isfinite(x_max)]
    if x_max.size == 0:
        return None
    med = float(np.median(x_max))
    lo, hi = np.percentile(x_max, [2.5, 97.5]) if x_max.size >= 3 else (med, med)
    return med, float(lo), float(hi), x_max

from scipy.signal import find_peaks, peak_widths

def dd_bootstrap_maxima(boot, x):
    boot_safe = np.where(np.isfinite(boot), boot, -np.inf)
    idx = np.argmax(boot_safe, axis=1)
    ages = x[idx]
    return ages[np.isfinite(ages)]

def plot_dd_grid(cases_to_plot, tiers=TIERS,
                 title=None, out_prefix=None, save=False, show=True):
    nrows, ncols = len(cases_to_plot), len(tiers)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(3.2*ncols, 3.0*nrows),
                             sharex=True, sharey=True)

    # Ensure 2D axes
    if nrows == 1 and ncols == 1: axes = np.array([[axes]])
    elif nrows == 1:              axes = axes.reshape(1, ncols)
    elif ncols == 1:              axes = axes.reshape(nrows, 1)

    for r, case in enumerate(cases_to_plot):
        for c, tier in enumerate(tiers):
            ax = axes[r, c]
            ax.set_xlim(0, 2000)
            ax.set_ylim(0, 1.08)
            ax.set_autoscale_on(False)
            try:
                x, boot, y_med = _load_dd_panel(case, tier)
            except FileNotFoundError as e:
                ax.text(0.5, 0.5, f"Missing data {case}{tier}\n{e}",
                        ha="center", va="center", transform=ax.transAxes, fontsize=8)
                ax.axis("off")
                continue

            # background: all bootstrap curves
            for yb in boot:
                ax.plot(x, yb, color="0.75", lw=0.4, alpha=0.6, zorder=1)

            # rug: per-bootstrap maxima
            x_max = dd_bootstrap_maxima(boot, x)
            ax.scatter(x_max, np.full_like(x_max, 1.03), marker="|", s=18, color="0.5",
                    alpha=0.5, transform=ax.get_xaxis_transform(), zorder=3)
            # --- DD-PEAKS (single-curve, multiscale) ---
            # annotate_peaks_with_ci(
            #     ax, x, y_med, x_max,
            #     smooth_ma=10.0,    # more smoothing
            #     prom_frac=0.02,    # higher prominence
            #     prom_min_abs=0.005,
            #     min_dist_ma=60.0,  # further apart
            #     min_width_ma=25.0,
            #     max_labels=_label_limit_for_case(case)
            # )

            # median curve
            ax.plot(x, y_med, color="k", lw=1.5, zorder=2)

            # dashed verticals at true ages
            for age in CASES.get(case, []):
                ax.axvline(age, ls="--", lw=0.8, color=TRUE_COLOR, zorder=0)

            # --- DD REPORTED STATISTIC: median of bootstrap maxima ±95% CI ---
            mc = _dd_median_ci_of_maxima(boot, x)
            if mc is not None:
                x_med, lo, hi, x_max = mc

                # rug of per-bootstrap maxima along the top (same as before)
                ax.scatter(
                    x_max, np.full_like(x_max, 1.03),
                    marker="|", s=18, color="0.5", alpha=0.5, zorder=3,
                    transform=ax.get_xaxis_transform(),  # y in axes coords
                )

                # shaded CI band + age label (CDC-style)
                draw_dd_ci_band(ax, x_med, lo, hi)

            # cosmetics
            ax.set_xlim(0, 2000)
            ax.set_ylim(0, 1.1)  # extra headroom for rugs/labels
            if c == ncols - 1:
                ax.text(1.04, 0.5, f"Case {case}",
                        transform=ax.transAxes, rotation=-90,
                        va="center", ha="left", fontsize=9, fontweight="bold")
            show_xlabels = (r == nrows - 1)
            show_ylabels = (c == 0)
            ax.tick_params(
                direction='in',
                labelsize=7,
                pad=2,
                labelbottom=show_xlabels,
                labelleft=show_ylabels,
            )
    # Axis labels on representative axes so they sit close to the plots
    # x-label on bottom middle axis
    axes[-1, ncols // 2].set_xlabel("Lower-intercept age (Ma)", fontsize=9)

    # y-label on middle left axis
    axes[nrows // 2, 0].set_ylabel("Normalised likelihood", fontsize=9)


    fig.tight_layout()
    plt.show()
    # if title:
    #     fig.suptitle(title, fontsize=12, y=1.02)
    #     fig.subplots_adjust(top=0.90)

    # if save and out_prefix:
    #     out_pdf = BASE_DIR / f"{out_prefix}.pdf"
    #     out_png = BASE_DIR / f"{out_prefix}.png"
    #     fig.savefig(out_pdf)
    #     fig.savefig(out_png, dpi=300)
    #     print(f"✓ wrote {out_pdf.name} and {out_png.name}")

    # if show:
    #     plt.show()
    # else:
    #     plt.close(fig)


# ─────────────────────────── run: 1–4 and 5–7 ────────────────────────────────
plot_dd_grid(
    cases_to_plot=["1", "2", "3", "4"],
    title="Discordance‑dating likelihood surfaces — Tiers A–C × Cases 1–4",
    out_prefix="DD_likelihood_grid_tiersAtoC_cases1to4",
    save=False, show=True
)
plot_dd_grid(
    cases_to_plot=["5", "6", "7"],
    title="Discordance‑dating likelihood surfaces — Tiers A–C × Cases 5–7",
    out_prefix="DD_likelihood_grid_tiersAtoC_cases5to7",
    save=False, show=True
)

