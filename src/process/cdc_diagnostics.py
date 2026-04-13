"""Optional CDC diagnostics and paper-export utilities.

- No outputs are written unless CDC_WRITE_OUTPUTS=1 (and/or CDC_KS_EXPORT_DIR is set).
- Paths are configurable via environment variables (see process.cdc_config).

"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, Sequence

import numpy as np

from process import calculations
from process.cdc_config import (
    CDC_ENABLE_RUNLOG,
    CDC_WRITE_OUTPUTS,
    CATALOGUE_CSV_PEN,
    CATALOGUE_CSV_RAW,
    DIAG_DIR,
    KS_EXPORT_ROOT,
    RUN_FIELDS,
    RUNLOG,
)
from process.cdc_tw import age_ma_from_pb207pb206, age_ma_from_u238pb206
from process.cdc_utils import safe_prefix

try:
    import resource  # Unix only
except ImportError:  # pragma: no cover
    resource = None


CATALOGUE_CI_METHOD = "stability_bounds"
CATALOGUE_CI_INTERPRETATION = "bootstrap_percentile_stability_bounds_of_assigned_run_ages"


def _spot_age_proxy_ma(spot) -> float:
    """Stable age proxy (Ma) from TW coordinates for one spot."""
    t = age_ma_from_pb207pb206(spot.pbPbValue)
    if np.isfinite(t):
        return float(t)
    t2 = age_ma_from_u238pb206(spot.uPbValue)
    return float(t2) if np.isfinite(t2) else float("nan")


def concordant_ages_ma(spots):
    """Approximate concordant ages (Ma) used for diagnostics exports."""
    return np.asarray([_spot_age_proxy_ma(s) for s in spots], float)


def ensure_output_dirs() -> None:
    """Create output directories for diagnostics if CDC_WRITE_OUTPUTS is enabled."""
    if not CDC_WRITE_OUTPUTS:
        return
    CATALOGUE_CSV_PEN.parent.mkdir(parents=True, exist_ok=True)
    CATALOGUE_CSV_RAW.parent.mkdir(parents=True, exist_ok=True)
    RUNLOG.parent.mkdir(parents=True, exist_ok=True)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)


def reset_csv(path: Path, header: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="") as fh:
        fh.write(header + "\n")


def rss_mb() -> float:
    """Return peak RSS in MB if available; NaN otherwise."""
    if resource is None:
        return float("nan")
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru / (1024 * 1024.0) if sys.platform == "darwin" else ru / 1024.0


def write_runlog(row: dict) -> None:
    if not CDC_ENABLE_RUNLOG:
        return
    RUNLOG.parent.mkdir(parents=True, exist_ok=True)
    write_header = not RUNLOG.exists()
    with RUNLOG.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=RUN_FIELDS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)


def append_catalogue_rows(sample_name: str, rows: Sequence[Dict], dest_path: Path) -> None:
    if not rows:
        return
    dest_path = Path(dest_path)
    write_header = not dest_path.exists()
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with dest_path.open("a", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "sample",
                "peak_no",
                "age_ma",
                "ci_low",
                "ci_high",
                "support",
                "support_low",
                "support_high",
                "stability_low",
                "stability_high",
                "age_mode",
                "ci_method",
                "ci_interpretation",
                "stability_method",
            ],
            extrasaction="ignore",
        )
        if write_header:
            w.writeheader()
        for i, r in enumerate(rows, 1):
            w.writerow(
                dict(
                    sample=sample_name,
                    peak_no=r.get("peak_no", i),
                    age_ma=r["age_ma"],
                    ci_low=r["ci_low"],
                    ci_high=r["ci_high"],
                    support=r.get("support", float("nan")),
                    support_low=r.get("support_low", r["ci_low"]),
                    support_high=r.get("support_high", r["ci_high"]),
                    stability_low=r.get("stability_low", r["ci_low"]),
                    stability_high=r.get("stability_high", r["ci_high"]),
                    age_mode=r.get("age_mode", "vote_median"),
                    ci_method=r.get("ci_method", CATALOGUE_CI_METHOD),
                    ci_interpretation=r.get("ci_interpretation", CATALOGUE_CI_INTERPRETATION),
                    stability_method=r.get("stability_method", "vote_percentile"),
                )
            )


def write_npz_diagnostics(
    sample_name: str,
    ages_ma: np.ndarray,
    ages_y: np.ndarray,
    runs,
    S_runs_raw: np.ndarray,
    S_runs_pen: np.ndarray,
    Smed_raw: np.ndarray,
    Smed_pen: np.ndarray,
    S_view: np.ndarray,
    rows_for_ui: Sequence[Dict],
) -> None:
    """Write the NPZ diagnostics used for manuscript figures (guarded by CDC_WRITE_OUTPUTS)."""
    if not CDC_WRITE_OUTPUTS:
        return
    ensure_output_dirs()
    prefix = safe_prefix(sample_name)

    np.savez_compressed(
        DIAG_DIR / f"{prefix}_runs_S.npz",
        age_Ma=ages_ma,
        S_runs_raw=S_runs_raw,
        S_runs_pen=S_runs_pen,
        optima_Ma=np.array([r.optimal_pb_loss_age / 1e6 for r in runs], float),
    )

    np.savez_compressed(
        DIAG_DIR / f"{prefix}_ensemble_surfaces.npz",
        age_Ma=ages_ma,
        Smed_raw=Smed_raw,
        Smed_pen=Smed_pen,
        S_view=S_view,
        peaks_age_Ma=np.array([r["age_ma"] for r in rows_for_ui], float) if rows_for_ui else np.array([], float),
        peaks_ci_low=np.array([r["ci_low"] for r in rows_for_ui], float) if rows_for_ui else np.array([], float),
        peaks_ci_high=np.array([r["ci_high"] for r in rows_for_ui], float) if rows_for_ui else np.array([], float),
        peaks_support_low=np.array([r.get("support_low", r["ci_low"]) for r in rows_for_ui], float) if rows_for_ui else np.array([], float),
        peaks_support_high=np.array([r.get("support_high", r["ci_high"]) for r in rows_for_ui], float) if rows_for_ui else np.array([], float),
        peaks_stability_low=np.array([r.get("stability_low", r["ci_low"]) for r in rows_for_ui], float) if rows_for_ui else np.array([], float),
        peaks_stability_high=np.array([r.get("stability_high", r["ci_high"]) for r in rows_for_ui], float) if rows_for_ui else np.array([], float),
        peaks_support=np.array([r.get("support", float("nan")) for r in rows_for_ui], float) if rows_for_ui else np.array([], float),
    )

    # Per-run exports (large; keep behind CDC_WRITE_OUTPUTS)
    for r_idx, r in enumerate(runs, start=1):
        d_raw = np.array([r.statistics_by_pb_loss_age[a].test_statistics[0] for a in ages_y], float)
        d_pen = np.array([r.statistics_by_pb_loss_age[a].score for a in ages_y], float)
        np.savez_compressed(
            DIAG_DIR / f"{prefix}_{r_idx:03d}.npz",
            age_Ma=ages_ma,
            D_raw=d_raw,
            D_pen=d_pen,
            S_raw=1.0 - d_raw,
            S_pen=1.0 - d_pen,
            opt_Ma=float(r.optimal_pb_loss_age / 1e6),
        )


def ks_ui_ages_for_rim_Ma(discordantSpots, rim_Ma: float) -> np.ndarray:
    """
    Reconstruct upper-intercept ages (Ma) for a trial lower-intercept age rim_Ma.

    Uses calculations.discordant_age along the chord between the point on
    TW concordia at rim_Ma and each discordant spot.
    """
    t_low = float(rim_Ma) * 1e6
    x_low = calculations.u238pb206_from_age(t_low)
    y_low = calculations.pb207pb206_from_age(t_low)

    ui_list = []
    for spot in discordantSpots:
        x = float(spot.uPbValue)
        y = float(spot.pbPbValue)

        # calculations.discordant_age expects x1 > x2
        if x_low > x:
            x1, y1, x2, y2 = x_low, y_low, x, y
        else:
            x1, y1, x2, y2 = x, y, x_low, y_low

        t_up = calculations.discordant_age(x1, y1, x2, y2)  # years or None
        if t_up is None or not np.isfinite(t_up):
            continue
        if t_up <= t_low:
            continue

        ui_list.append(t_up / 1e6)  # store in Ma

    ui = np.asarray(ui_list, float)
    ui = ui[np.isfinite(ui)]
    ui.sort()
    return ui


def _ma_or_blank(x):
    if x is None:
        return ""
    try:
        x = float(x)
    except Exception:
        return ""
    if not np.isfinite(x):
        return ""
    return f"{(x / 1e6):.6f}"


def _surface_optimum_years(d_curve: np.ndarray, ages_y: np.ndarray) -> float:
    d_curve = np.where(np.isfinite(d_curve), d_curve, np.inf)
    return float(ages_y[int(np.nanargmin(d_curve))])


def _run_optima_years(runs, ages_y: np.ndarray, which: str) -> np.ndarray:
    vals = []
    for r in runs:
        if which == "pen":
            vals.append(float(getattr(r, "optimal_pb_loss_age", np.nan)))
            continue
        d_by_age = np.array([r.statistics_by_pb_loss_age[a].test_statistics[0] for a in ages_y], float)
        d_by_age = np.where(np.isfinite(d_by_age), d_by_age, np.inf)
        vals.append(float(ages_y[int(np.nanargmin(d_by_age))]))
    return np.asarray(vals, float)


def export_legacy_ks(
    sample,
    settings,
    runs,
    ages_y,
    D_raw=None,
    D_pen=None,
    ui_opt_years=None,
    ui_low95_years=None,
    ui_high95_years=None,
    ui_stability_low_years=None,
    ui_stability_high_years=None,
    run_optima_years=None,
    legacy_opt_years=None,
) -> None:
    """Export the legacy KS goodness curve and CDF files used for a manuscript figure."""
    if KS_EXPORT_ROOT is None:
        return
    if not runs:
        return

    # Only export once complete
    try:
        if len(runs) < int(settings.monteCarloRuns):
            return
    except Exception:
        pass

    conc_spots = getattr(sample, "_ks_concordantSpots", None)
    disc_spots = getattr(sample, "_ks_discordantSpots", None)
    if not conc_spots or not disc_spots:
        return

    ages_y = np.asarray(ages_y, float)
    ages_ma = ages_y / 1e6
    n_runs = len(runs)

    # Compute curves if not supplied
    if D_raw is None:
        D_raw = np.array(
            [np.mean([r.statistics_by_pb_loss_age[a].test_statistics[0] for r in runs]) for a in ages_y],
            dtype=float,
        )
    if D_pen is None:
        D_pen = np.array(
            [np.mean([r.statistics_by_pb_loss_age[a].score for r in runs]) for a in ages_y],
            dtype=float,
        )

    # Goodness curves
    S_raw = 1.0 - D_raw
    S_pen = 1.0 - D_pen

    KS_EXPORT_ROOT.mkdir(parents=True, exist_ok=True)

    run_optima_by_tag = {
        "raw": _run_optima_years(runs, ages_y, "raw"),
        "pen": _run_optima_years(runs, ages_y, "pen"),
    }

    # Preserve caller-provided optima for the active channel when available.
    active_tag = "pen" if bool(getattr(settings, "penaliseInvalidAges", False)) else "raw"
    if run_optima_years is not None:
        run_optima_by_tag[active_tag] = np.asarray(run_optima_years, float)

    conc_ages = concordant_ages_ma(conc_spots)
    conc_ages = conc_ages[np.isfinite(conc_ages)]
    conc_ages.sort()

    for tag, d_curve in (("raw", D_raw), ("pen", D_pen)):
        opt_years = run_optima_by_tag[tag]
        opt_years = opt_years[np.isfinite(opt_years)]
        stability_low = stability_high = float("nan")
        if opt_years.size:
            ui_opt = float(np.median(opt_years))
            stability_low = float(np.quantile(opt_years, 0.025))
            stability_high = float(np.quantile(opt_years, 0.975))
        else:
            ui_opt = float("nan")

        ui_low = stability_low
        ui_high = stability_high
        if tag == active_tag:
            if ui_low95_years is not None and np.isfinite(float(ui_low95_years)):
                ui_low = float(ui_low95_years)
            if ui_high95_years is not None and np.isfinite(float(ui_high95_years)):
                ui_high = float(ui_high95_years)
            if ui_opt_years is not None and np.isfinite(float(ui_opt_years)):
                ui_opt = float(ui_opt_years)
            if ui_stability_low_years is not None and np.isfinite(float(ui_stability_low_years)):
                stability_low = float(ui_stability_low_years)
            if ui_stability_high_years is not None and np.isfinite(float(ui_stability_high_years)):
                stability_high = float(ui_stability_high_years)

        rim_opt_ma = float(ui_opt) / 1e6 if np.isfinite(ui_opt) else float(_surface_optimum_years(d_curve, ages_y) / 1e6)
        rim_opt_int = int(round(rim_opt_ma))
        surface_opt = _surface_optimum_years(d_curve, ages_y)

        good_path = KS_EXPORT_ROOT / f"KS_goodness_{tag}.csv"
        with good_path.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["age_Ma", "S_raw", "S_pen", "D_raw", "D_pen", "n_runs"])
            for a, sr, spv, dr, dp in zip(ages_ma, S_raw, S_pen, D_raw, D_pen):
                w.writerow([f"{a:.6f}", f"{sr:.6f}", f"{spv:.6f}", f"{dr:.6f}", f"{dp:.6f}", n_runs])

        summ_path = KS_EXPORT_ROOT / f"KS_optimum_summary_{tag}.csv"
        with summ_path.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(
                [
                    "opt_ui_Ma",
                    "support_low_ui_Ma",
                    "support_high_ui_Ma",
                    "stability_low_ui_Ma",
                    "stability_high_ui_Ma",
                    "opt_surface_Ma",
                    "n_runs",
                ]
            )
            w.writerow(
                [
                    _ma_or_blank(ui_opt),
                    _ma_or_blank(ui_low),
                    _ma_or_blank(ui_high),
                    _ma_or_blank(stability_low),
                    _ma_or_blank(stability_high),
                    _ma_or_blank(surface_opt),
                    n_runs,
                ]
            )

        opt_path = KS_EXPORT_ROOT / f"KS_run_optima_{tag}.csv"
        with opt_path.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["run", "opt_age_Ma"])
            for i, t in enumerate(opt_years):
                if np.isfinite(t):
                    w.writerow([i, f"{(float(t) / 1e6):.6f}"])

        subdir = KS_EXPORT_ROOT / f"KS_failure_{tag}"
        subdir.mkdir(parents=True, exist_ok=True)

        with (subdir / "concordant.csv").open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["uPb", "pbPb"])
            for s in conc_spots:
                w.writerow([float(s.uPbValue), float(s.pbPbValue)])

        with (subdir / "discordant.csv").open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["uPb", "pbPb"])
            for s in disc_spots:
                w.writerow([float(s.uPbValue), float(s.pbPbValue)])

        with (subdir / "cdf_concordant.csv").open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["# age_Ma"])
            for t in conc_ages:
                w.writerow([f"{t:.6f}"])

        for rim in (300, rim_opt_int, 1800):
            rim_for_calc = float(rim_opt_ma) if rim == rim_opt_int else float(rim)
            ui_ma = ks_ui_ages_for_rim_Ma(disc_spots, rim_for_calc)
            header = f"# age_Ma_UI_rim{rim_opt_ma:.3f}" if rim == rim_opt_int else f"# age_Ma_UI_rim{rim}"
            with (subdir / f"cdf_UI_{rim}.csv").open("w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow([header])
                for t in ui_ma:
                    w.writerow([f"{t:.6f}"])
