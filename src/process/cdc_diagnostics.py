"""Optional CDC diagnostics and paper-export utilities.

Everything in here should be safe to *exclude* from normal GUI usage:
- No outputs are written unless CDC_WRITE_OUTPUTS=1 (and/or CDC_KS_EXPORT_DIR is set).
- Paths are configurable via environment variables (see process.cdc_config).

Keeping this separate from the main processing pipeline makes it much easier to:
- build a clean GUI release; and
- keep manuscript reproduction artifacts in one place.
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
from process.cdc_population import concordant_ages_ma
from process.cdc_utils import safe_prefix

try:
    import resource  # Unix only
except ImportError:  # pragma: no cover
    resource = None


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
            fieldnames=["sample", "peak_no", "age_ma", "ci_low", "ci_high", "support"],
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

    use_pen = bool(getattr(settings, "penaliseInvalidAges", False))
    tag = "pen" if use_pen else "raw"

    KS_EXPORT_ROOT.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # Choose which age is treated as the "optimal rim" for UI CDF export:
    #   - Prefer UI median-of-run-optima if provided
    #   - Otherwise fall back to the curve optimum (legacy behaviour)
    # ------------------------------------------------------------
    if ui_opt_years is not None and np.isfinite(ui_opt_years):
        rim_opt_ma = float(ui_opt_years) / 1e6
        rim_opt_int = int(round(rim_opt_ma))
    else:
        D_for_opt = D_pen if use_pen else D_raw
        D_for_opt = np.where(np.isfinite(D_for_opt), D_for_opt, np.inf)
        i_opt = int(np.nanargmin(D_for_opt))
        rim_opt_ma = float(ages_ma[i_opt])
        rim_opt_int = int(round(rim_opt_ma))

    # 1D curve export
    good_path = KS_EXPORT_ROOT / f"KS_goodness_{tag}.csv"
    with good_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["age_Ma", "S_raw", "S_pen", "D_raw", "D_pen", "n_runs"])
        for a, sr, spv, dr, dp in zip(ages_ma, S_raw, S_pen, D_raw, D_pen):
            w.writerow([f"{a:.6f}", f"{sr:.6f}", f"{spv:.6f}", f"{dr:.6f}", f"{dp:.6f}", n_runs])

    # Summary export (so UI optimum/CI is reproducible from exports)
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

    summ_path = KS_EXPORT_ROOT / f"KS_optimum_summary_{tag}.csv"
    with summ_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["opt_ui_Ma", "low95_ui_Ma", "high95_ui_Ma", "opt_surface_Ma", "n_runs"])
        w.writerow(
            [
                _ma_or_blank(ui_opt_years),
                _ma_or_blank(ui_low95_years),
                _ma_or_blank(ui_high95_years),
                _ma_or_blank(legacy_opt_years),
                n_runs,
            ]
        )

    # Per-run optima export (lets you rebuild median/CI exactly)
    if run_optima_years is not None:
        opt_path = KS_EXPORT_ROOT / f"KS_run_optima_{tag}.csv"
        with opt_path.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["run", "opt_age_Ma"])
            vals = np.asarray(run_optima_years, float) / 1e6
            for i, t in enumerate(vals):
                if np.isfinite(t):
                    w.writerow([i, f"{t:.6f}"])

    # Per-sample folder
    subdir = KS_EXPORT_ROOT / f"KS_failure_{tag}"
    subdir.mkdir(parents=True, exist_ok=True)

    # TW coordinates
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

    # Concordant ages (for ECDF)
    conc_ages = concordant_ages_ma(conc_spots)
    conc_ages = conc_ages[np.isfinite(conc_ages)]
    conc_ages.sort()
    with (subdir / "cdf_concordant.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["# age_Ma"])
        for t in conc_ages:
            w.writerow([f"{t:.6f}"])

    # UI age CDFs at 300, chosen optimum (file name rounded), 1800
    for rim in (300, rim_opt_int, 1800):
        rim_for_calc = float(rim_opt_ma) if rim == rim_opt_int else float(rim)
        ui_ma = ks_ui_ages_for_rim_Ma(disc_spots, rim_for_calc)
        header = f"# age_Ma_UI_rim{rim_opt_ma:.3f}" if rim == rim_opt_int else f"# age_Ma_UI_rim{rim}"
        with (subdir / f"cdf_UI_{rim}.csv").open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow([header])
            for t in ui_ma:
                w.writerow([f"{t:.6f}"])
