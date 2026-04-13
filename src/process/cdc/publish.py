"""Final result publication for CDC runs.

This module translates the internal CDC state into:
- sample attributes used by the GUI
- emitted progress/signals payloads
- CSV and NPZ diagnostics exports
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from process.cdc.state import ProgressType
from process.cdc.guards import _snap_rows_to_curve
from process.cdc_config import (
    CATALOGUE_CSV_PEN,
    CATALOGUE_CSV_RAW,
    CDC_WRITE_OUTPUTS,
    KS_EXPORT_ROOT,
    RUNLOG,
)
from process.cdc_diagnostics import (
    append_catalogue_rows as _append_catalogue_rows,
    export_legacy_ks as _export_legacy_ks,
    reset_csv as _reset_csv,
    write_npz_diagnostics as _write_npz_diagnostics,
)
from utils.peakHelpers import fmt_peak_stats


def reset_output_exports():
    """Reset CSV outputs for a fresh CDC batch run when writing is enabled."""
    if not CDC_WRITE_OUTPUTS:
        return
    _reset_csv(
        CATALOGUE_CSV_PEN,
        "sample,peak_no,age_ma,ci_low,ci_high,support,support_low,support_high,stability_low,stability_high,age_mode,ci_method,ci_interpretation,stability_method",
    )
    _reset_csv(
        CATALOGUE_CSV_RAW,
        "sample,peak_no,age_ma,ci_low,ci_high,support,support_low,support_high,stability_low,stability_high,age_mode,ci_method,ci_interpretation,stability_method",
    )
    _reset_csv(
        RUNLOG,
        "method,phase,sample,tier,R,n_grid,elapsed_s,per_run_median_s,per_run_p95_s,rss_peak_mb,python,numpy",
    )


def _emit_summedKS(signals, sample, progress, ages_ma, y_curve, rows_for_ui):
    """
    Send the plotted curve and peak arrays to the sample signal and legacy bus.
    """
    plot_rows = [dict(r) for r in rows_for_ui if str(r.get("mode", "")) != "recent_boundary"]
    ui_peaks_age = [float(r["age_ma"]) for r in plot_rows]
    ui_peaks_ci = [[float(r["ci_low"]), float(r["ci_high"])] for r in plot_rows]
    ui_support = [float(r.get("support", float("nan"))) for r in plot_rows]

    sample.summedKS_peaks_Ma = np.asarray(ui_peaks_age, float)
    sample.summedKS_ci_low_Ma = np.asarray([lo for lo, _ in ui_peaks_ci], float)
    sample.summedKS_ci_high_Ma = np.asarray([hi for _, hi in ui_peaks_ci], float)

    payload = (ages_ma.tolist(), y_curve.tolist(), ui_peaks_age, ui_peaks_ci, ui_support)
    if hasattr(sample.signals, "summedKS"):
        try:
            sample.signals.summedKS.emit(payload)
        except (AttributeError, RuntimeError, TypeError):
            import traceback
            traceback.print_exc()
    try:
        signals.progress("summedKS", progress, sample.name, payload)
    except TypeError:
        try:
            signals.progress("summedKS", progress, sample.name, (payload[0], payload[1], payload[2]))
        except (AttributeError, RuntimeError, TypeError):
            pass


def _public_interval_row(row: Dict) -> Dict:
    """Return one row with stability bounds promoted to the public interval."""
    rr = dict(row)
    stability_low = float(rr.get("stability_low", rr.get("ci_low", np.nan)))
    stability_high = float(rr.get("stability_high", rr.get("ci_high", np.nan)))
    support_low = float(rr.get("support_low", rr.get("ci_low", np.nan)))
    support_high = float(rr.get("support_high", rr.get("ci_high", np.nan)))

    if (not np.isfinite(support_low)) or (not np.isfinite(support_high)) or (support_high <= support_low):
        support_low, support_high = stability_low, stability_high

    age = float(rr.get("age_ma", np.nan))
    if np.isfinite(age):
        if np.isfinite(support_low) and age < support_low:
            support_low = age
        if np.isfinite(support_high) and age > support_high:
            support_high = age

    rr["stability_low"] = stability_low
    rr["stability_high"] = stability_high
    rr["support_low"] = support_low
    rr["support_high"] = support_high
    rr["ci_low"] = stability_low
    rr["ci_high"] = stability_high
    # Public rows expose stability bounds as the reported interval.
    rr["ci_method"] = "stability_bounds"
    rr["ci_interpretation"] = "bootstrap_percentile_stability_bounds_of_assigned_run_ages"
    rr["stability_method"] = str(rr.get("stability_method", "vote_percentile"))
    return rr


def _public_interval_rows(rows: List[Dict]) -> List[Dict]:
    """Promote stability bounds to the public interval for a row sequence."""
    return [_public_interval_row(r) for r in (rows or [])]


def _publish_legacy_only(
    signals, sample, progress, settings, runs, ages_ma, ages_y,
    S_optimal_curve, optimalAge, lower95, upper95, opt_all,
    meanD, meanP, meanInv, meanSc, mean_primary,
):
    """Publish the legacy single-age result when ensemble peak-picking is disabled."""
    sample.peak_catalogue = []
    _emit_summedKS(signals, sample, progress, ages_ma, S_optimal_curve, rows_for_ui=[])

    if KS_EXPORT_ROOT is not None:
        _export_legacy_ks(
            sample,
            settings,
            runs,
            ages_y,
            D_pen=mean_primary,
            ui_opt_years=optimalAge,
            ui_low95_years=lower95,
            ui_high95_years=upper95,
            run_optima_years=opt_all,
            legacy_opt_years=optimalAge,
        )

    payload = (optimalAge, lower95, upper95, meanD, meanP, meanInv, meanSc, "—", [])
    try:
        signals.progress(ProgressType.OPTIMAL, 1.0, sample.name, payload)
    except TypeError:
        signals.progress(ProgressType.OPTIMAL, 1.0, sample.name, payload[:7])


def _publish_results(
    signals, sample, progress, settings, runs, raw, pen,
    rows_for_ui, rejected_rows, ages_ma, ages_y, S_view,
    optimalAge, lower95, upper95, opt_all,
    meanD, meanP, meanInv, meanSc,
):
    """Renumber peaks, export CSVs/NPZs, and publish the final UI payload."""
    for row_list in (rows_for_ui, raw.rows, pen.rows):
        for i, r in enumerate(row_list, 1):
            r["peak_no"] = i

    public_rows_for_ui = _public_interval_rows(rows_for_ui)
    public_raw_rows = _public_interval_rows(raw.rows)
    public_pen_rows = _public_interval_rows(pen.rows)

    if CDC_WRITE_OUTPUTS:
        _append_catalogue_rows(sample.name, public_pen_rows, dest_path=CATALOGUE_CSV_PEN)
        _append_catalogue_rows(sample.name, public_raw_rows, dest_path=CATALOGUE_CSV_RAW)
        _write_npz_diagnostics(
            sample_name=sample.name,
            ages_ma=ages_ma,
            ages_y=ages_y,
            runs=runs,
            S_runs_raw=raw.S_runs,
            S_runs_pen=pen.S_runs,
            Smed_raw=raw.Smed,
            Smed_pen=pen.Smed,
            S_view=S_view,
            rows_for_ui=public_rows_for_ui,
        )

    rows_for_plot = _public_interval_rows(_snap_rows_to_curve(rows_for_ui, ages_ma, S_view))
    published_peak_tuples = [(r["age_ma"], r["ci_low"], r["ci_high"], r["support"]) for r in public_rows_for_ui]
    peak_str = fmt_peak_stats(published_peak_tuples) if published_peak_tuples else "—"

    plot_peaks = np.asarray([float(r.get("age_ma", np.nan)) for r in rows_for_plot], float)
    plot_ci_low = np.asarray([float(r.get("ci_low", np.nan)) for r in rows_for_plot], float)
    plot_ci_high = np.asarray([float(r.get("ci_high", np.nan)) for r in rows_for_plot], float)
    sample.summedKS_peaks_Ma = plot_peaks
    sample.summedKS_ci_low_Ma = plot_ci_low
    sample.summedKS_ci_high_Ma = plot_ci_high
    sample.peak_uncertainty_str = peak_str

    detailed_catalogue = [
        dict(
            sample=sample.name,
            peak_no=i + 1,
            ci_low=lo,
            age_ma=med,
            ci_high=hi,
            support=sup,
            direct_support=float(r.get("direct_support", sup)),
            winner_support=float(r.get("winner_support", sup)),
            support_low=float(r.get("support_low", lo)),
            support_high=float(r.get("support_high", hi)),
            stability_low=float(r.get("stability_low", lo)),
            stability_high=float(r.get("stability_high", hi)),
            age_mode=str(r.get("age_mode", "vote_median")),
            peak_left_edge_ma=r.get("peak_left_edge_ma", np.nan),
            peak_right_edge_ma=r.get("peak_right_edge_ma", np.nan),
            peak_half_prom_width_frac=r.get("peak_half_prom_width_frac", np.nan),
            peak_right_left_ratio=r.get("peak_right_left_ratio", np.nan),
            selection=r.get("selection", "strict"),
            mode=r.get("mode", ""),
            label=r.get("label", ""),
            ci_method="stability_bounds",
            ci_interpretation="bootstrap_percentile_stability_bounds_of_assigned_run_ages",
            stability_method=str(r.get("stability_method", "vote_percentile")),
        )
        for i, (r, (med, lo, hi, sup)) in enumerate(zip(public_rows_for_ui, published_peak_tuples))
    ]
    sample.peak_catalogue = detailed_catalogue

    _emit_summedKS(signals, sample, progress, ages_ma, S_view, rows_for_plot)

    if KS_EXPORT_ROOT is not None:
        _export_legacy_ks(
            sample,
            settings,
            runs,
            ages_y,
            ui_opt_years=optimalAge,
            ui_low95_years=lower95,
            ui_high95_years=upper95,
            run_optima_years=opt_all,
            legacy_opt_years=optimalAge,
        )

    payload = (
        optimalAge,
        lower95,
        upper95,
        meanD,
        meanP,
        meanInv,
        meanSc,
        peak_str,
        detailed_catalogue,
        {"rejected_peak_candidates": list(rejected_rows or [])},
    )
    try:
        signals.progress(ProgressType.OPTIMAL, 1.0, sample.name, payload)
    except TypeError:
        signals.progress(ProgressType.OPTIMAL, 1.0, sample.name, payload[:7])
