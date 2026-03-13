#!/usr/bin/env python3
"""Run a manuscript-style CDC audit across synthetic Cases 1-7.

This script evaluates one locked ensemble-only CDC settings profile, then
compares the current outputs against the submitted manuscript CDC catalogue.
It writes:

  - manuscript_audit_cases1to7_locked.csv
  - manuscript_audit_cases1to7_locked_summary.md

Both outputs are written next to this script.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
UTIL_DIR = REPO_ROOT / "papers/2025-peak-picking/scripts/_util"
for path in (SRC_DIR, UTIL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_definitions import CASES_TRUE  # noqa: E402
from model.sample import Sample  # noqa: E402
from model.settings.calculation import (  # noqa: E402
    DiscordanceClassificationMethod,
    LeadLossCalculationSettings,
)
from model.settings.imports import LeadLossImportSettings  # noqa: E402
from model.spot import Spot  # noqa: E402
from process.cdc_pipeline import ProgressType, processSamples  # noqa: E402
from utils import csvUtils  # noqa: E402


INPUT_CASES_1TO4 = (
    REPO_ROOT
    / "papers/2025-peak-picking/data/inputs/Cases 1-7 Pb loss Inputs/cases1to4_synth_TW.csv"
)
INPUT_CASES_5TO7 = (
    REPO_ROOT
    / "papers/2025-peak-picking/data/inputs/Cases 1-7 Pb loss Inputs/synthetic_teraW_5to7_all.csv"
)
MANUSCRIPT_CATALOGUE = (
    REPO_ROOT / "papers/2025-peak-picking/data/derived/ensemble_catalogue.csv"
)
OUT_CSV = Path(__file__).with_name("manuscript_audit_cases1to7_locked.csv")
OUT_MD = Path(__file__).with_name("manuscript_audit_cases1to7_locked_summary.md")


class _HarnessSignals:
    def __init__(self, samples):
        self._samples = {s.name: s for s in samples}

    def newTask(self, *args):
        pass

    def halt(self):
        return False

    def cancelled(self, *args):
        pass

    def completed(self, *args):
        pass

    def skipped(self, sample_name, skip_reason):
        sample = self._samples.get(sample_name)
        if sample is not None:
            sample.setSkipReason(skip_reason)

    def progress(self, *args):
        kind = args[0]
        progress = args[1]
        if kind == ProgressType.CONCORDANCE and float(progress) == 1.0:
            sample_name, concordancy, discordances, *rest = args[2:]
            reverse_flags = rest[0] if rest else None
            self._samples[sample_name].updateConcordance(concordancy, discordances, reverse_flags)
            return
        if kind == ProgressType.SAMPLING:
            sample_name, run = args[2:]
            self._samples[sample_name].addMonteCarloRun(run)
            return
        if kind == ProgressType.OPTIMAL:
            sample_name, payload = args[2:]
            self._samples[sample_name].setOptimalAge(payload)
            return
        if kind == "summedKS":
            sample_name, payload = args[2:]
            sample = self._samples.get(sample_name)
            if sample is not None:
                sample.summedKS_ages_Ma = np.asarray(payload[0], float)
                sample.summedKS_goodness = np.asarray(payload[1], float)
                sample.summedKS_peaks_Ma = np.asarray(payload[2], float)


def _load_spots(csv_path: Path):
    imp = LeadLossImportSettings()
    _, rows = csvUtils.read_input(str(csv_path), imp)
    by_name = defaultdict(list)
    for row in rows:
        spot = Spot(row, imp)
        by_name[spot.sampleName].append(spot)
    return by_name


def _build_samples(by_name, *, mc_runs: int):
    samples = []
    for idx, name in enumerate(sorted(by_name)):
        sample = Sample(idx, name, by_name[name])
        calc = LeadLossCalculationSettings()
        calc.discordanceClassificationMethod = DiscordanceClassificationMethod.ERROR_ELLIPSE
        calc.discordanceEllipseSigmas = 2
        calc.minimumRimAge = 1.0e6
        calc.maximumRimAge = 2000.0e6
        calc.rimAgesSampled = 200
        calc.monteCarloRuns = int(mc_runs)
        calc.penaliseInvalidAges = True
        calc.enable_ensemble_peak_picking = True
        calc.conservative_abstain_on_monotonic = True
        calc.merge_nearby_peaks = True
        sample.startCalculation(calc)
        samples.append(sample)
    return samples


def _run_profile(by_name, *, mc_runs: int):
    samples = _build_samples(by_name, mc_runs=mc_runs)
    processSamples(_HarnessSignals(samples), samples)
    return {sample.name: sample for sample in samples}


def _run_single_sample(sample_name, spots, *, mc_runs: int):
    samples = _build_samples({sample_name: spots}, mc_runs=mc_runs)
    processSamples(_HarnessSignals(samples), samples)
    return samples[0]


def _format_peak_string(rows) -> str:
    if not rows:
        return ""
    return ";".join(f"{float(_row_age(row)):.3f}" for row in rows if np.isfinite(_row_age(row)))


def _format_ci_string(rows) -> str:
    if not rows:
        return ""
    return ";".join(
        f"{float(_row_ci_low(row)):.3f}-{float(_row_ci_high(row)):.3f}"
        for row in rows
        if np.isfinite(_row_ci_low(row)) and np.isfinite(_row_ci_high(row))
    )


def _row_age(row):
    if isinstance(row, dict):
        return float(row.get("age_ma", np.nan))
    if isinstance(row, (list, tuple)) and len(row) >= 1:
        return float(row[0])
    return float("nan")


def _row_ci_low(row):
    if isinstance(row, dict):
        return float(row.get("ci_low", np.nan))
    if isinstance(row, (list, tuple)) and len(row) >= 2:
        return float(row[1])
    return float("nan")


def _row_ci_high(row):
    if isinstance(row, dict):
        return float(row.get("ci_high", np.nan))
    if isinstance(row, (list, tuple)) and len(row) >= 3:
        return float(row[2])
    return float("nan")


def _sorted_peak_ages(rows):
    if not rows:
        return np.asarray([], float)
    vals = [_row_age(row) for row in rows if np.isfinite(_row_age(row))]
    return np.asarray(sorted(vals), float)


def _mean_abs_delta(a, b):
    a = np.asarray(sorted(a), float)
    b = np.asarray(sorted(b), float)
    if a.size == 0 or b.size == 0:
        return float("nan")
    n = min(a.size, b.size)
    return float(np.mean(np.abs(a[:n] - b[:n])))


def _max_abs_delta(a, b):
    a = np.asarray(sorted(a), float)
    b = np.asarray(sorted(b), float)
    if a.size == 0 or b.size == 0:
        return float("nan")
    n = min(a.size, b.size)
    return float(np.max(np.abs(a[:n] - b[:n])))


def _nearest_errors(rows, truths):
    peaks = _sorted_peak_ages(rows)
    out = []
    for truth in truths:
        if peaks.size == 0:
            out.append(float("nan"))
            continue
        out.append(float(np.min(np.abs(peaks - float(truth)))))
    return out


def _ci_covers_truth(rows, truths):
    out = []
    for truth in truths:
        cover = False
        for row in rows or []:
            lo = _row_ci_low(row)
            hi = _row_ci_high(row)
            if lo <= float(truth) <= hi:
                cover = True
                break
        out.append(bool(cover))
    return out


def _load_manuscript_rows():
    df = pd.read_csv(MANUSCRIPT_CATALOGUE)
    df.columns = [str(c).strip().lower() for c in df.columns]
    grouped = defaultdict(list)
    for _, row in df.iterrows():
        grouped[str(row["sample"]).strip()].append(
            dict(
                age_ma=float(row["age_ma"]),
                ci_low=float(row["ci_low"]),
                ci_high=float(row["ci_high"]),
                support=float(row["support"]),
            )
        )
    return grouped


def _sample_case(name: str) -> str:
    return str(name).strip()[0]


def _sample_tier(name: str) -> str:
    return str(name).strip()[-1]


def _note_for_row(man_rows, current_rows, current_abstain):
    notes = []
    if len(current_rows) != len(man_rows):
        notes.append("peak_count_changed_vs_manuscript")
    if current_abstain:
        notes.append("current_run_abstained")
    return ";".join(notes)


def _write_outputs(records, mc_runs: int):
    df = pd.DataFrame.from_records(records).sort_values(["case", "tier"])
    df.to_csv(OUT_CSV, index=False)

    same_peak_count = int(np.sum(df["manuscript_n_peaks"] == df["current_n_peaks"]))
    changed_peak_count = int(len(df) - same_peak_count)
    abstained_count = int(np.sum(df["abstain_reason"].astype(str) != ""))
    mean_delta = float(np.nanmean(df["mean_abs_delta_manuscript_vs_current_ma"]))
    max_delta = float(np.nanmax(df["max_abs_delta_manuscript_vs_current_ma"]))

    lines = [
        "# Manuscript Audit: Cases 1-7 Locked CDC Profile",
        "",
        "Locked settings:",
        f"- Pb-loss grid: 1 to 2000 Ma",
        f"- Grid nodes: 200",
        f"- Monte Carlo runs: {mc_runs}",
        "- Penalise invalid ages: on",
        "- Ensemble peak picking: on",
        "- Conservative abstention on",
        "- Merge nearby peaks on",
        "",
        "Headline summary:",
        f"- Samples audited: {len(df)}",
        f"- Manuscript vs current run: same peak count in {same_peak_count}/{len(df)} samples",
        f"- Manuscript vs current run: changed peak count in {changed_peak_count}/{len(df)} samples",
        f"- Current run abstained in {abstained_count}/{len(df)} samples",
        f"- Mean absolute peak-age delta, manuscript vs current run: {mean_delta:.2f} Ma",
        f"- Maximum absolute peak-age delta, manuscript vs current run: {max_delta:.2f} Ma",
        "",
        "Files:",
        f"- CSV: `{OUT_CSV}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")
    return df


def main():
    mc_runs = 100
    by_name = {}
    by_name.update(_load_spots(INPUT_CASES_1TO4))
    by_name.update(_load_spots(INPUT_CASES_5TO7))

    manuscript = _load_manuscript_rows()

    records = []
    sample_names = sorted(manuscript)
    total = len(sample_names)
    for idx, sample_name in enumerate(sample_names, start=1):
        case = _sample_case(sample_name)
        truths = CASES_TRUE[case]
        man_rows = manuscript.get(sample_name, [])
        current_sample = _run_single_sample(
            sample_name,
            by_name[sample_name],
            mc_runs=mc_runs,
        )
        current_rows = list(current_sample.peak_catalogue or [])

        current_errs = _nearest_errors(current_rows, truths)
        current_cov = _ci_covers_truth(current_rows, truths)

        record = {
            "sample": sample_name,
            "case": case,
            "tier": _sample_tier(sample_name),
            "true_ages_ma": "|".join(f"{float(t):.1f}" for t in truths),
            "manuscript_peaks_ma": _format_peak_string(man_rows),
            "current_peaks_ma": _format_peak_string(current_rows),
            "current_ci_ma": _format_ci_string(current_rows),
            "manuscript_n_peaks": len(man_rows),
            "current_n_peaks": len(current_rows),
            "abstain_reason": getattr(current_sample, "ensemble_abstain_reason", None) or "",
            "current_optimal_age_ma": float(current_sample.optimalAge) / 1e6 if np.isfinite(float(current_sample.optimalAge)) else np.nan,
            "mean_abs_delta_manuscript_vs_current_ma": _mean_abs_delta(
                [_row_age(row) for row in man_rows],
                [_row_age(row) for row in current_rows],
            ),
            "max_abs_delta_manuscript_vs_current_ma": _max_abs_delta(
                [_row_age(row) for row in man_rows],
                [_row_age(row) for row in current_rows],
            ),
            "truth1_err_current_ma": current_errs[0] if len(current_errs) > 0 else np.nan,
            "truth1_cov_current": current_cov[0] if len(current_cov) > 0 else False,
            "notes": _note_for_row(
                man_rows,
                current_rows,
                getattr(current_sample, "ensemble_abstain_reason", None),
            ),
        }
        if len(truths) > 1:
            record["truth2_err_current_ma"] = current_errs[1]
            record["truth2_cov_current"] = current_cov[1]
        else:
            record["truth2_err_current_ma"] = np.nan
            record["truth2_cov_current"] = False
        records.append(record)
        _write_outputs(records, mc_runs)
        print(
            f"[{idx}/{total}] {sample_name}: current={len(current_rows)} peaks"
            f"{' (' + record['abstain_reason'] + ')' if record['abstain_reason'] else ''}; "
            f"manuscript={len(man_rows)} peaks; "
            f"man-vs-current delta={record['mean_abs_delta_manuscript_vs_current_ma']:.2f} Ma",
            flush=True,
        )

    _write_outputs(records, mc_runs)
    print(f"Wrote {OUT_CSV}", flush=True)
    print(f"Wrote {OUT_MD}", flush=True)


if __name__ == "__main__":
    main()
