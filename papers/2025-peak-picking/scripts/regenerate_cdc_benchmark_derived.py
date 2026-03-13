#!/usr/bin/env python3
"""
Regenerate CDC benchmark-derived manuscript artefacts for the ensemble-only workflow.

This script reruns the synthetic Cases 1A–7C through the same CDC worker path used
by the GUI, writes manuscript-style derived outputs into
`papers/2025-peak-picking/data/derived/`, and archives the previous CDC-derived
benchmark bundle before overwriting it.
"""

from __future__ import annotations

import argparse
import copy
import csv
import os
import re
import shutil
import sys
import tarfile
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _paper_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _benchmark_inputs(include_case8: bool) -> List[Path]:
    paper_dir = _paper_dir()
    inputs = [
        paper_dir / "data" / "inputs" / "Cases 1-7 Pb loss Inputs" / "cases1to4_synth_TW.csv",
        paper_dir / "data" / "inputs" / "Cases 1-7 Pb loss Inputs" / "synthetic_teraW_5to7_all.csv",
    ]
    if include_case8:
        inputs.append(
            paper_dir / "data" / "inputs" / "Case 8 fan-to-zero Inputs" / "case8_fan_to_zero_synth_TW.csv"
        )
    return inputs


def _set_manuscript_env(derived_dir: Path) -> None:
    os.environ["CDC_PROFILE"] = "PAPER"
    os.environ["CDC_WRITE_OUTPUTS"] = "1"
    os.environ["CDC_ENABLE_RUNLOG"] = "1"
    os.environ["CDC_OUT_DIR"] = str(derived_dir)
    os.environ["CDC_KS_EXPORT_DIR"] = str(derived_dir / "ks_exports")
    os.environ.setdefault("PYTHONHASHSEED", "0")


def _timestamp() -> str:
    return time.strftime("%Y-%m-%d-%H%M%S")


def _archive_existing_outputs(derived_dir: Path) -> Path | None:
    archive_root = derived_dir / "archive"
    archive_dir = archive_root / f"pre-ensemble-only-benchmark-rerun-{_timestamp()}"
    candidates = [
        derived_dir / "ensemble_catalogue.csv",
        derived_dir / "ensemble_catalogue_np.csv",
        derived_dir / "runtime_log.csv",
        derived_dir / "ks_exports",
        derived_dir / "ks_diagnostics",
        derived_dir / "ks_diagnostics_npz.tar.gz",
    ]
    existing = [p for p in candidates if p.exists()]
    if not existing:
        return None

    archive_dir.mkdir(parents=True, exist_ok=False)
    for path in existing:
        shutil.move(str(path), str(archive_dir / path.name))
    return archive_dir


def _pack_npz_bundle(derived_dir: Path) -> Path:
    src_dir = derived_dir / "diag_ks"
    dst_dir = derived_dir / "ks_diagnostics"
    tar_path = derived_dir / "ks_diagnostics_npz.tar.gz"

    if not src_dir.exists():
        raise FileNotFoundError(f"Expected CDC diagnostics folder not found: {src_dir}")

    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    shutil.move(str(src_dir), str(dst_dir))

    if tar_path.exists():
        tar_path.unlink()
    with tarfile.open(tar_path, "w:gz") as tf:
        tf.add(dst_dir, arcname="ks_diagnostics")
    return tar_path


@dataclass
class HeadlessSignals:
    total_samples: int

    def __post_init__(self) -> None:
        self.completed_samples = 0
        self.current_task = None

    def newTask(self, description: str) -> None:
        self.current_task = description
        print(description, flush=True)

    def progress(self, kind, progress, *payload) -> None:
        sample_name = payload[0] if payload else ""
        if getattr(kind, "name", None) == "OPTIMAL" and float(progress) >= 1.0:
            self.completed_samples += 1
            print(
                f"[{self.completed_samples}/{self.total_samples}] completed {sample_name}",
                flush=True,
            )

    def completed(self) -> None:
        print("[cdc] benchmark processing complete", flush=True)

    def skipped(self, sample_name: str, skip_reason: str) -> None:
        print(f"[cdc] skipped {sample_name}: {skip_reason}", flush=True)

    def cancelled(self) -> None:
        raise RuntimeError("CDC processing cancelled")

    def halt(self) -> bool:
        return False


def _load_modules():
    repo_root = _repo_root()
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from model.sample import Sample
    from model.spot import Spot
    from model.settings.calculation import (
        DiscordanceClassificationMethod,
        LeadLossCalculationSettings,
    )
    from model.settings.imports import LeadLossImportSettings
    from process.processing import processSamples
    from utils.csvUtils import read_input

    return {
        "Sample": Sample,
        "Spot": Spot,
        "DiscordanceClassificationMethod": DiscordanceClassificationMethod,
        "LeadLossCalculationSettings": LeadLossCalculationSettings,
        "LeadLossImportSettings": LeadLossImportSettings,
        "processSamples": processSamples,
        "read_input": read_input,
    }


def _build_calculation_settings(mods, mc_runs: int):
    settings = mods["LeadLossCalculationSettings"]()
    settings.discordanceClassificationMethod = mods["DiscordanceClassificationMethod"].ERROR_ELLIPSE
    settings.discordanceEllipseSigmas = 2
    settings.minimumRimAge = 1.0e6
    settings.maximumRimAge = 2000.0e6
    settings.rimAgesSampled = 200
    settings.monteCarloRuns = mc_runs
    settings.penaliseInvalidAges = True
    settings.enable_ensemble_peak_picking = True
    settings.conservative_abstain_on_monotonic = True
    settings.merge_nearby_peaks = True
    return settings


def _load_samples(input_paths: Iterable[Path], mods, mc_runs: int, sample_regex: str | None):
    import_settings = mods["LeadLossImportSettings"]()
    base_calc_settings = _build_calculation_settings(mods, mc_runs)
    spots_by_sample = defaultdict(list)
    sample_rx = re.compile(sample_regex) if sample_regex else None

    for input_path in input_paths:
        headers, rows = mods["read_input"](str(input_path), import_settings)
        _ = headers
        for row in rows:
            spot = mods["Spot"](row, import_settings)
            if sample_rx and not sample_rx.search(spot.sampleName):
                continue
            spots_by_sample[spot.sampleName].append(spot)

    samples = []
    for idx, sample_name in enumerate(sorted(spots_by_sample.keys())):
        sample = mods["Sample"](idx, sample_name, spots_by_sample[sample_name])
        sample.signals = None
        sample.startCalculation(copy.deepcopy(base_calc_settings))
        samples.append(sample)
    return samples


def _write_headless_summary(samples, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    optimal_path = out_dir / "benchmark_optimal_age_table.csv"
    with optimal_path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "sample",
                "n_concordant",
                "n_discordant",
                "ci_low_ma",
                "age_ma",
                "ci_high_ma",
                "d_value",
                "p_value",
                "score",
                "n_peaks",
                "skip_reason",
            ],
        )
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "sample": sample.name,
                    "n_concordant": len(sample.concordantSpots()),
                    "n_discordant": len(sample.discordantSpots()),
                    "ci_low_ma": (sample.optimalAgeLowerBound / 1e6) if sample.optimalAgeLowerBound else "",
                    "age_ma": (sample.optimalAge / 1e6) if sample.optimalAge else "",
                    "ci_high_ma": (sample.optimalAgeUpperBound / 1e6) if sample.optimalAgeUpperBound else "",
                    "d_value": sample.optimalAgeDValue,
                    "p_value": sample.optimalAgePValue,
                    "score": sample.optimalAgeScore,
                    "n_peaks": len(sample.peak_catalogue or []),
                    "skip_reason": sample.skip_reason or "",
                }
            )

    detailed_path = out_dir / "benchmark_ensemble_catalogue_detailed.csv"
    with detailed_path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "sample",
                "peak_no",
                "ci_low",
                "age_ma",
                "ci_high",
                "support",
                "direct_support",
                "winner_support",
                "selection",
            ],
        )
        writer.writeheader()
        for sample in samples:
            for peak in sample.peak_catalogue or []:
                writer.writerow(
                    {
                        "sample": sample.name,
                        "peak_no": peak.get("peak_no", ""),
                        "ci_low": peak.get("ci_low", ""),
                        "age_ma": peak.get("age_ma", ""),
                        "ci_high": peak.get("ci_high", ""),
                        "support": peak.get("support", ""),
                        "direct_support": peak.get("direct_support", ""),
                        "winner_support": peak.get("winner_support", ""),
                        "selection": peak.get("selection", ""),
                    }
                )


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate benchmark CDC manuscript derived outputs.")
    parser.add_argument("--mc-runs", type=int, default=200, help="Monte Carlo runs per sample. Default: 200.")
    parser.add_argument("--include-case8", action="store_true", help="Also process the fan-to-zero Case 8 dataset.")
    parser.add_argument("--no-archive", action="store_true", help="Overwrite current derived outputs without archiving.")
    parser.add_argument("--sample-regex", type=str, default=None, help="Optional regex to restrict processed sample IDs.")
    parser.add_argument(
        "--derived-dir",
        type=Path,
        default=None,
        help="Override the manuscript derived-data directory (default: <paper>/data/derived).",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Override the validation-report output directory (default: <repo>/reports/publication_validation).",
    )
    args = parser.parse_args()

    repo_root = _repo_root()
    paper_dir = _paper_dir()
    derived_dir = args.derived_dir.expanduser().resolve() if args.derived_dir else (paper_dir / "data" / "derived")
    report_dir = args.report_dir.expanduser().resolve() if args.report_dir else (repo_root / "reports" / "publication_validation")
    input_paths = _benchmark_inputs(include_case8=args.include_case8)

    for input_path in input_paths:
        if not input_path.exists():
            raise FileNotFoundError(f"Missing benchmark input: {input_path}")

    archive_dir = None if args.no_archive else _archive_existing_outputs(derived_dir)
    if archive_dir is not None:
        print(f"[archive] moved previous CDC benchmark outputs to {archive_dir}", flush=True)

    _set_manuscript_env(derived_dir)
    mods = _load_modules()
    samples = _load_samples(input_paths, mods, mc_runs=args.mc_runs, sample_regex=args.sample_regex)

    print(
        f"[cdc] processing {len(samples)} benchmark samples with R={args.mc_runs}, "
        "1–2000 Ma, 200 grid nodes, 2σ error ellipse, penalisation on, ensemble on",
        flush=True,
    )
    started = time.time()
    mods["processSamples"](HeadlessSignals(total_samples=len(samples)), samples)
    bundle = _pack_npz_bundle(derived_dir)
    _write_headless_summary(samples, report_dir)

    elapsed = time.time() - started
    print(f"[cdc] wrote manuscript benchmark bundle to {derived_dir}", flush=True)
    print(f"[cdc] wrote benchmark summary CSVs to {report_dir}", flush=True)
    print(f"[cdc] packed NPZ diagnostics to {bundle}", flush=True)
    print(f"[cdc] elapsed {elapsed / 60.0:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
