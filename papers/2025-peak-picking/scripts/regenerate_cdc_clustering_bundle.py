#!/usr/bin/env python3
"""
Regenerate a fresh clustering-enabled CDC benchmark bundle.

This script reruns synthetic Cases 1A-7C plus the Case 8 fan-to-zero dataset
through the same CDC worker path used by the GUI, writes manuscript-style
derived outputs into a fresh settings-named folder, and exports clustering
diagnostics needed for cluster-assignment figures and audit tables.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import re
import subprocess
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


def _timestamp() -> str:
    return time.strftime("%Y-%m-%d-%H%M%S")


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


def _set_cdc_env(derived_dir: Path) -> None:
    os.environ["CDC_PROFILE"] = "PAPER"
    os.environ["CDC_WRITE_OUTPUTS"] = "1"
    os.environ["CDC_ENABLE_RUNLOG"] = "1"
    os.environ["CDC_OUT_DIR"] = str(derived_dir)
    os.environ["CDC_KS_EXPORT_DIR"] = str(derived_dir / "ks_exports")
    os.environ.setdefault("PYTHONHASHSEED", "0")


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

    from model.model import LeadLossModel
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
        "LeadLossModel": LeadLossModel,
        "Sample": Sample,
        "Spot": Spot,
        "DiscordanceClassificationMethod": DiscordanceClassificationMethod,
        "LeadLossCalculationSettings": LeadLossCalculationSettings,
        "LeadLossImportSettings": LeadLossImportSettings,
        "processSamples": processSamples,
        "read_input": read_input,
    }


def _build_calculation_settings(mods, args):
    settings = mods["LeadLossCalculationSettings"]()
    settings.discordanceClassificationMethod = mods["DiscordanceClassificationMethod"].ERROR_ELLIPSE
    settings.discordanceEllipseSigmas = int(args.ellipse_sigmas)
    settings.minimumRimAge = float(args.min_age_ma) * 1.0e6
    settings.maximumRimAge = float(args.max_age_ma) * 1.0e6
    settings.rimAgesSampled = int(args.grid_nodes)
    settings.monteCarloRuns = int(args.mc_runs)
    settings.penaliseInvalidAges = True
    settings.use_discordant_clustering = True
    settings.enable_ensemble_peak_picking = True
    settings.conservative_abstain_on_monotonic = True
    settings.merge_nearby_peaks = False
    return settings


def _load_samples(input_paths: Iterable[Path], mods, args, sample_regex: str | None):
    import_settings = mods["LeadLossImportSettings"]()
    base_calc_settings = _build_calculation_settings(mods, args)
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
                "ensemble_status",
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
                    "ensemble_status": getattr(sample, "ensemble_abstain_reason", "") or ("resolved" if sample.peak_catalogue else ""),
                }
            )

    detailed_path = out_dir / "benchmark_ensemble_catalogue_detailed.csv"
    with detailed_path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "sample",
                "peak_no",
                "cluster_id",
                "mode",
                "ci_low",
                "age_ma",
                "ci_high",
                "support",
                "direct_support",
                "winner_support",
                "selection",
                "label",
            ],
        )
        writer.writeheader()
        for sample in samples:
            for peak in sample.peak_catalogue or []:
                writer.writerow(
                    {
                        "sample": sample.name,
                        "peak_no": peak.get("peak_no", ""),
                        "cluster_id": peak.get("cluster_id", ""),
                        "mode": peak.get("mode", ""),
                        "ci_low": peak.get("ci_low", ""),
                        "age_ma": peak.get("age_ma", ""),
                        "ci_high": peak.get("ci_high", ""),
                        "support": peak.get("support", ""),
                        "direct_support": peak.get("direct_support", ""),
                        "winner_support": peak.get("winner_support", ""),
                        "selection": peak.get("selection", ""),
                        "label": peak.get("label", ""),
                    }
                )


def _pack_npz_bundle(derived_dir: Path) -> Path:
    src_dir = derived_dir / "diag_ks"
    dst_dir = derived_dir / "ks_diagnostics"
    tar_path = derived_dir / "ks_diagnostics_npz.tar.gz"

    if not src_dir.exists():
        raise FileNotFoundError(f"Expected CDC diagnostics folder not found: {src_dir}")

    if dst_dir.exists():
        for path in dst_dir.iterdir():
            if path.is_file():
                path.unlink()
        dst_dir.rmdir()
    src_dir.rename(dst_dir)

    if tar_path.exists():
        tar_path.unlink()
    with tarfile.open(tar_path, "w:gz") as tf:
        tf.add(dst_dir, arcname="ks_diagnostics")
    return tar_path


def _git_value(repo_root: Path, *args: str) -> str:
    try:
        out = subprocess.check_output(["git", *args], cwd=str(repo_root), text=True)
        return out.strip()
    except Exception:
        return ""


def _write_manifest(derived_dir: Path, report_dir: Path, args, input_paths: Iterable[Path], sample_count: int) -> None:
    repo_root = _repo_root()
    manifest = {
        "generated_at": _timestamp(),
        "repo_root": str(repo_root),
        "paper_dir": str(_paper_dir()),
        "derived_dir": str(derived_dir),
        "report_dir": str(report_dir),
        "git_branch": _git_value(repo_root, "branch", "--show-current"),
        "git_commit": _git_value(repo_root, "rev-parse", "HEAD"),
        "settings": {
            "discordance_method": "error_ellipse",
            "ellipse_sigmas": int(args.ellipse_sigmas),
            "minimum_rim_age_ma": float(args.min_age_ma),
            "maximum_rim_age_ma": float(args.max_age_ma),
            "grid_nodes": int(args.grid_nodes),
            "monte_carlo_runs": int(args.mc_runs),
            "use_discordant_clustering": True,
            "enable_ensemble_peak_picking": True,
            "conservative_abstain_on_monotonic": True,
            "merge_nearby_peaks": False,
            "penalise_invalid_ages": True,
        },
        "sample_regex": args.sample_regex,
        "include_case8": bool(args.include_case8),
        "input_files": [str(p) for p in input_paths],
        "sample_count": int(sample_count),
        "outputs": {
            "root_csvs": [
                "ensemble_catalogue.csv",
                "ensemble_catalogue_np.csv",
                "runtime_log.csv",
            ],
            "directories": [
                "ks_exports",
                "ks_diagnostics",
                "clustering_diagnostics",
            ],
            "archives": [
                "ks_diagnostics_npz.tar.gz",
            ],
        },
    }

    (derived_dir / "README.md").write_text(
        "\n".join(
            [
                "# Ensemble v2 anchor-clustered benchmark bundle",
                "",
                "This folder contains a fresh ensemble-v2 anchor-clustered rerun bundle.",
                "",
                "Locked settings:",
                f"- Error ellipse: {int(args.ellipse_sigmas)} sigma",
                f"- Modelling window: {float(args.min_age_ma):.0f}-{float(args.max_age_ma):.0f} Ma",
                f"- Grid nodes: {int(args.grid_nodes)}",
                f"- Monte Carlo runs: {int(args.mc_runs)}",
                "- Discordant clustering: on",
                "- Ensemble peak picking: on",
                "- Conservative abstention: on",
                "- Merge nearby peaks: off",
                "",
                "Key outputs:",
                "- `ensemble_catalogue.csv`",
                "- `ensemble_catalogue_np.csv`",
                "- `runtime_log.csv`",
                "- `ks_exports/`",
                "- `ks_diagnostics/`",
                "- `ks_diagnostics_npz.tar.gz`",
                "- `clustering_diagnostics/`",
                "",
                "Clustering diagnostics:",
                "- combined tables: `clustering_diagnostics/all_samples_*.csv`",
                "- per-sample tables: `clustering_diagnostics/by_sample/<sample>/<sample>_*.csv`",
                "",
                "To regenerate benchmark CDC figures/tables from this bundle into a separate output folder:",
                f"- `python papers/2025-peak-picking/scripts/run_clustering_bundle.py --derived-root {derived_dir} --clean`",
                "- Use the paper-generation Python environment for that command, i.e. one with `papers/2025-peak-picking/requirements.txt` installed.",
                "",
                "Report summaries:",
                f"- `{report_dir / 'benchmark_optimal_age_table.csv'}`",
                f"- `{report_dir / 'benchmark_ensemble_catalogue_detailed.csv'}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (derived_dir / "settings_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _export_clustering_diagnostics(mods, samples, derived_dir: Path) -> None:
    diag_root = derived_dir / "clustering_diagnostics"
    by_sample_root = diag_root / "by_sample"
    by_sample_root.mkdir(parents=True, exist_ok=True)

    model = mods["LeadLossModel"](signals=None)
    model.samples = list(samples)
    model.exportClusteringDiagnostics(diag_root / "all_samples", samples=samples)

    for sample in samples:
        sample_dir = by_sample_root / sample.name
        sample_dir.mkdir(parents=True, exist_ok=True)
        model.exportClusteringDiagnostics(sample_dir / sample.name, samples=[sample])


def _default_bundle_name(args) -> str:
    return (
        f"ensemble_v2_anchor_clustered_sigma{int(args.ellipse_sigmas)}_"
        f"{int(args.min_age_ma)}to{int(args.max_age_ma)}_"
        f"nodes{int(args.grid_nodes)}_mc{int(args.mc_runs)}"
    )


def _parse_args() -> argparse.Namespace:
    paper_dir = _paper_dir()
    repo_root = _repo_root()

    ap = argparse.ArgumentParser(description="Regenerate a fresh clustering CDC benchmark bundle.")
    ap.add_argument("--mc-runs", type=int, default=100, help="Monte Carlo runs per sample. Default: 100.")
    ap.add_argument("--ellipse-sigmas", type=int, default=1, help="Error ellipse sigma cutoff. Default: 1.")
    ap.add_argument("--min-age-ma", type=float, default=1.0, help="Minimum model age in Ma. Default: 1.")
    ap.add_argument("--max-age-ma", type=float, default=2000.0, help="Maximum model age in Ma. Default: 2000.")
    ap.add_argument("--grid-nodes", type=int, default=100, help="Number of sampled ages in the grid. Default: 100.")
    ap.add_argument("--include-case8", action="store_true", default=True, help="Include the fan-to-zero Case 8 dataset.")
    ap.add_argument("--sample-regex", type=str, default=None, help="Optional regex to restrict processed sample IDs.")
    ap.add_argument(
        "--derived-dir",
        type=Path,
        default=None,
        help="Override the derived-data directory (default: fresh settings-named folder under paper data/derived).",
    )
    ap.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Override the report output directory (default: matching settings-named folder under reports/publication_validation).",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing derived/report folder.",
    )
    args = ap.parse_args()

    bundle_name = _default_bundle_name(args)
    if args.derived_dir is None:
        args.derived_dir = paper_dir / "data" / "derived" / bundle_name
    else:
        args.derived_dir = args.derived_dir.expanduser().resolve()

    if args.report_dir is None:
        args.report_dir = repo_root / "reports" / "publication_validation" / bundle_name
    else:
        args.report_dir = args.report_dir.expanduser().resolve()

    return args


def main() -> int:
    args = _parse_args()
    derived_dir = args.derived_dir
    report_dir = args.report_dir
    input_paths = _benchmark_inputs(include_case8=args.include_case8)

    for input_path in input_paths:
        if not input_path.exists():
            raise FileNotFoundError(f"Missing benchmark input: {input_path}")

    if not args.overwrite:
        for path in (derived_dir, report_dir):
            if path.exists():
                raise FileExistsError(
                    f"Refusing to overwrite existing output folder without --overwrite: {path}"
                )

    derived_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    _set_cdc_env(derived_dir)
    mods = _load_modules()
    samples = _load_samples(input_paths, mods, args=args, sample_regex=args.sample_regex)
    _write_manifest(derived_dir, report_dir, args, input_paths, sample_count=len(samples))

    print(
        f"[cdc] processing {len(samples)} samples with R={args.mc_runs}, "
        f"{int(args.min_age_ma)}-{int(args.max_age_ma)} Ma, {args.grid_nodes} grid nodes, "
        f"{args.ellipse_sigmas}σ error ellipse, clustering on, ensemble on",
        flush=True,
    )
    started = time.time()
    mods["processSamples"](HeadlessSignals(total_samples=len(samples)), samples)
    bundle = _pack_npz_bundle(derived_dir)
    _write_headless_summary(samples, report_dir)
    _export_clustering_diagnostics(mods, samples, derived_dir)

    elapsed = time.time() - started
    print(f"[cdc] wrote clustering bundle to {derived_dir}", flush=True)
    print(f"[cdc] wrote benchmark summary CSVs to {report_dir}", flush=True)
    print(f"[cdc] packed NPZ diagnostics to {bundle}", flush=True)
    print(f"[cdc] elapsed {elapsed / 60.0:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
