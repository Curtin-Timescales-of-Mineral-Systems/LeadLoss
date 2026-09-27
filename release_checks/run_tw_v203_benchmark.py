#!/usr/bin/env python3
"""Run one frozen historical Tera-Wasserburg benchmark.

The script is intentionally compatible with both LeadLoss v2.0.3 and the
v2.1.0 release candidate.  It uses the historical TW calculation path: no
rho column, percentage 1-sigma input errors, 2-sigma ellipse classification,
50 Monte Carlo runs, and a 200-node 1--2000 Ma search grid.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_plain(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _finite_or_none(value, scale=1.0):
    if value is None:
        return None
    value = float(value) / float(scale)
    return value if np.isfinite(value) else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    sys.path.insert(0, str(args.source_root / "src"))

    from model.sample import Sample
    from model.settings.imports import LeadLossImportSettings
    from model.settings.calculation import (
        DiscordanceClassificationMethod,
        LeadLossCalculationSettings,
    )
    from model.spot import Spot
    from process.cdc_pipeline import ProgressType, processSamples
    from utils import csvUtils

    class HarnessSignals:
        def __init__(self, samples):
            self.samples = {sample.name: sample for sample in samples}

        def newTask(self, *unused):
            pass

        def halt(self):
            return False

        def cancelled(self, *unused):
            pass

        def completed(self, *unused):
            pass

        def skipped(self, sample_name, reason):
            sample = self.samples.get(sample_name)
            if sample is not None:
                sample.setSkipReason(reason)

        def progress(self, *args):
            kind, progress, *payload = args
            if kind == ProgressType.CONCORDANCE and float(progress) == 1.0:
                sample_name, concordancy, discordances, *rest = payload
                reverse_flags = rest[0] if rest else None
                self.samples[sample_name].updateConcordance(
                    concordancy, discordances, reverse_flags
                )
            elif kind == ProgressType.SAMPLING:
                sample_name, run = payload
                self.samples[sample_name].addMonteCarloRun(run)
            elif kind == ProgressType.OPTIMAL:
                sample_name, result = payload
                self.samples[sample_name].setOptimalAge(result)
            elif kind == "summedKS":
                sample_name, result = payload
                sample = self.samples.get(sample_name)
                if sample is not None:
                    sample.summedKS_ages_Ma = np.asarray(result[0], float)
                    sample.summedKS_goodness = np.asarray(result[1], float)

    imp = LeadLossImportSettings()
    imp.uPbErrorType = "Percentage"
    imp.uPbErrorSigmas = 1
    imp.pbPbErrorType = "Percentage"
    imp.pbPbErrorSigmas = 1
    imp.rhoColumn = None

    _, rows = csvUtils.read_input(str(args.input), imp)
    by_name = defaultdict(list)
    for row in rows:
        spot = Spot(row, imp)
        by_name[spot.sampleName].append(spot)

    samples = []
    for index, (name, spots) in enumerate(sorted(by_name.items())):
        try:
            sample = Sample(index, name, spots, importSettings=imp)
        except TypeError:
            sample = Sample(index, name, spots)

        calc = LeadLossCalculationSettings()
        calc.discordanceClassificationMethod = DiscordanceClassificationMethod.ERROR_ELLIPSE
        calc.discordanceEllipseSigmas = 2
        calc.minimumRimAge = 1.0e6
        calc.maximumRimAge = 2000.0e6
        calc.rimAgesSampled = 200
        calc.monteCarloRuns = 50
        calc.penaliseInvalidAges = True
        calc.enable_ensemble_peak_picking = True
        calc.conservative_abstain_on_monotonic = True
        calc.merge_nearby_peaks = False
        sample.startCalculation(calc)
        samples.append(sample)

    processSamples(HarnessSignals(samples), samples)

    report = {
        "input_file": args.input.name,
        "settings": {
            "input_space": "Tera-Wasserburg",
            "rho_column": None,
            "error_type": "Percentage",
            "error_sigmas": 1,
            "classification": "2-sigma error ellipse",
            "minimum_age_ma": 1,
            "maximum_age_ma": 2000,
            "grid_nodes": 200,
            "monte_carlo_runs": 50,
        },
        "samples": {},
    }

    for sample in sorted(samples, key=lambda item: item.name):
        peaks = []
        for peak in list(getattr(sample, "peak_catalogue", []) or []):
            peaks.append(
                {
                    key: _plain(peak.get(key))
                    for key in (
                        "age_ma",
                        "ci_low",
                        "ci_high",
                        "direct_support",
                        "winner_support",
                        "evidence_class",
                        "qualification_reason",
                    )
                    if key in peak
                }
            )

        report["samples"][sample.name] = {
            "concordant": len(sample.concordantSpots()),
            "discordant": len(sample.discordantSpots()),
            "reverse_discordant": len(sample.reverseDiscordantSpots()),
            "optimal_age_ma": _finite_or_none(sample.optimalAge, 1.0e6),
            "optimal_low_ma": _finite_or_none(sample.optimalAgeLowerBound, 1.0e6),
            "optimal_high_ma": _finite_or_none(sample.optimalAgeUpperBound, 1.0e6),
            "optimal_d": _finite_or_none(sample.optimalAgeDValue),
            "optimal_p": _finite_or_none(sample.optimalAgePValue),
            "optimal_invalid": _finite_or_none(sample.optimalAgeNumberOfInvalidPoints),
            "optimal_score": _finite_or_none(sample.optimalAgeScore),
            "goodness_age_ma": _plain(getattr(sample, "summedKS_ages_Ma", [])),
            "goodness": _plain(getattr(sample, "summedKS_goodness", [])),
            "peaks": peaks,
            "skip_reason": _plain(getattr(sample, "skip_reason", None)),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
