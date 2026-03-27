import unittest
from collections import defaultdict
from pathlib import Path

import numpy as np

from model.sample import Sample
from model.settings.imports import LeadLossImportSettings
from model.settings.calculation import (
    DiscordanceClassificationMethod,
    LeadLossCalculationSettings,
)
from model.spot import Spot
from process.cdc_pipeline import ProgressType, processSamples
from utils import config
from utils import csvUtils


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


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _build_samples(csv_path: Path, *, sample_filter=None, mc_runs: int = 20):
    imp = LeadLossImportSettings()
    _, rows = csvUtils.read_input(str(csv_path), imp)
    by_name = defaultdict(list)
    for row in rows:
        spot = Spot(row, imp)
        if (sample_filter is None) or (spot.sampleName in sample_filter):
            by_name[spot.sampleName].append(spot)

    out = []
    for i, (name, spots) in enumerate(sorted(by_name.items())):
        sample = Sample(i, name, spots)
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
        out.append(sample)
    return out

def _run_pipeline(samples):
    signals = _HarnessSignals(samples)
    processSamples(signals, samples)
    return {s.name: s for s in samples}


class CDCPipelineRegressionTest(unittest.TestCase):
    def test_fan_to_zero_boundary_regression(self):
        csv_path = _repo_root() / "papers/2025-peak-picking/data/inputs/Case 8 fan-to-zero Inputs/case8_fan_to_zero_synth_TW.csv"
        names = {"8A", "8B", "8C"}

        samples = _run_pipeline(_build_samples(csv_path, sample_filter=names))

        for name in sorted(names):
            self.assertIn(name, samples)
            sample = samples[name]
            self.assertTrue(np.isfinite(sample.optimalAge))
            self.assertAlmostEqual(float(sample.optimalAge), 1.0e6, places=6)
            self.assertAlmostEqual(float(sample.optimalAgeLowerBound), 1.0e6, places=6)
            self.assertTrue(float(sample.optimalAgeUpperBound) >= 1.0e6)
            self.assertEqual(len(sample.peak_catalogue or []), 0)
            self.assertIn(
                getattr(sample, "ensemble_abstain_reason", None),
                {"flat_or_monotonic_surface", "boundary_dominated_surface", "no_supported_peaks"},
            )

    def test_processing_preserves_user_model_window_and_grid(self):
        csv_path = _repo_root() / "papers/2025-peak-picking/data/inputs/Case 8 fan-to-zero Inputs/case8_fan_to_zero_synth_TW.csv"
        sample = _build_samples(csv_path, sample_filter={"8A"}, mc_runs=100)[0]

        settings = sample.calculationSettings
        expected_min = float(settings.minimumRimAge)
        expected_max = float(settings.maximumRimAge)
        expected_nodes = int(settings.rimAgesSampled)

        sample = _run_pipeline([sample])["8A"]

        self.assertEqual(float(sample.calculationSettings.minimumRimAge), expected_min)
        self.assertEqual(float(sample.calculationSettings.maximumRimAge), expected_max)
        self.assertEqual(int(sample.calculationSettings.rimAgesSampled), expected_nodes)

    def test_ui_curve_matches_catalogue_case4a(self):
        csv_path = _repo_root() / "papers/2025-peak-picking/data/inputs/Cases 1-7 Pb loss Inputs/cases1to4_synth_TW.csv"

        sample = _run_pipeline(_build_samples(csv_path, sample_filter={"4A"}))["4A"]

        self.assertTrue(np.isfinite(sample.summedKS_ages_Ma).all())
        self.assertGreater(len(sample.peak_catalogue or []), 0)
        self.assertEqual(sample.ensemble_surface_flags["view_surface_source"], "global_all")

        catalogue_ages = np.asarray([float(row["age_ma"]) for row in sample.peak_catalogue], float)
        plotted_ages = np.asarray(sample.summedKS_peaks_Ma, float)
        self.assertEqual(plotted_ages.size, catalogue_ages.size)

        step = float(np.median(np.diff(np.asarray(sample.summedKS_ages_Ma, float))))
        self.assertTrue(
            np.all(np.abs(plotted_ages - catalogue_ages) <= (1.5 * step)),
            msg=f"catalogue ages {catalogue_ages} drifted from plotted ages {plotted_ages}",
        )

    def test_unimodal_case1c_retains_single_peak(self):
        csv_path = _repo_root() / "papers/2025-peak-picking/data/inputs/Cases 1-7 Pb loss Inputs/cases1to4_synth_TW.csv"

        sample = _run_pipeline(_build_samples(csv_path, sample_filter={"1C"}, mc_runs=100))["1C"]

        self.assertEqual(len(sample.peak_catalogue or []), 1)
        self.assertEqual(sample.ensemble_surface_flags["view_surface_source"], "global_all")
        self.assertIn("direct_support", sample.peak_catalogue[0])
        self.assertIn("winner_support", sample.peak_catalogue[0])

        first_run = sample.monteCarloRuns[0]
        observed_heatmap = np.asarray(first_run.heatmapColumnData, float)
        first_run.createHeatmapData(
            sample.calculationSettings.minimumRimAge,
            sample.calculationSettings.maximumRimAge,
            config.HEATMAP_RESOLUTION,
        )
        expected_heatmap = np.asarray(first_run.heatmapColumnData, float)
        self.assertTrue(np.allclose(observed_heatmap, expected_heatmap, equal_nan=True))


if __name__ == "__main__":
    unittest.main()
