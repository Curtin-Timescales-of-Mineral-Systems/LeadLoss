import unittest
import copy
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


def _build_samples_with_profile(
    csv_path: Path,
    *,
    sample_filter=None,
    import_settings: LeadLossImportSettings,
    calc_settings: LeadLossCalculationSettings,
):
    _, rows = csvUtils.read_input(str(csv_path), import_settings)
    by_name = defaultdict(list)
    for row in rows:
        spot = Spot(row, import_settings)
        if (sample_filter is None) or (spot.sampleName in sample_filter):
            by_name[spot.sampleName].append(spot)

    out = []
    for i, (name, spots) in enumerate(sorted(by_name.items())):
        sample = Sample(i, name, spots)
        sample.startCalculation(copy.deepcopy(calc_settings))
        out.append(sample)
    return out


def _yilgarn_profile(*, mc_runs: int = 100, use_clustering: bool = True):
    imp = LeadLossImportSettings()
    imp.uPbErrorType = "Percentage"
    imp.uPbErrorSigmas = 1
    imp.pbPbErrorType = "Percentage"
    imp.pbPbErrorSigmas = 1

    calc = LeadLossCalculationSettings()
    calc.discordanceClassificationMethod = DiscordanceClassificationMethod.ERROR_ELLIPSE
    calc.discordanceEllipseSigmas = 1
    calc.minimumRimAge = 1.0e6
    calc.maximumRimAge = 1500.0e6
    calc.rimAgesSampled = 150
    calc.monteCarloRuns = int(mc_runs)
    calc.penaliseInvalidAges = True
    calc.enable_ensemble_peak_picking = True
    calc.conservative_abstain_on_monotonic = True
    calc.merge_nearby_peaks = False
    calc.use_discordant_clustering = bool(use_clustering)
    return imp, calc


def _yilgarn_csv_path() -> Path:
    return (
        Path.home()
        / "OneDrive - Curtin"
        / "Pb_loss-KIRKLC-SE10523"
        / "Pb loss modelling"
        / "yilgarn_ga"
        / "GA_Yilgarn_filtered.csv"
    )


def _assert_age_near(testcase: unittest.TestCase, observed: float, expected: float, tol: float):
    testcase.assertTrue(
        abs(float(observed) - float(expected)) <= float(tol),
        msg=f"expected age near {expected} Ma (+/- {tol}), got {observed} Ma",
    )


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

    def test_clustering_case2a_recovers_two_truth_peaks(self):
        csv_path = _repo_root() / "papers/2025-peak-picking/data/inputs/Cases 1-7 Pb loss Inputs/cases1to4_synth_TW.csv"
        samples = _build_samples(csv_path, sample_filter={"2A"}, mc_runs=100)
        samples[0].calculationSettings.use_discordant_clustering = True
        samples[0].calculationSettings.merge_nearby_peaks = False

        sample = _run_pipeline(samples)["2A"]
        summary = getattr(sample, "disc_cluster_summary", {}) or {}
        rows = sample.peak_catalogue or []

        self.assertTrue(summary.get("accepted"))
        self.assertEqual(summary.get("reason"), "accepted")
        self.assertEqual(len(rows), 2)
        _assert_age_near(self, rows[0]["age_ma"], 300.67, 40.0)
        _assert_age_near(self, rows[1]["age_ma"], 1800.74, 40.0)
        self.assertEqual(rows[0].get("mode"), "")
        self.assertEqual(rows[1].get("mode"), "")

    def test_clustering_case3a_rejects_cluster_only_multi_interior_split(self):
        csv_path = _repo_root() / "papers/2025-peak-picking/data/inputs/Cases 1-7 Pb loss Inputs/cases1to4_synth_TW.csv"
        samples = _build_samples(csv_path, sample_filter={"3A"}, mc_runs=100)
        samples[0].calculationSettings.use_discordant_clustering = True
        samples[0].calculationSettings.merge_nearby_peaks = False

        sample = _run_pipeline(samples)["3A"]
        summary = getattr(sample, "disc_cluster_summary", {}) or {}
        rows = sample.peak_catalogue or []

        self.assertFalse(summary.get("accepted"))
        self.assertEqual(summary.get("reason"), "cluster_multi_interior_without_global_support")
        self.assertEqual(len(rows), 1)
        _assert_age_near(self, rows[0]["age_ma"], 402.30, 30.0)
        self.assertEqual(getattr(sample, "ensemble_surface_flags", {}).get("view_surface_source"), "global_all")

    def test_clustering_case8a_current_rejection_regression(self):
        csv_path = _repo_root() / "papers/2025-peak-picking/data/inputs/Case 8 fan-to-zero Inputs/case8_fan_to_zero_synth_TW.csv"
        samples = _build_samples(csv_path, sample_filter={"8A"}, mc_runs=100)
        samples[0].calculationSettings.use_discordant_clustering = True
        samples[0].calculationSettings.merge_nearby_peaks = False

        sample = _run_pipeline(samples)["8A"]
        summary = getattr(sample, "disc_cluster_summary", {}) or {}

        self.assertFalse(summary.get("accepted"))
        self.assertEqual(summary.get("reason"), "rejected_by_global_surface")
        self.assertEqual(sample.peak_catalogue or [], [])
        self.assertEqual(getattr(sample, "ensemble_abstain_reason", None), "no_supported_peaks")

    def test_clustering_natural_96969025_reports_boundary_plus_interior_peak(self):
        csv_path = _yilgarn_csv_path()
        if not csv_path.exists():
            self.skipTest(f"missing natural-sample file: {csv_path}")

        imp, calc = _yilgarn_profile(mc_runs=100, use_clustering=True)
        sample = _run_pipeline(
            _build_samples_with_profile(csv_path, sample_filter={"96969025"}, import_settings=imp, calc_settings=calc)
        )["96969025"]

        summary = getattr(sample, "disc_cluster_summary", {}) or {}
        rows = sample.peak_catalogue or []

        self.assertTrue(summary.get("accepted"))
        self.assertEqual(summary.get("reason"), "accepted")
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].get("mode"), "recent_boundary")
        self.assertEqual(rows[0].get("label"), "Recent boundary mode")
        self.assertAlmostEqual(float(rows[0]["ci_high"]), 11.0604, places=2)
        _assert_age_near(self, rows[1]["age_ma"], 1022.94, 40.0)
        self.assertEqual(getattr(sample, "ensemble_surface_flags", {}).get("view_surface_source"), "clustered")

    def test_clustering_natural_97969138_partial_cluster_resolution_regression(self):
        csv_path = _yilgarn_csv_path()
        if not csv_path.exists():
            self.skipTest(f"missing natural-sample file: {csv_path}")

        imp, calc = _yilgarn_profile(mc_runs=100, use_clustering=True)
        sample = _run_pipeline(
            _build_samples_with_profile(csv_path, sample_filter={"97969138"}, import_settings=imp, calc_settings=calc)
        )["97969138"]

        summary = getattr(sample, "disc_cluster_summary", {}) or {}
        rows = sample.peak_catalogue or []
        rejected = getattr(sample, "rejected_peak_candidates", []) or []

        self.assertTrue(summary.get("accepted"))
        self.assertEqual(summary.get("reason"), "partial_cluster_resolution")
        self.assertEqual(len(rows), 1)
        _assert_age_near(self, rows[0]["age_ma"], 147.29, 30.0)
        self.assertGreaterEqual(len(rejected), 1)
        _assert_age_near(self, rejected[0]["age_ma"], 11.06, 5.0)

    def test_clustering_natural_97969138_twosigma_plot_rows_match_catalogue(self):
        csv_path = _yilgarn_csv_path()
        if not csv_path.exists():
            self.skipTest(f"missing natural-sample file: {csv_path}")

        imp = LeadLossImportSettings()
        imp.uPbErrorType = "Percentage"
        imp.uPbErrorSigmas = 1
        imp.pbPbErrorType = "Percentage"
        imp.pbPbErrorSigmas = 1

        calc = LeadLossCalculationSettings()
        calc.discordanceClassificationMethod = DiscordanceClassificationMethod.ERROR_ELLIPSE
        calc.discordanceEllipseSigmas = 2
        calc.minimumRimAge = 1.0e6
        calc.maximumRimAge = 1500.0e6
        calc.rimAgesSampled = 150
        calc.monteCarloRuns = 100
        calc.penaliseInvalidAges = True
        calc.enable_ensemble_peak_picking = True
        calc.conservative_abstain_on_monotonic = True
        calc.merge_nearby_peaks = False
        calc.use_discordant_clustering = True

        sample = _run_pipeline(
            _build_samples_with_profile(csv_path, sample_filter={"97969138"}, import_settings=imp, calc_settings=calc)
        )["97969138"]

        interior_rows = [r for r in (sample.peak_catalogue or []) if str(r.get("mode", "")) != "recent_boundary"]
        plotted = np.asarray(sample.summedKS_peaks_Ma, float)

        self.assertEqual(len(interior_rows), 1)
        self.assertEqual(plotted.size, 1)
        self.assertAlmostEqual(float(plotted[0]), float(interior_rows[0]["age_ma"]), places=6)

    def test_clustering_natural_96969042_falls_back_after_no_robust_proxy_split(self):
        csv_path = _yilgarn_csv_path()
        if not csv_path.exists():
            self.skipTest(f"missing natural-sample file: {csv_path}")

        imp, calc = _yilgarn_profile(mc_runs=100, use_clustering=True)
        sample = _run_pipeline(
            _build_samples_with_profile(csv_path, sample_filter={"96969042"}, import_settings=imp, calc_settings=calc)
        )["96969042"]

        summary = getattr(sample, "disc_cluster_summary", {}) or {}
        rows = sample.peak_catalogue or []

        self.assertFalse(summary.get("accepted"))
        self.assertEqual(summary.get("reason"), "no_robust_proxy_split")
        self.assertEqual(len(rows), 1)
        _assert_age_near(self, rows[0]["age_ma"], 911.47, 60.0)
        self.assertEqual(getattr(sample, "ensemble_surface_flags", {}).get("view_surface_source"), "global_all")

    def test_clustering_case4c_falls_back_after_ambiguous_assignments(self):
        csv_path = _repo_root() / "papers/2025-peak-picking/data/inputs/Cases 1-7 Pb loss Inputs/cases1to4_synth_TW.csv"
        samples = _build_samples(csv_path, sample_filter={"4C"}, mc_runs=100)
        samples[0].calculationSettings.use_discordant_clustering = True
        samples[0].calculationSettings.merge_nearby_peaks = False

        sample = _run_pipeline(samples)["4C"]
        summary = getattr(sample, "disc_cluster_summary", {}) or {}
        rows = sample.peak_catalogue or []

        self.assertFalse(summary.get("accepted"))
        self.assertEqual(summary.get("reason"), "too_many_ambiguous_assignments")
        self.assertEqual(len(rows), 2)
        _assert_age_near(self, rows[0]["age_ma"], 603.71, 40.0)
        _assert_age_near(self, rows[1]["age_ma"], 1733.80, 60.0)
        self.assertEqual(getattr(sample, "ensemble_surface_flags", {}).get("view_surface_source"), "global_all")

    def test_clustering_case4a_falls_back_after_ambiguous_assignments(self):
        csv_path = _repo_root() / "papers/2025-peak-picking/data/inputs/Cases 1-7 Pb loss Inputs/cases1to4_synth_TW.csv"
        samples = _build_samples(csv_path, sample_filter={"4A"}, mc_runs=100)
        samples[0].calculationSettings.use_discordant_clustering = True
        samples[0].calculationSettings.merge_nearby_peaks = False

        sample = _run_pipeline(samples)["4A"]
        summary = getattr(sample, "disc_cluster_summary", {}) or {}
        rows = sample.peak_catalogue or []

        self.assertFalse(summary.get("accepted"))
        self.assertEqual(summary.get("reason"), "too_many_ambiguous_assignments")
        self.assertEqual(len(rows), 2)
        _assert_age_near(self, rows[0]["age_ma"], 553.49, 40.0)
        _assert_age_near(self, rows[1]["age_ma"], 1825.42, 60.0)
        self.assertEqual(getattr(sample, "ensemble_surface_flags", {}).get("view_surface_source"), "global_all")

    def test_clustering_case4b_falls_back_after_ambiguous_assignments(self):
        csv_path = _repo_root() / "papers/2025-peak-picking/data/inputs/Cases 1-7 Pb loss Inputs/cases1to4_synth_TW.csv"
        samples = _build_samples(csv_path, sample_filter={"4B"}, mc_runs=100)
        samples[0].calculationSettings.use_discordant_clustering = True
        samples[0].calculationSettings.merge_nearby_peaks = False

        sample = _run_pipeline(samples)["4B"]
        summary = getattr(sample, "disc_cluster_summary", {}) or {}
        rows = sample.peak_catalogue or []

        self.assertFalse(summary.get("accepted"))
        self.assertEqual(summary.get("reason"), "too_many_ambiguous_assignments")
        self.assertEqual(len(rows), 2)
        _assert_age_near(self, rows[0]["age_ma"], 573.58, 40.0)
        _assert_age_near(self, rows[1]["age_ma"], 1798.21, 60.0)
        self.assertEqual(getattr(sample, "ensemble_surface_flags", {}).get("view_surface_source"), "global_all")


if __name__ == "__main__":
    unittest.main()
