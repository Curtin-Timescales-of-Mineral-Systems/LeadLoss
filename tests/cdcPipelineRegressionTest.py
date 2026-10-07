import unittest
from collections import defaultdict
from pathlib import Path

import numpy as np

from model.sample import Sample
from model.settings.imports import LeadLossImportSettings
from model.settings.ratio import ConcordiaRatioSpace, ConcordiaSpaceSelection
from model.settings.calculation import (
    DiscordanceClassificationMethod,
    LeadLossCalculationSettings,
)
from model.spot import Spot
from process import calculations
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


def _fixture(name: str) -> Path:
    return Path(__file__).resolve().parent / "fixtures" / name


def _tw_rows_as_wetherill(rows):
    converted = []
    for row in rows:
        tw_x = float(row[1])
        tw_sx = float(row[2]) / 2.0
        tw_y = float(row[3])
        tw_sy = float(row[4]) / 2.0
        w_x, w_y = calculations.tw_to_wetherill(tw_x, tw_y)
        w_sx, w_sy, w_rho = calculations.convert_ratio_covariance(
            tw_x,
            tw_y,
            tw_sx,
            tw_sy,
            0.0,
            ConcordiaRatioSpace.TERA_WASSERBURG,
            ConcordiaRatioSpace.WETHERILL,
        )
        converted.append([
            row[0],
            f"{w_x:.16g}",
            f"{2.0 * w_sx:.16g}",
            f"{w_y:.16g}",
            f"{2.0 * w_sy:.16g}",
            f"{w_rho:.16g}",
        ])
    return converted


def _build_samples(
    csv_path: Path,
    *,
    sample_filter=None,
    mc_runs: int = 20,
    rim_samples: int = 200,
    input_space=ConcordiaRatioSpace.TERA_WASSERBURG,
    projection_space=ConcordiaSpaceSelection.SAME_AS_INPUT,
):
    read_settings = LeadLossImportSettings()
    _, rows = csvUtils.read_input(str(csv_path), read_settings)

    imp = LeadLossImportSettings()
    imp.inputRatioSpace = input_space
    imp.displayRatioSpace = ConcordiaSpaceSelection.SAME_AS_INPUT
    if input_space == ConcordiaRatioSpace.WETHERILL:
        imp.rhoColumn = 5
        rows = _tw_rows_as_wetherill(rows)
    by_name = defaultdict(list)
    for row in rows:
        spot = Spot(row, imp)
        if (sample_filter is None) or (spot.sampleName in sample_filter):
            by_name[spot.sampleName].append(spot)

    out = []
    for i, (name, spots) in enumerate(sorted(by_name.items())):
        sample = Sample(i, name, spots, importSettings=imp)
        calc = LeadLossCalculationSettings()
        calc.discordanceClassificationMethod = DiscordanceClassificationMethod.ERROR_ELLIPSE
        calc.discordanceEllipseSigmas = 2
        calc.minimumRimAge = 1.0e6
        calc.maximumRimAge = 2000.0e6
        calc.rimAgesSampled = int(rim_samples)
        calc.monteCarloRuns = int(mc_runs)
        calc.penaliseInvalidAges = True
        calc.concordiaProjectionGeometry = projection_space
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
        csv_path = _fixture("case8_fan_to_zero_synth_TW.csv")
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
        csv_path = _fixture("case8_fan_to_zero_synth_TW.csv")
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
        csv_path = _fixture("cases1to4_synth_TW.csv")

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

    def test_mixed_case_catalogue_uses_vote_median_and_stability_bounds(self):
        csv_path = _fixture("cases1to4_synth_TW.csv")

        samples = _run_pipeline(_build_samples(csv_path, sample_filter={"2A", "4A"}, mc_runs=200))

        sample_2a = samples["2A"]
        peaks_2a = sorted(sample_2a.peak_catalogue or [], key=lambda row: float(row["age_ma"]))
        self.assertEqual(len(peaks_2a), 2)
        self.assertEqual(str(peaks_2a[0].get("age_mode")), "vote_median")
        self.assertEqual(str(peaks_2a[1].get("age_mode")), "vote_median")
        self.assertGreater(float(sample_2a.optimalAge) / 1e6, 370.0)
        self.assertGreater(float(peaks_2a[0]["age_ma"]), 360.0)
        self.assertLess(float(peaks_2a[0]["age_ma"]), 410.0)
        self.assertGreater(float(peaks_2a[1]["age_ma"]), 1780.0)
        self.assertLess(float(peaks_2a[1]["age_ma"]), 1805.0)
        self.assertEqual(str(peaks_2a[0].get("ci_method")), "stability_bounds")
        self.assertLess(float(peaks_2a[0]["ci_low"]), float(peaks_2a[0]["ci_high"]))
        self.assertLess(float(peaks_2a[0]["support_low"]), float(peaks_2a[0]["support_high"]))

        sample_4a = samples["4A"]
        peaks_4a = sorted(sample_4a.peak_catalogue or [], key=lambda row: float(row["age_ma"]))
        self.assertEqual(len(peaks_4a), 2)
        self.assertEqual(str(peaks_4a[0].get("age_mode")), "vote_median")
        self.assertEqual(str(peaks_4a[1].get("age_mode")), "vote_median")
        self.assertGreater(float(sample_4a.optimalAge) / 1e6, 540.0)
        self.assertGreater(float(peaks_4a[0]["age_ma"]), 530.0)
        self.assertLess(float(peaks_4a[0]["age_ma"]), 580.0)
        self.assertGreater(float(peaks_4a[1]["age_ma"]), 1810.0)
        self.assertLess(float(peaks_4a[1]["age_ma"]), 1840.0)
        self.assertEqual(str(peaks_4a[0].get("ci_method")), "stability_bounds")
        self.assertLess(float(peaks_4a[0]["ci_low"]), float(peaks_4a[0]["ci_high"]))
        self.assertLess(float(peaks_4a[0]["support_low"]), float(peaks_4a[0]["support_high"]))

    def test_unimodal_case1c_retains_single_peak(self):
        csv_path = _fixture("cases1to4_synth_TW.csv")

        sample = _run_pipeline(_build_samples(csv_path, sample_filter={"1C"}, mc_runs=100))["1C"]

        self.assertEqual(len(sample.peak_catalogue or []), 1)
        self.assertEqual(sample.ensemble_surface_flags["view_surface_source"], "global_all")
        self.assertIn("direct_support", sample.peak_catalogue[0])
        self.assertIn("winner_support", sample.peak_catalogue[0])
        self.assertEqual(str(sample.peak_catalogue[0].get("age_mode")), "vote_median")
        self.assertEqual(str(sample.peak_catalogue[0].get("evidence_class")), "formal")

        first_run = sample.monteCarloRuns[0]
        observed_heatmap = np.asarray(first_run.heatmapColumnData, float)
        first_run.createHeatmapData(
            sample.calculationSettings.minimumRimAge,
            sample.calculationSettings.maximumRimAge,
            config.HEATMAP_RESOLUTION,
        )
        expected_heatmap = np.asarray(first_run.heatmapColumnData, float)
        self.assertTrue(np.allclose(observed_heatmap, expected_heatmap, equal_nan=True))

    def test_input_and_projection_space_matrix_completes(self):
        csv_path = _fixture("cases1to4_synth_TW.csv")
        cases = [
            (
                ConcordiaRatioSpace.TERA_WASSERBURG,
                ConcordiaSpaceSelection.SAME_AS_INPUT,
                ConcordiaRatioSpace.TERA_WASSERBURG,
            ),
            (
                ConcordiaRatioSpace.TERA_WASSERBURG,
                ConcordiaSpaceSelection.WETHERILL,
                ConcordiaRatioSpace.WETHERILL,
            ),
            (
                ConcordiaRatioSpace.WETHERILL,
                ConcordiaSpaceSelection.SAME_AS_INPUT,
                ConcordiaRatioSpace.WETHERILL,
            ),
            (
                ConcordiaRatioSpace.WETHERILL,
                ConcordiaSpaceSelection.TERA_WASSERBURG,
                ConcordiaRatioSpace.TERA_WASSERBURG,
            ),
        ]

        results = {}
        for input_space, projection_space, expected_model_space in cases:
            with self.subTest(input_space=input_space.value, projection_space=projection_space.value):
                sample = _run_pipeline(
                    _build_samples(
                        csv_path,
                        sample_filter={"1C"},
                        mc_runs=4,
                        rim_samples=60,
                        input_space=input_space,
                        projection_space=projection_space,
                    )
                )["1C"]

                self.assertTrue(np.isfinite(sample.optimalAge))
                self.assertEqual(sample.getModelRatioSpace(), expected_model_space)
                self.assertEqual(sample.ensemble_surface_flags["model_space"], expected_model_space.value)
                self.assertGreater(len(sample.monteCarloRuns), 0)
                self.assertEqual(sample.monteCarloRuns[0].modelRatioSpace, expected_model_space)
                self.assertTrue(np.isfinite(sample.monteCarloRuns[0].optimal_uPb))
                results[(input_space.value, expected_model_space.value)] = sample

        # With the same native TW input and random draws, changing only the
        # projection geometry should not produce a grossly different result.
        tw_result = results[(
            ConcordiaRatioSpace.TERA_WASSERBURG.value,
            ConcordiaRatioSpace.TERA_WASSERBURG.value,
        )]
        wetherill_result = results[(
            ConcordiaRatioSpace.TERA_WASSERBURG.value,
            ConcordiaRatioSpace.WETHERILL.value,
        )]
        grid_step = float(np.median(np.diff(np.asarray(tw_result.summedKS_ages_Ma, float))))
        self.assertLessEqual(
            abs(float(tw_result.optimalAge - wetherill_result.optimalAge)) / 1e6,
            2.0 * grid_step,
        )


if __name__ == "__main__":
    unittest.main()
