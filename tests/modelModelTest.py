import unittest

import numpy as np

from model.model import LeadLossModel
from model.sample import Sample


class ModelNearestSampledAgeShapeTest(unittest.TestCase):
    def test_empty_statistics_returns_stable_four_tuple(self):
        model = LeadLossModel(signals=object())

        result = model.getNearestSampledAge(requestedAge=500.0)
        self.assertEqual(len(result), 4)
        self.assertEqual(result, (None, None, None, []))

    def test_emit_summedks_preserves_plot_peaks_and_ci(self):
        model = LeadLossModel(signals=object())
        sample = Sample(0, "4A", [])
        model.samples = [sample]
        model.samplesByName = {"4A": sample}

        payload = (
            [1.0, 2.0, 3.0],
            [0.1, 0.2, 0.3],
            [2.5, 7.5],
            [(2.0, 3.0), (7.0, 8.0)],
            [1.0, 0.8],
        )
        model.emitSummedKS("4A", payload)

        self.assertTrue(np.allclose(sample.summedKS_ages_Ma, [1.0, 2.0, 3.0]))
        self.assertTrue(np.allclose(sample.summedKS_goodness, [0.1, 0.2, 0.3]))
        self.assertTrue(np.allclose(sample.summedKS_peaks_Ma, [2.5, 7.5]))
        self.assertTrue(np.allclose(sample.summedKS_ci_low_Ma, [2.0, 7.0]))
        self.assertTrue(np.allclose(sample.summedKS_ci_high_Ma, [3.0, 8.0]))

    def test_sample_optimal_payload_preserves_detailed_peak_support_fields(self):
        sample = Sample(0, "2A", [])
        payload = (
            382.7,
            372.6,
            392.8,
            0.5,
            1.0e-6,
            0.0,
            0.5,
            "302.2; 1801.8",
            [
                dict(age_ma=302.21, ci_low=277.10, ci_high=327.33, support=0.72, direct_support=0.72, winner_support=0.51),
                dict(age_ma=1801.80, ci_low=1776.70, ci_high=1826.90, support=0.68, direct_support=0.68, winner_support=0.49),
            ],
        )

        sample.setOptimalAge(payload)

        self.assertEqual(len(sample.peak_catalogue), 2)
        self.assertAlmostEqual(float(sample.peak_catalogue[0]["direct_support"]), 0.72)
        self.assertAlmostEqual(float(sample.peak_catalogue[0]["winner_support"]), 0.51)


if __name__ == "__main__":
    unittest.main()
