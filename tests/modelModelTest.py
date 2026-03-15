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

    def test_clustering_diagnostics_tables_include_split_and_cluster_rows(self):
        model = LeadLossModel(signals=object())
        sample = Sample(0, "96969025", [])
        sample.calculationSettings = type("Settings", (), {"use_discordant_clustering": True})()
        sample.disc_cluster_summary = {
            "accepted": True,
            "split_accepted": True,
            "reporting_accepted": True,
            "reason": "accepted",
            "n_anchors": 1,
            "anchor_means_ma": [2750.0],
            "anchors": [dict(anchor_id=0, age_ma=2750.0, n_concordant=12)],
            "n_discordant": 35,
            "n_assigned": 35,
            "n_ambiguous": 0,
            "assigned_fraction": 1.0,
            "n_valid_proxies": 35,
            "clusters": [dict(k=0, n=12, median_ma=169.0), dict(k=1, n=23, median_ma=1199.0)],
            "assignments": [],
        }
        sample._cdc_cluster_split_accepted = True
        sample._cdc_cluster_reporting_accepted = True
        sample.peak_catalogue = [
            dict(
                peak_no=1,
                cluster_id=0,
                mode="recent_boundary",
                age_ma=1.0,
                ci_low=1.0,
                ci_high=11.05,
                direct_support=1.0,
                winner_support=1.0,
                support=1.0,
            ),
            dict(
                peak_no=2,
                cluster_id=1,
                mode="",
                age_ma=1025.27,
                ci_low=944.91,
                ci_high=1141.04,
                direct_support=1.0,
                winner_support=1.0,
                support=1.0,
            ),
        ]
        sample.rejected_peak_candidates = [dict(cluster_id=1, age_ma=169.0, reason="no_reportable_cluster_peak")]
        model.samples = [sample]

        tables = model.buildClusteringDiagnosticsTables()
        self.assertIn("summary", tables)
        self.assertIn("clusters", tables)
        self.assertIn("peaks", tables)

        summary_headers, summary_rows = tables["summary"]
        self.assertIn("split_accepted", summary_headers)
        self.assertEqual(len(summary_rows), 1)
        self.assertEqual(summary_rows[0][0], "96969025")
        self.assertTrue(summary_rows[0][2])
        self.assertTrue(summary_rows[0][3])

        _cluster_headers, cluster_rows = tables["clusters"]
        self.assertEqual(len(cluster_rows), 2)
        self.assertEqual(cluster_rows[0][0], "96969025")

        _peak_headers, peak_rows = tables["peaks"]
        self.assertEqual(len(peak_rows), 2)
        self.assertEqual(peak_rows[0][2], 0)


if __name__ == "__main__":
    unittest.main()
