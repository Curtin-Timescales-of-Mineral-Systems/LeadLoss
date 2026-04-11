import unittest

import numpy as np

from process.cdc_pipeline import (
    _is_effectively_monotonic,
    _recompute_winner_support,
    _snap_rows_to_curve,
    _single_crest_fallback_row,
)
from process.ensemble import build_ensemble_catalogue


class EnsembleCatalogueTests(unittest.TestCase):
    def test_monotonic_surface_abstains(self):
        ages = np.linspace(1.0, 2000.0, 200)
        # Strictly increasing runs (no interior hump).
        runs = np.vstack([
            np.linspace(0.10, 0.90, ages.size),
            np.linspace(0.11, 0.91, ages.size),
            np.linspace(0.09, 0.89, ages.size),
        ])
        rows = build_ensemble_catalogue(
            "mono",
            "A",
            ages,
            runs,
            orientation="max",
            smooth_frac=0.01,
            f_d=0.10,
            f_p=0.10,
            f_v=0.50,
            f_w=0.10,
            support_min=0.10,
            r_min=2,
            f_r=0.25,
        )
        self.assertEqual(rows, [])

    def test_effectively_monotonic_helper(self):
        y_mono = np.linspace(0.2, 0.8, 200)
        delta_mono = float(np.nanpercentile(y_mono, 95) - np.nanpercentile(y_mono, 5))
        self.assertTrue(_is_effectively_monotonic(y_mono, delta_mono))

        x = np.linspace(0.0, 1.0, 200)
        y_hump = 0.5 + 0.2 * np.exp(-0.5 * ((x - 0.5) / 0.08) ** 2)
        delta_hump = float(np.nanpercentile(y_hump, 95) - np.nanpercentile(y_hump, 5))
        self.assertFalse(_is_effectively_monotonic(y_hump, delta_hump))

    def test_winner_recompute_keeps_direct_support_as_primary(self):
        rows = [dict(age_ma=700.0, ci_low=650.0, ci_high=750.0, support=0.30, direct_support=0.30)]
        optima_ma = np.array([700.0] * 80 + [710.0] * 20, float)
        ages_ma = np.linspace(1.0, 2000.0, 200)

        out = _recompute_winner_support(rows, optima_ma, ages_ma)
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(float(out[0]["direct_support"]), 0.30, places=9)
        self.assertAlmostEqual(float(out[0]["support"]), 0.30, places=9)
        self.assertGreater(float(out[0]["winner_support"]), 0.90)

    def test_single_crest_fallback_accepts_clear_interior_hump(self):
        ages = np.linspace(1.0, 2000.0, 200)
        x = np.linspace(0.0, 1.0, ages.size)
        S = 0.35 + 0.30 * np.exp(-0.5 * ((x - 0.58) / 0.08) ** 2)
        opt = np.clip(np.random.default_rng(7).normal(1150.0, 35.0, 100), ages[0], ages[-1])

        row = _single_crest_fallback_row(ages, S, opt, min_support=0.10)
        self.assertIsNotNone(row)
        self.assertGreater(float(row["age_ma"]), 900.0)
        self.assertLess(float(row["age_ma"]), 1300.0)
        self.assertEqual(str(row.get("selection")), "fallback")

    def test_single_crest_fallback_rejects_boundary_hump(self):
        ages = np.linspace(1.0, 2000.0, 200)
        x = np.linspace(0.0, 1.0, ages.size)
        # Strong left-edge hump that should be treated as boundary-ambiguous.
        S = 0.20 + 0.50 * np.exp(-0.5 * ((x - 0.03) / 0.04) ** 2)
        opt = np.clip(np.random.default_rng(11).normal(30.0, 4.0, 100), ages[0], ages[-1])

        row = _single_crest_fallback_row(ages, S, opt, min_support=0.10)
        self.assertIsNone(row)

    def test_single_crest_fallback_accepts_broad_low_prominence_crest(self):
        ages = np.linspace(1.0, 2000.0, 200)
        x = np.linspace(0.0, 1.0, ages.size)
        # Broad interior crest with modest absolute prominence but clear run-optima pile-up.
        base = 0.72 - 0.68 * x
        crest = 0.13 * np.exp(-0.5 * ((x - 0.32) / 0.10) ** 2)
        S = base + crest
        opt = np.clip(np.random.default_rng(19).normal(610.0, 55.0, 100), ages[0], ages[-1])

        row = _single_crest_fallback_row(ages, S, opt, min_support=0.10)
        self.assertIsNotNone(row)
        self.assertGreater(float(row["age_ma"]), 450.0)
        self.assertLess(float(row["age_ma"]), 800.0)

    def test_snap_rows_to_curve_centres_flat_crest(self):
        ages = np.arange(0.0, 11.0, 1.0)
        curve = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 3.0, 2.0, 1.0, 0.0], float)
        rows = [dict(age_ma=5.2, ci_low=4.0, ci_high=6.0, support=1.0)]

        out = _snap_rows_to_curve(rows, ages, curve)
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(float(out[0]["age_ma"]), 5.0, places=6)

    def test_candidate_only_hump_is_reported_in_diagnostics(self):
        ages = np.linspace(1.0, 2500.0, 250)
        curve = (
            0.34
            + 0.05 * np.exp(-0.5 * ((ages - 220.0) / 35.0) ** 2)
            + 0.22 * np.exp(-0.5 * ((ages - 1050.0) / 170.0) ** 2)
            - 0.00003 * (ages - 1000.0)
        )
        runs = np.vstack([curve + 0.005 * np.sin(ages / 90.0 + phase) for phase in np.linspace(0.0, 1.5, 12)])
        diagnostics = []

        rows = build_ensemble_catalogue(
            "diag",
            "A",
            ages,
            runs,
            orientation="max",
            smooth_frac=0.01,
            f_d=0.10,
            f_p=0.10,
            f_v=0.50,
            f_w=0.10,
            support_min=0.10,
            r_min=3,
            f_r=0.25,
            per_run_prom_frac=0.10,
            per_run_min_dist=5,
            per_run_min_width=3,
            per_run_require_full_prom=False,
            height_frac=0.50,
            optima_ma=np.full(runs.shape[0], 1050.0),
            merge_per_hump=True,
            merge_shoulders=True,
            diagnostic_rows=diagnostics,
        )

        self.assertEqual(len(rows), 1)
        self.assertGreater(float(rows[0]["age_ma"]), 900.0)
        self.assertLess(float(rows[0]["age_ma"]), 1200.0)
        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(str(diagnostics[0]["reason"]), "coarse_surface_no_separate_mode")
        self.assertGreater(float(diagnostics[0]["age_ma"]), 150.0)
        self.assertLess(float(diagnostics[0]["age_ma"]), 300.0)

    def test_visible_subthreshold_hump_gets_diagnostic_reason(self):
        ages = np.linspace(1.0, 2000.0, 200)
        curve = (
            0.43
            + 0.04 * np.exp(-0.5 * ((ages - 180.0) / 45.0) ** 2)
            + 0.18 * np.exp(-0.5 * ((ages - 920.0) / 130.0) ** 2)
            - 0.00006 * (ages - 900.0)
        )
        runs = np.vstack([
            curve + 0.003 * np.sin(ages / 80.0 + phase)
            for phase in np.linspace(0.0, 1.2, 10)
        ])
        diagnostics = []

        rows = build_ensemble_catalogue(
            "diag2",
            "A",
            ages,
            runs,
            orientation="max",
            smooth_frac=0.01,
            f_d=0.10,
            f_p=0.18,
            f_v=0.50,
            f_w=0.10,
            support_min=0.10,
            r_min=3,
            f_r=0.25,
            per_run_prom_frac=0.10,
            per_run_min_dist=5,
            per_run_min_width=3,
            per_run_require_full_prom=False,
            optima_ma=np.full(runs.shape[0], 920.0),
            merge_per_hump=False,
            merge_shoulders=False,
            diagnostic_rows=diagnostics,
        )

        self.assertEqual(len(rows), 1)
        self.assertGreater(float(rows[0]["age_ma"]), 800.0)
        self.assertLess(float(rows[0]["age_ma"]), 1050.0)
        self.assertTrue(any(
            (str(r["reason"]) == "below_ensemble_prominence")
            and (120.0 < float(r["age_ma"]) < 260.0)
            for r in diagnostics
        ))

if __name__ == "__main__":
    unittest.main()
