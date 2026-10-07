import unittest

import numpy as np

from process.cdcHeatmap import (
    build_density_heatmap_from_run_columns,
    build_density_heatmap_from_runs,
)


class BuildDensityHeatmapFromRunsTest(unittest.TestCase):
    def test_returns_edges_and_column_normalised_density(self):
        ages_ma = np.array([100.0, 200.0, 300.0], float)
        s_runs = np.array(
            [
                [0.90, 0.80, 0.70],
                [0.80, 0.75, 0.65],
                [0.85, 0.70, np.nan],
            ],
            float,
        )

        x_edges, y_edges, density = build_density_heatmap_from_runs(
            ages_ma,
            s_runs,
            resolution=4,
        )

        self.assertEqual(x_edges.shape, (4,))
        self.assertEqual(y_edges.shape, (5,))
        self.assertEqual(density.shape, (4, 3))
        self.assertTrue(np.allclose(np.sum(density, axis=0), 1.0))

    def test_empty_column_reuses_previous_histogram(self):
        ages_ma = np.array([100.0, 200.0], float)
        s_runs = np.array(
            [
                [0.90, np.nan],
                [0.80, np.nan],
            ],
            float,
        )

        _, _, density = build_density_heatmap_from_runs(
            ages_ma,
            s_runs,
            resolution=5,
        )

        self.assertTrue(np.allclose(density[:, 1], density[:, 0]))


class BuildDensityHeatmapFromRunColumnsTest(unittest.TestCase):
    class _Settings:
        minimumRimAge = 100_000_000.0
        maximumRimAge = 300_000_000.0

    class _Run:
        def __init__(self, values):
            self.heatmapColumnData = values

    def test_returns_square_density_surface_over_requested_age_range(self):
        runs = [
            self._Run([0.10, 0.20, 0.30, 0.40]),
            self._Run([0.20, 0.25, 0.35, 0.45]),
            self._Run([0.15, 0.30, 0.40, 0.50]),
        ]

        x_edges, y_edges, density = build_density_heatmap_from_run_columns(
            runs,
            self._Settings(),
            resolution=4,
        )

        self.assertTrue(np.allclose(x_edges, [100.0, 150.0, 200.0, 250.0, 300.0]))
        self.assertEqual(y_edges.shape, (5,))
        self.assertEqual(density.shape, (4, 4))
        self.assertTrue(np.all(np.isfinite(density)))
        self.assertTrue(np.all(density >= 0.0))
        self.assertTrue(np.all(np.sum(density, axis=0) > 0.0))


if __name__ == "__main__":
    unittest.main()
