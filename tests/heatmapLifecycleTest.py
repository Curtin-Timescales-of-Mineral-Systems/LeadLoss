import os
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure

from model.settings.calculation import LeadLossCalculationSettings
from process.cdcHeatmap import calculateHeatmapData
from view.axes.heatmapAxis import HeatmapAxis


class _Signals:
    def __init__(self):
        self.events = []

    def progress(self, *args):
        self.events.append("progress")

    def completed(self):
        self.events.append("completed")


class _Run:
    def __init__(self):
        self.heatmapColumnData = np.linspace(0.2, 0.8, 100)


class HeatmapLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_background_job_reports_completion(self):
        signals = _Signals()
        calculateHeatmapData(signals, [_Run()], LeadLossCalculationSettings(), 1)
        self.assertEqual(signals.events, ["progress", "completed"])

    def test_completed_runs_draw_heatmap(self):
        figure = Figure()
        axis = HeatmapAxis(figure.add_subplot(111), FigureCanvas(figure), figure)
        axis.plotFinalRuns([_Run()], LeadLossCalculationSettings())
        self.assertEqual(len(axis.axis.collections), 1)


if __name__ == "__main__":
    unittest.main()
