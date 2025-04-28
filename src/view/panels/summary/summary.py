from PyQt5.QtWidgets import QSplitter, QTabWidget
from PyQt5.QtCore import Qt
from view.figures.summaryFigure import SummaryFigure
from view.axes.concordia.summaryWetherillConcordiaAxis import SummaryWetherillConcordiaAxis
from view.figures.abstractFigure import AbstractFigure
from view.panels.summary.data import SummaryDataPanel
from view.figures.summaryWetherillFigure import SummaryWetherillFigure
import matplotlib.pyplot as plt

class SummaryPanel(QSplitter):
    def __init__(self, controller, samples):
        super().__init__(Qt.Horizontal)

        # Left side: summary table
        self.data = SummaryDataPanel(controller, samples)

        # Right side: tabbed for Tera-W vs. Wetherill
        self.tabs = QTabWidget()

        # 1) Tera-W existing figure
        self.twFigure = SummaryFigure(controller, samples)

        # 2) Wetherill figure
        self.wethFigure = SummaryWetherillFigure(controller, samples)

        self.tabs.addTab(self.twFigure, "Tera-Wasserburg")
        self.tabs.addTab(self.wethFigure, "Wetherill")

        # Add them side by side
        self.addWidget(self.data)
        self.addWidget(self.tabs)
        self.setSizes([10000, 10000])
