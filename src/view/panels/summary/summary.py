from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QSplitter, QTabWidget

from view.figures.summaryFigure import SummaryFigure
from view.figures.summaryWetherillFigure import SummaryWetherillFigure
from view.panels.summary.data import SummaryDataPanel



class SummaryPanel(QSplitter):

    def __init__(self, controller, samples):
        super().__init__(Qt.Horizontal)

        self.data = SummaryDataPanel(controller, samples)

        self.twFigure = SummaryFigure(controller, samples)
        self.wetherillFigure = SummaryWetherillFigure(controller, samples)

        self.figureTabs = QTabWidget()
        self.figureTabs.addTab(self.twFigure, "TW concordia")
        self.figureTabs.addTab(self.wetherillFigure, "Wetherill concordia")

        self.addWidget(self.data)
        self.addWidget(self.figureTabs)

        self.setSizes([10000, 10000])
        self.setContentsMargins(1, 1, 1, 1)

    def getButtons(self):
        return []
