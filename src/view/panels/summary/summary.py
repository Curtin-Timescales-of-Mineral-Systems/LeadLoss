from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QSplitter

from view.panels.summary.data import SummaryDataPanel
from view.figures.summaryFigure import SummaryFigure


class SummaryPanel(QSplitter):

    def __init__(self, controller, samples):
        super().__init__(Qt.Horizontal)

        self.data = SummaryDataPanel(controller, samples)
        self.figure = SummaryFigure(controller, samples)

        self.addWidget(self.data)
        self.addWidget(self.figure)
        self.setSizes([10000, 10000])
        self.setContentsMargins(1, 1, 1, 1)

    def getButtons(self):
        return []