from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTabWidget, QSplitter

from model.settings.calculation import ConcordiaMode
from view.figures.summaryFigure import SummaryFigure
from view.figures.summaryWetherillFigure import SummaryWetherillFigure
from view.panels.summary.data import SummaryDataPanel


class SummaryPanel(QWidget):
    def __init__(self, controller, samples):
        super().__init__()

        tabs = QTabWidget()
        tabs.addTab(self._make_mode_panel(controller, samples, ConcordiaMode.TW), "TW summary")
        tabs.addTab(self._make_mode_panel(controller, samples, ConcordiaMode.WETHERILL), "Wetherill summary")

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(tabs)
        self.setLayout(layout)

    def _make_mode_panel(self, controller, samples, mode: ConcordiaMode):
        split = QSplitter(Qt.Horizontal)

        data = SummaryDataPanel(controller, samples, mode=mode)

        if ConcordiaMode.coerce(mode) == ConcordiaMode.TW:
            fig = SummaryFigure(controller, samples)
        else:
            fig = SummaryWetherillFigure(controller, samples)

        split.addWidget(data)
        split.addWidget(fig)
        split.setSizes([10000, 10000])
        split.setContentsMargins(1, 1, 1, 1)
        return split

    def getButtons(self):
        return []
