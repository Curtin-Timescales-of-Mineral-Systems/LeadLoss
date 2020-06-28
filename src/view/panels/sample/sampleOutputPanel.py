from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QSplitter, QVBoxLayout

from utils.ui import uiUtils
from view.figures.sampleOutputFigure import SampleOutputFigure
from view.panels.sample.output.sampleCalculationSettingsPanel import SampleCalculationSettingsPanel
from view.panels.sample.output.sampleOutputResultsPanel import SampleOutputResultsPanel
from view.panels.sample.output.sampleOutputSpotClassificationPanel import SampleOutputSpotClassificationPanel


class SampleOutputPanel(QWidget):

    def __init__(self, controller, sample):
        super().__init__()
        self.sample = sample
        self.controller = controller

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.dataWidget = self._createDataWidget()
        self.noDataWidget = uiUtils.createNoDataWidget(self.sample.name)

        self._showNoDataPanel()

        self.sample.signals.concordancyCalculated.connect(self._onConcordanceCalculated)

    ########
    ## UI ##
    ########

    def _createDataWidget(self):
        self.spotClassificationWidget = SampleOutputSpotClassificationPanel(self.sample)
        self.resultsAndSettingsWidget = self._createResultsAndSettingsWidget()
        self.graphWidget = SampleOutputFigure(self.controller, self.sample)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.resultsAndSettingsWidget)
        splitter.addWidget(self.spotClassificationWidget)
        splitter.addWidget(self.graphWidget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(0, 3)
        splitter.setContentsMargins(0, 0, 0, 0)
        return splitter

    def _createResultsAndSettingsWidget(self):
        self.resultsPanel = SampleOutputResultsPanel(self.sample)
        self.settingsPanel = SampleCalculationSettingsPanel(self.sample)

        layout = QVBoxLayout()
        layout.addWidget(self.resultsPanel)
        layout.addWidget(self.settingsPanel)

        widget = QWidget()
        widget.setLayout(layout)
        widget.setContentsMargins(0,0,0,0)
        return widget

    #############
    ## Actions ##
    #############

    def _showNoDataPanel(self):
        uiUtils.clearChildren(self.layout)
        self.layout.addWidget(self.noDataWidget)

    def _showDataPanel(self):
        uiUtils.clearChildren(self.layout)
        self.layout.addWidget(self.dataWidget)

    ############
    ## Events ##
    ############

    def _onConcordanceCalculated(self):
        self._showDataPanel()