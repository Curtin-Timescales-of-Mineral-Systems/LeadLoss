from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QSplitter, QVBoxLayout, QScrollArea

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

        self.sample.signals.skipped.connect(self._onSampleSkipped)
    
    ########
    ## UI ##
    ########

    def _createDataWidget(self):
        self.spotClassificationWidget = SampleOutputSpotClassificationPanel(self.sample)
        self.resultsAndSettingsWidget = self._createResultsAndSettingsWidget()
        self.resultsAndSettingsScroll = QScrollArea()
        self.resultsAndSettingsScroll.setWidgetResizable(True)
        self.resultsAndSettingsScroll.setFrameShape(QScrollArea.NoFrame)
        self.resultsAndSettingsScroll.setWidget(self.resultsAndSettingsWidget)
        self.graphWidget = SampleOutputFigure(self.controller, self.sample)

        try:
            self.resultsPanel.peakRowSelected.connect(
                lambda idx: self.graphWidget.highlight_catalogue_row(idx if (idx is not None and idx >= 0) else None)
            )
        except Exception:
            pass

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.resultsAndSettingsScroll)
        splitter.addWidget(self.spotClassificationWidget)
        splitter.addWidget(self.graphWidget)

        # Make the left panel relatively wide, the middle smaller, graphs moderate
        splitter.setStretchFactor(0, 3)  # results/settings
        splitter.setStretchFactor(1, 2)  # spot classification
        splitter.setStretchFactor(2, 3)  # graphs

        # Optional: set a minimum width for the results/settings column
        self.resultsAndSettingsScroll.setMinimumWidth(360)
        splitter.setChildrenCollapsible(False)

        # Optional: give an initial size distribution (pixels, Qt normalises)
        splitter.setSizes([520, 320, 640])

        splitter.setContentsMargins(0, 0, 0, 0)
        return splitter

    def _createResultsAndSettingsWidget(self):
        self.resultsPanel = SampleOutputResultsPanel(self.controller, self.sample)
        self.settingsPanel = SampleCalculationSettingsPanel(self.sample)

        layout = QVBoxLayout()
        layout.addWidget(self.resultsPanel, 3)   # more space to show the catalogue
        layout.addWidget(self.settingsPanel, 1)

        widget = QWidget()
        widget.setLayout(layout)
        widget.setContentsMargins(0,0,0,0)
        return widget

    #############
    ## Actions ##
    #############

    def _onSampleSkipped(self):
        self._showNoDataPanel()
    
    def _showNoDataPanel(self):
        uiUtils.clearChildren(self.layout)
        if self.sample.skip_reason:
            message = f"Skipped during processing. Sample '{self.sample.name}' has {self.sample.skip_reason}."
        else:
            message = None
        self.noDataWidget = uiUtils.createNoDataWidget(self.sample.name, message)
        if self.noDataWidget is not None:
            self.layout.addWidget(self.noDataWidget)

    def _showDataPanel(self):
        uiUtils.clearChildren(self.layout)
        if self.dataWidget is not None:
            self.layout.addWidget(self.dataWidget)

    ############
    ## Events ##
    ############

    def _onConcordanceCalculated(self):
        self._showDataPanel()
