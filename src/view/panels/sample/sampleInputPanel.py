from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QVBoxLayout, QWidget, QSplitter, QTabWidget

from model.settings.type import SettingsType
from utils.settings import Settings
from utils.ui import uiUtils, spotTable
from utils.ui.icons import Icons
from view.figures.sampleInputFigure import SampleInputFigure
from view.figures.sampleInputWetherillFigure import SampleInputWetherillFigure

class SampleInputDataPanel(QWidget):

    def __init__(self, controller, sample):
        super().__init__()
        self.sample = sample
        self.controller = controller

        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(self._createLHS())
        self.splitter.addWidget(self._createRHS())
        self.splitter.setContentsMargins(0, 0, 0, 0)

        layout = QVBoxLayout()
        layout.addWidget(self.splitter)
        self.setLayout(layout)

    #################
    ## UI creation ##
    #################

    def _createLHS(self):
        invalidWarningWidget = self._createInvalidWarningWidget()
        headers = Settings.get(SettingsType.IMPORT).getHeaders()
        self.unclassifiedTableWidget = spotTable.createSpotTable(headers, self.sample.spots)

        layout = QVBoxLayout()
        if invalidWarningWidget:
            layout.addWidget(invalidWarningWidget)
        layout.addWidget(self.unclassifiedTableWidget)

        widget = QWidget()
        widget.setLayout(layout)
        return widget

    def _createRHS(self):
        # Create a QTabWidget
        tabWidget = QTabWidget()

        # Tab 1: Tera‐W input figure
        twFigure = SampleInputFigure(self.sample)
        tabWidget.addTab(twFigure, "Tera-Wasserburg")

        # Tab 2: Wetherill input figure
        wethFigure = SampleInputWetherillFigure(self.sample)
        tabWidget.addTab(wethFigure, "Wetherill")

        return tabWidget

    def _createInvalidWarningWidget(self):
        n = len(self.sample.invalidSpots)
        if n == 0:
            return None

        if n == 1:
            pointText = str(n) + " invalid point"
        else:
            pointText = str(n) + " invalid points"
        return uiUtils.createIconWithLabel(Icons.warning(), pointText + " found")

    #################
    ## UI creation ##
    #################

    def getSplitters(self):
        return [self.splitter]