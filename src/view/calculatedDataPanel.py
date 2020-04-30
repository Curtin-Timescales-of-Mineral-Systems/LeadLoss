from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QVBoxLayout, QTableWidgetItem, QGroupBox, QPushButton, QLineEdit, QWidget, QHBoxLayout, \
    QTableWidget, QStyle, QFormLayout

from model.settings.type import SettingsType
from utils import config, stringUtils
from utils.settings import Settings
from utils.ui import uiUtils
from utils.ui.numericInput import AgeInput, FloatInput


class CalculatedDataPanel(QGroupBox):

    HEADERS = ["Pb-loss age (Ma)", "D value", "p Value"]

    def __init__(self, controller, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.controller = controller
        self._initUI()

        self.controller.signals.statisticUpdated.connect(self.onStatisticUpdated)
        self.controller.signals.optimalAgeFound.connect(self.onOptimalAgeFound)

        self.controller.signals.processingCleared.connect(self.onProcessingCleared)
        self.controller.signals.processingStarted.connect(self.onProcessingStart)



    #############
    ## UI spec ##
    #############

    def _initUI(self):
        self._initDataTable()
        self._initActionButtonsWidget()
        self._initOutputWidget()

        layout = QVBoxLayout()
        layout.addWidget(self.outputWidget)
        layout.addWidget(self.dataTable)
        layout.addWidget(self.actionButtonsWidget)
        self.setLayout(layout)

        self.actionButtonsWidget.hide()

    def _initDataTable(self):
        self.dataTable = QTableWidget(0, len(self.HEADERS))
        self.dataTable.setHorizontalHeaderLabels(self.HEADERS)

        self.dataTable.resizeColumnsToContents()
        self.dataTable.resizeRowsToContents()
        uiUtils.retainSizeWhenHidden(self.dataTable)

    def _initTableWidgetItem(self, content):
        roundedContent = str(stringUtils.round_to_sf(content, 5))
        cell = QTableWidgetItem(roundedContent)
        cell.setTextAlignment(Qt.AlignHCenter)
        cell.setFlags(cell.flags() ^ Qt.ItemIsEditable)
        return cell

    def _initActionButtonsWidget(self):
        self.processButton = QPushButton("  Process")
        self.processButton.clicked.connect(self.controller.process)
        self.processButton.setIcon(self.style().standardIcon(QStyle.SP_ArrowForward))

        self.exportButton = QPushButton("  Export CSV")
        self.exportButton.clicked.connect(self.controller.exportCSV)
        self.exportButton.setEnabled(False)
        self.exportButton.setIcon(self.style().standardIcon(QStyle.SP_DialogSaveButton))

        layout = QHBoxLayout()
        layout.addWidget(self.processButton)
        layout.addWidget(self.exportButton)
        layout.setContentsMargins(0, 0, 0, 5)

        self.actionButtonsWidget = QWidget()
        self.actionButtonsWidget.setLayout(layout)

    def _initOutputWidget(self):
        self.optimalLeadLossAge = AgeInput(defaultValue=None, sf=5)
        self.optimalLeadLossAge.setReadOnly(True)

        self.dValue = FloatInput(defaultValue=None, sf=5)
        self.dValue.setReadOnly(True)

        self.pValue = FloatInput(defaultValue=None, sf=5)
        self.dValue.setReadOnly(True)

        layout = QFormLayout()
        layout.addRow("Optimal radiogenic-Pb loss age", self.optimalLeadLossAge)
        layout.addRow("D value", self.dValue)
        layout.addRow("p value", self.pValue)
        layout.setContentsMargins(0, 5, 0, 5)

        self.outputWidget = QWidget()
        self.outputWidget.setLayout(layout)

    def getActionButtons(self):
        return [self.importButton, self.processButton, self.exportButton]

    #############
    ## Actions ##
    #############

    def _setRowData(self, i, dValue, pValue):
        dCell = self._initTableWidgetItem(dValue)
        pCell = self._initTableWidgetItem(pValue)
        self.dataTable.setItem(i, 1, dCell)
        self.dataTable.setItem(i, 2, pCell)

    def _resizeTable(self):
        self.dataTable.resizeColumnsToContents()
        self.dataTable.resizeRowsToContents()
        self.dataTable.viewport().update()

    #######################
    ## Processing events ##
    #######################

    def onProcessingCleared(self):
        self.optimalLeadLossAge.setValue(None)
        self.pValue.setValue(None)
        self.dValue.setValue(None)
        self.dataTable.setRowCount(0)

    def onProcessingStart(self):
        settings = Settings.get(SettingsType.CALCULATION)
        minAge = settings.minimumRimAge
        maxAge = settings.maximumRimAge
        samples = settings.rimAgesSampled

        self.dataTable.setRowCount(samples)
        for i in range(samples):
            rimAge = minAge + i * ((maxAge - minAge) / (samples - 1))
            rimAge /= (10 ** 6)
            tableCell = self._initTableWidgetItem(rimAge)
            self.dataTable.setItem(i, 0, tableCell)

        self._resizeTable()

    def onStatisticUpdated(self, i, dValue, pValue):
        self._setRowData(i, dValue, pValue)
        if i == 0:
            self._resizeTable()
        else:
            self.dataTable.viewport().update()

    def onOptimalAgeFound(self, rimAge, pValue, dValue, reconstructedAges, reconstructedAgeRange):
        self.optimalLeadLossAge.setValue(rimAge)
        self.pValue.setValue(stringUtils.round_to_sf(pValue))
        self.dValue.setValue(stringUtils.round_to_sf(dValue))


