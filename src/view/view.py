from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QSplitter, QVBoxLayout, QFileDialog, QDialog, QMessageBox, QWidget

from utils.settings import Settings
from view.calculatedDataPanel import CalculatedDataPanel
from view.importedDataPanel import ImportedDataPanel
from view.graphPanel import LeadLossGraphPanel

from view.settingsDialogs.calculation import LeadLossCalculationSettingsDialog
from view.settingsDialogs.imports import LeadLossImportSettingsDialog
from model.settings.type import SettingsType
from utils.ui.statusBar import StatusBarWidget

class LeadLossView(QWidget):

    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.initUI()

    def initUI(self):
        self.graphPanel = LeadLossGraphPanel(self.controller)
        self.importedDataPanel = ImportedDataPanel(self.controller)
        self.calculatedDataPanel = CalculatedDataPanel(self.controller)
        self.statusBar = StatusBarWidget(self.controller.signals)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.importedDataPanel)
        splitter.addWidget(self.calculatedDataPanel)
        splitter.addWidget(self.graphPanel)
        splitter.setSizes([10000, 5000, 10000])
        splitter.setContentsMargins(1, 1, 1, 1)

        layout = QVBoxLayout()
        layout.addWidget(splitter, 1)
        layout.addWidget(self.statusBar, 0)
        self.setLayout(layout)

    ########
    ## IO ##
    ########

    def getInputFile(self):
        return QFileDialog.getOpenFileName(
            caption='Open CSV file',
            directory='/home/matthew/Dropbox/Academia/Code/Python/UnmixConcordia/tests'
        )[0]
        # return '/home/matthew/Dropbox/Academia/Code/Python/UnmixConcordia/data/unmixTest.csv'

    def getOutputFile(self):
        return QFileDialog.getSaveFileName(
            caption='Save CSV file',
            directory='/home/matthew/Dropbox/Academia/Code/Python/UnmixConcordia/tests'
        )[0]

    def getSettings(self, settingsType, rows, callback):
        settingsPopup = self._getSettingsDialog(settingsType, rows)

        def outerCallback(result):
            if result == QDialog.Rejected:
                return None
            callback(settingsPopup.settings)

        settingsPopup.finished.connect(outerCallback)
        settingsPopup.show()

    def _getSettingsDialog(self, settingsType, rows):
        defaultSettings = Settings.get(settingsType)
        if settingsType == SettingsType.IMPORT:
            return LeadLossImportSettingsDialog(defaultSettings)
        if settingsType == SettingsType.CALCULATION:
            return LeadLossCalculationSettingsDialog(defaultSettings, rows)

        raise Exception("Unknown settingsDialogs " + str(type(settingsType)))
