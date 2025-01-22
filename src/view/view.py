from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QVBoxLayout, QFileDialog, QDialog, QWidget, QTabWidget, QPushButton, QLineEdit, QHBoxLayout, \
    QStyle, QMainWindow, QAction

from utils.settings import Settings
from utils.ui import icons, uiUtils
from utils.ui.icons import Icons

from view.dialogs.settings.calculation import LeadLossCalculationSettingsDialog
from view.dialogs.settings.imports import LeadLossImportSettingsDialog
from model.settings.type import SettingsType
from utils.ui.statusBar import StatusBarWidget, QLabel
from view.panels.main import MainPanel
from view.panels.sample.samplePanel import SamplePanel
from view.panels.summary.data import SummaryDataPanel
from view.panels.summary.summary import SummaryPanel
from view.panels.welcome import WelcomePanel


class LeadLossView(QMainWindow):

    def __init__(self, controller, title, version):
        super().__init__()
        self.controller = controller

        self.left = 10
        self.top = 10
        self.width = 1220
        self.height = 500

        self.setWindowTitle(title + " (v" + version + ")")
        self.setGeometry(self.left, self.top, self.width, self.height)

        self._createMenuBar()
        self._createCentralWidget()

        controller.signals.inputDataLoaded.connect(self._onInputDataLoaded)

    def _createMenuBar(self):
        menubar = self.menuBar()

        importAction = QAction(Icons.importCSV(), "Import CSV", self)
        importAction.triggered.connect(self.controller.importCSV)

        closeAction = QAction(Icons.close(), "Exit", self)
        closeAction.triggered.connect(self.close)

        helpAction = QAction(Icons.help(), "Help", self)
        helpAction.triggered.connect(self.controller.showHelp)

        fileMenu = menubar.addMenu("File")
        fileMenu.addAction(importAction)
        fileMenu.addAction(closeAction)

        fileMenu = menubar.addMenu("Help")
        fileMenu.addAction(helpAction)


    def _createCentralWidget(self):
        self.bottomPanel = self._createBottomPanel()

        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget)

        self.welcomePanel = WelcomePanel(self.controller)
        self.mainPanel = None

        self.layout = QVBoxLayout()
        self.centralWidget.setLayout(self.layout)
        self.showWelcomePanel()


    def _createBottomPanel(self):
        return StatusBarWidget(self.controller.signals)

    #############
    ## Actions ##
    #############

    def showWelcomePanel(self):
        if self.mainPanel:
            self.mainPanel.setParent(None)
            self.layout.removeWidget(self.mainPanel)

        self.bottomPanel.setParent(None)
        self.layout.removeWidget(self.bottomPanel)
        self.layout.addWidget(self.welcomePanel)
        self.layout.addWidget(self.bottomPanel)

    def showMainPanel(self, file, samples):
        uiUtils.clearChildren(self.layout)

        self.mainPanel = MainPanel(self.controller, file, samples)
        self.layout.addWidget(self.mainPanel)
        self.layout.addWidget(self.bottomPanel)

    ############
    ## Events ##
    ############

    def _onInputDataLoaded(self, file, samples):
        self.showMainPanel(file, samples)

    ########
    ## IO ##
    ########

    def getInputFile(self):
        return QFileDialog.getOpenFileName(
            caption='Open CSV file',
            directory='.',
            options=QFileDialog.DontUseNativeDialog
        )[0]

    def getOutputFile(self):
        return QFileDialog.getSaveFileName(
            caption='Save CSV file',
            directory='.',
            options=QFileDialog.DontUseNativeDialog
        )[0]

    def getImportSettings(self, callback):
        defaultSettings = Settings.get(SettingsType.IMPORT)
        dialog = LeadLossImportSettingsDialog(defaultSettings)
        self._getSettings(dialog, callback)

    def getCalculationSettings(self, samples, defaultSettings, callback):
        dialog = LeadLossCalculationSettingsDialog(samples, defaultSettings)
        self._getSettings(dialog, callback)

    def _getSettings(self, dialog, callback):
        def outerCallback(result):
            if result == QDialog.Rejected:
                return None

            callback(dialog.settings)

        dialog.finished.connect(outerCallback)
        dialog.show()

