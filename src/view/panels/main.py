from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QTabWidget, QVBoxLayout, QPushButton, QLabel, QLineEdit, QHBoxLayout, QStyle, \
    QSpacerItem, QMessageBox

from utils.ui.icons import Icons
from view.panels.sample.samplePanel import SamplePanel
from view.panels.summary.summary import SummaryPanel


class MainPanel(QWidget):

    def __init__(self, controller, file, samples):
        super().__init__()
        self.controller = controller
        self.samplePanels = []

        layout = QVBoxLayout()
        layout.addWidget(self._createTop(file, samples))
        layout.addWidget(self._createBody(controller, samples))
        self.setLayout(layout)

        controller.signals.processingStarted.connect(self._onProcessingStarted)
        controller.signals.processingFinished.connect(self._onProcessingFinished)

    #############
    ## Widgets ##
    #############

    def _createTop(self, file, samples):
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 5)

        self.processAllButton = QPushButton()
        self.processAllButton.clicked.connect(self.onProcessAllClicked)
        self.processAllButton.setIcon(Icons.process())
        layout.addWidget(self.processAllButton)

        if len(samples) == 1:
            processAllText = "  Process"
            self.processOneButton = None
        else:
            processAllText = "  Process all"
            self.processOneButton = QPushButton("  Process sample")
            self.processOneButton.clicked.connect(self.onProcessSampleClicked)
            self.processOneButton.setIcon(Icons.process())
            layout.addWidget(self.processOneButton)
        self.processAllButton.setText(processAllText)

        self.cancelButton = QPushButton("  Cancel processing")
        self.cancelButton.clicked.connect(self.onProcessCancelClicked)
        self.cancelButton.setIcon(Icons.cancel())
        self.cancelButton.setVisible(False)
        layout.addWidget(self.cancelButton)

        layout.addSpacerItem(QSpacerItem(20, 0))

        self.importFileLabel = QLabel("Current file: ")

        self.importFileText = QLineEdit(file)
        self.importFileText.setReadOnly(True)

        layout.addWidget(self.importFileLabel)
        layout.addWidget(self.importFileText)

        # Add a button to export Monte Carlo runs
        self.exportButton = QPushButton("Export Monte Carlo Runs")
        self.exportButton.clicked.connect(self.onExportClicked)
        layout.addWidget(self.exportButton)

        widget = QWidget()
        widget.setLayout(layout)
        return widget

    def _createBody(self, controller, samples):
        if len(samples) == 1:
            samplePanel = SamplePanel(controller, samples[0])
            self.samplePanels.append(samplePanel)
            return samplePanel

        self.tabs = QTabWidget()
        self.tabs.currentChanged.connect(self.onTabChanged)

        summaryPanel = SummaryPanel(controller, samples)
        self.tabs.addTab(summaryPanel, "Summary")

        for sample in samples:
            samplePanel = SamplePanel(controller, sample)
            samplePanel.tabs.currentChanged.connect(self.onSampleTabChanged)
            self.tabs.addTab(samplePanel, sample.name)
            self.samplePanels.append(samplePanel)

        return self.tabs

    #############
    ## Getters ##
    #############

    def _getButtons(self):
        return [button for panel in self.samplePanels for button in panel.getButtons()]

    ############
    ## Events ##
    ############

    def onTabChanged(self, i):
        if i == 0:
            self.processOneButton.setVisible(False)
        else:
            self.processOneButton.setVisible(True)
            self.processOneButton.setText("  Process " + self.samplePanels[i-1].sample.name)

    def onSampleTabChanged(self, i):
        for samplePanel in self.samplePanels:
            samplePanel.tabs.setCurrentIndex(i)

    def onProcessAllClicked(self):
        self.controller.processAllSamples()

    def onProcessSampleClicked(self):
        sampleTab = self.tabs.currentWidget()
        self.controller.processSample(sampleTab.sample)

    def onProcessCancelClicked(self):
        self.controller.cancelProcessing()

    def _onProcessingStarted(self):
        if self.processOneButton:
            self.processOneButton.setVisible(False)
        self.processAllButton.setVisible(False)
        self.cancelButton.setVisible(True)

    def _onProcessingFinished(self):
        if self.processOneButton:
            self.processOneButton.setVisible(self.tabs.currentIndex() != 0)
        self.processAllButton.setVisible(True)
        self.cancelButton.setVisible(False)
    
    def onExportClicked(self):
        try:
            self.controller.exportMonteCarloRuns() 
        except IOError as e:
            self.controller.exception_hook(type(e), e, e.__traceback__)
            # Show an error message to the user
            QMessageBox.critical(self, "Export Error", f"An error occurred while exporting Monte Carlo runs: {e}")
