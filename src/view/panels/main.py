from PyQt5.QtWidgets import QWidget, QTabWidget, QVBoxLayout, QPushButton, QLabel, QLineEdit, QHBoxLayout, QSpacerItem, QMessageBox, QSizePolicy
from PyQt5.QtCore import QSignalBlocker
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

    @staticmethod
    def _repolish(w):
        w.style().unpolish(w)
        w.style().polish(w)
        w.update()

    def _createTop(self, file, samples):
        layout = QHBoxLayout()
        layout.setContentsMargins(10, 10, 10, 6)
        layout.setSpacing(10)

        # Process all
        self.processAllButton = QPushButton()
        self.processAllButton.setObjectName("PrimaryButton")
        self.processAllButton.setIcon(Icons.process())
        self.processAllButton.clicked.connect(self.onProcessAllClicked)
        layout.addWidget(self.processAllButton)

        # Process one (only if multiple samples)
        if len(samples) == 1:
            processAllText = "  Process"
            self.processOneButton = None
        else:
            processAllText = "  Process all"
            self.processOneButton = QPushButton("  Process sample")
            self.processOneButton.setObjectName("PrimaryButton")
            self.processOneButton.setIcon(Icons.process())
            self.processOneButton.clicked.connect(self.onProcessSampleClicked)
            layout.addWidget(self.processOneButton)

        self.processAllButton.setText(processAllText)

        # Cancel
        self.cancelButton = QPushButton("  Cancel processing")
        self.cancelButton.setObjectName("DangerButton")
        self.cancelButton.setIcon(Icons.cancel())
        self.cancelButton.clicked.connect(self.onProcessCancelClicked)
        self.cancelButton.setVisible(False)
        layout.addWidget(self.cancelButton)

        layout.addSpacing(10)

        # File path
        self.importFileLabel = QLabel("Current file:")
        layout.addWidget(self.importFileLabel)

        self.importFileText = QLineEdit(file)
        self.importFileText.setReadOnly(True)
        self.importFileText.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addWidget(self.importFileText, 1)

        # Export
        self.exportButton = QPushButton("Export Monte Carlo Runs")
        self.exportButton.setObjectName("GoldButton")
        self.exportButton.clicked.connect(self.onExportClicked)
        layout.addWidget(self.exportButton)

        self.exportPeaksButton = QPushButton("Export Monte Carlo Peak Picks")
        self.exportPeaksButton.setObjectName("GoldButton")
        self.exportPeaksButton.clicked.connect(self.controller.exportPerRunPeaks)
        layout.addWidget(self.exportPeaksButton)

        # If QSS still doesn't apply, force it:
        self._repolish(self.processAllButton)
        if self.processOneButton:
            self._repolish(self.processOneButton)
        self._repolish(self.cancelButton)
        self._repolish(self.exportButton)

        widget = QWidget()
        widget.setObjectName("TopBar")
        widget.setLayout(layout)
        return widget

    
    def _createBody(self, controller, samples):
        if len(samples) == 1:
            samplePanel = SamplePanel(controller, samples[0])
            self.samplePanels.append(samplePanel)
            return samplePanel

        self.tabs = QTabWidget()
        self.tabs.setObjectName("MainTabs")
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
        # Prevent re-entry if something else triggers while we're syncing
        if getattr(self, "_syncing_tabs", False):
            return
        self._syncing_tabs = True
        try:
            sender_tabs = self.sender()  # QTabWidget that fired the signal
            for samplePanel in self.samplePanels:
                tabs = samplePanel.tabs
                # Skip the sender, and skip if already on i
                if tabs is sender_tabs or tabs.currentIndex() == i:
                    continue
                # Block signals so setCurrentIndex doesn't re-emit currentChanged
                with QSignalBlocker(tabs):
                    tabs.setCurrentIndex(i)
        finally:
            self._syncing_tabs = False


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
