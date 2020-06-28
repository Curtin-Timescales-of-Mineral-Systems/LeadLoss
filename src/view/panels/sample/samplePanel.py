from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTabWidget

from view.panels.sample.sampleInputPanel import SampleInputDataPanel
from view.panels.sample.sampleMonteCarloPanel import SampleOutputMonteCarloPanel
from view.panels.sample.sampleOutputPanel import SampleOutputPanel


class SamplePanel(QWidget):

    def __init__(self, controller, sample):
        super().__init__()
        self.sample = sample

        self.inputDataPanel = SampleInputDataPanel(controller, sample)
        self.summaryPanel = SampleOutputPanel(controller, sample)
        self.monteCarloPanel = SampleOutputMonteCarloPanel(controller, sample)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.inputDataPanel, "Input")
        self.tabs.addTab(self.summaryPanel, "Output")
        self.tabs.addTab(self.monteCarloPanel, "Monte-Carlo runs")

        layout = QVBoxLayout()
        layout.addWidget(self.tabs)
        self.setLayout(layout)

    def getButtons(self):
        return self.inputDataPanel.getActionButtons()

    def getSplitters(self):
        return self.inputDataPanel.getSplitters() + self.summaryPanel.getSplitters() + self.monteCarloPanel.getSplitters()