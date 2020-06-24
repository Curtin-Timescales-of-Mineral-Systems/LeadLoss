from PyQt5.QtWidgets import QVBoxLayout, QGroupBox, QWidget, QHBoxLayout, \
    QFormLayout, QLabel, QSpinBox, QSplitter

from utils.ui import uiUtils
from utils.ui.numericInput import AgeInput, FloatInput
from view.figures.sampleMonteCarloFigure import SampleMonteCarloFigure


class SampleOutputMonteCarloPanel(QWidget):

    HEADERS = ["Pb-loss age (Ma)", "D value", "p value"]

    def __init__(self, controller, sample):
        super().__init__()
        self.controller = controller
        self.sample = sample
        self.dataTable = None
        self.figure = None

        self.layout = QHBoxLayout()
        self.setLayout(self.layout)

        self.noDataWidget = uiUtils.createNoDataWidget(self.sample.name)
        self.dataWidget = self._createDataWidget()
        self._showNoDataPanel()

        sample.signals.monteCarloRunAdded.connect(self._onMonteCarloRunAdded)

    #############
    ## UI spec ##
    #############

    def _createDataWidget(self):
        #uiUtils.clearChildren(self.splitter)
        self.splitter = QSplitter()
        self.splitter.addWidget(self._createDataLHS())
        self.splitter.addWidget(self._createDataRHS())
        return self.splitter

    def _createDataLHS(self):
        layout = QVBoxLayout()
        layout.addWidget(self._createRunSpinner())
        layout.addWidget(self._createResultsWidget())

        widget = QWidget()
        widget.setLayout(layout)
        widget.setContentsMargins(0,0,0,0)
        return widget

    def _createRunSpinner(self):
        self.selectedRunInput = QSpinBox()
        self.selectedRunInput.setMinimum(1)
        self.selectedRunInput.setMaximum(1)
        self.selectedRunInput.valueChanged.connect(self._onSelectedMonteCarloRunChanged)

        self.maximumRunLabel = QLabel("/")

        runLayout = QHBoxLayout()
        runLayout.addWidget(self.selectedRunInput)
        runLayout.addWidget(self.maximumRunLabel)
        runLayout.setContentsMargins(0,0,0,0)

        runWidget = QWidget()
        runWidget.setLayout(runLayout)

        layout = QFormLayout()
        layout.addRow("Run: ", runWidget)

        widget = QWidget()
        widget.setLayout(layout)
        return widget

    def _createResultsWidget(self):
        self.optimalLeadLossAge = AgeInput(defaultValue=None, sf=5)
        self.optimalLeadLossAge.setReadOnly(True)

        self.dValue = FloatInput(defaultValue=None, sf=5)
        self.dValue.setReadOnly(True)

        self.pValue = FloatInput(defaultValue=None, sf=5)
        self.pValue.setReadOnly(True)

        layout = QFormLayout()
        layout.addRow("Optimal Pb-loss age", self.optimalLeadLossAge)
        layout.addRow("D value", self.dValue)
        layout.addRow("p value", self.pValue)
        layout.setContentsMargins(0, 5, 0, 5)

        widget = QGroupBox("Result")
        widget.setLayout(layout)
        return widget

    def _createDataRHS(self):
        self.figure = SampleMonteCarloFigure(self.controller)
        return self.figure

    #############
    ## Actions ##
    #############

    def _showNoDataPanel(self):
        uiUtils.clearChildren(self.layout)
        self.layout.addWidget(self.noDataWidget)

    def _showDataPanel(self):
        uiUtils.clearChildren(self.layout)
        self.layout.addWidget(self.dataWidget)

    def _selectRun(self, run):
        self.figure.selectRun(run)

        age = run.optimal_pb_loss_age
        d, p = run.statistics_by_pb_loss_age[age]
        self.optimalLeadLossAge.setValue(age)
        self.pValue.setValue(p)
        self.dValue.setValue(d)

    def _resizeTable(self):
        self.dataTable.resizeColumnsToContents()
        self.dataTable.resizeRowsToContents()
        self.dataTable.viewport().update()

    ############
    ## Events ##
    ############

    def _onProcessingCleared(self):
        self._showNoDataPanel()

    def _onMonteCarloRunAdded(self):
        total = len(self.sample.monteCarloRuns)
        if total == 1:
            self._showDataPanel()
            self.selectedRunInput.setValue(1)
            self._onSelectedMonteCarloRunChanged()

        self.selectedRunInput.setMaximum(total)
        self.maximumRunLabel.setText("/" + str(total))

    def _onSelectedMonteCarloRunChanged(self):
        index = self.selectedRunInput.value() - 1
        run = self.sample.monteCarloRuns[index]
        self._selectRun(run)