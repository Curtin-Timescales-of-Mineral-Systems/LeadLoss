from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QGroupBox, QWidget, QHBoxLayout, \
    QFormLayout, QLabel, QSpinBox, QSplitter

from utils.ui import uiUtils
from utils.ui.numericInput import AgeInput, FloatInput, IntInput
from view.figures.sampleMonteCarloFigure import SampleMonteCarloFigure


class AgeStatisticPanel(QGroupBox):

    def __init__(self, title):
        super().__init__(title)

        self.leadLossAge = AgeInput(defaultValue=None, sf=5)
        self.leadLossAge.setReadOnly(True)

        self.dValue = FloatInput(defaultValue=None, sf=5)
        self.dValue.setReadOnly(True)

        self.pValue = FloatInput(defaultValue=None, sf=5)
        self.pValue.setReadOnly(True)

        self.invalidPoints = IntInput(defaultValue=None)
        self.invalidPoints.setReadOnly(True)

        self.score = FloatInput(defaultValue=None, sf=5)
        self.score.setReadOnly(True)

        layout = QFormLayout()
        layout.addRow("Pb-loss age", self.leadLossAge)
        layout.addRow("D value (KS test)", self.dValue)
        layout.addRow("p value (KS test)", self.pValue)
        layout.addRow("Number of invalid ages", self.invalidPoints)
        layout.addRow("Score: ", self.score)
        self.setLayout(layout)

    def update(self, age, statistic):
        self.leadLossAge.setValue(age)
        self.pValue.setValue(statistic.test_statistics[1])
        self.dValue.setValue(statistic.test_statistics[0])
        self.invalidPoints.setValue(statistic.number_of_invalid_ages)
        self.score.setValue(statistic.score)

    def clear(self):
        self.leadLossAge.setValue(None)
        self.pValue.setValue(None)
        self.dValue.setValue(None)
        self.invalidPoints.setValue(None)
        self.score.setValue(None)

class SampleOutputMonteCarloPanel(QWidget):

    def __init__(self, controller, sample):
        super().__init__()
        self.controller = controller
        self.sample = sample

        self.dataTable = None
        self.figure = None
        self.currentRun = None

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
        self.splitter = QSplitter()
        self.splitter.addWidget(self._createDataLHS())
        self.splitter.addWidget(self._createDataRHS())
        return self.splitter

    def _createRunSpinner(self):
        self.selectedRunInput = QSpinBox()
        self.selectedRunInput.setMinimum(1)
        self.selectedRunInput.setMaximum(1)
        self.selectedRunInput.valueChanged.connect(self._onSelectedMonteCarloRunChanged)

        self.maximumRunLabel = QLabel("/")

        runLayout = QHBoxLayout()
        runLayout.addWidget(self.selectedRunInput)
        runLayout.addWidget(self.maximumRunLabel)

        widget = QWidget()
        widget.setLayout(runLayout)
        widget.setContentsMargins(0,0,0,0)
        return widget

    def _createDataLHS(self):
        runSpinner = self._createRunSpinner()
        self.optimalStatistic = AgeStatisticPanel("Optimal age")
        self.selectedStatistic = AgeStatisticPanel("Selected age")

        layout = QFormLayout()
        layout.addRow("Monte Carlo run", runSpinner)
        layout.addRow(self.optimalStatistic)
        layout.addRow(self.selectedStatistic)

        widget = QWidget()
        widget.setLayout(layout)
        return widget

    def _createDataRHS(self):
        self.figure = SampleMonteCarloFigure(self.sample, self)
        return self.figure

    #############
    ## Actions ##
    #############

    def _showNoDataPanel(self):
        print("Showing no data panel")
        uiUtils.clearChildren(self.layout)
        self.layout.addWidget(self.noDataWidget)

    def _showDataPanel(self):
        uiUtils.clearChildren(self.layout)
        self.layout.addWidget(self.dataWidget)

    def _selectRun(self, run):
        self.currentRun = run
        self.figure.selectRun(run)
        self.optimalStatistic.update(run.optimal_pb_loss_age, run.optimal_statistic)

    def _resizeTable(self):
        self.dataTable.resizeColumnsToContents()
        self.dataTable.resizeRowsToContents()
        self.dataTable.viewport().update()

    ###################
    ## Age selection ##
    ###################

    def selectAge(self, age):
        if self.currentRun is None:
            return

        selectedAge = self.sample.calculationSettings.getNearestSampledAge(age)
        self.selectedStatistic.update(selectedAge, self.currentRun.statistics_by_pb_loss_age[selectedAge])
        self.figure.selectAge(age)

    def deselectAge(self):
        self.figure.deselectAge()
        self.selectedStatistic.clear()

    ############
    ## Events ##
    ############

    def _onProcessingCleared(self):
        self._showNoDataPanel()

    def _onMonteCarloRunAdded(self):
        if self.sample.hasNoOptimalAge:
            self.noDataWidget.setText(f"Sample {self.sample.name} has no discordant or concordant spots so no data can be generated.")
            self._showNoDataPanel()
        else:
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