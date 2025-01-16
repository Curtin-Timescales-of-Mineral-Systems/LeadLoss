
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QFormLayout, QSpinBox, QLabel, QSplitter
from utils.ui import uiUtils
from utils.ui.numericInput import AgeInput, FloatInput, IntInput
from view.figures.sampleMonteCarloWetherillFigure import SampleMonteCarloWetherillFigure

class AgeStatisticPanelWetherill(QWidget):
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

class SampleOutputMonteCarloWetherillPanel(QWidget):
    """
    A separate panel for Wetherill runs, just like your Tera–W panel.
    """
    def __init__(self, controller, sample):
        super().__init__()
        self.controller = controller
        self.sample = sample

        self.layout = QHBoxLayout()
        self.setLayout(self.layout)

        self.noDataWidget = uiUtils.createNoDataWidget(sample.name)
        self.dataWidget = self._createDataWidget()

        if not sample.monteCarloRuns:
            self._showNoDataPanel()
        else:
            self._showDataPanel()

        sample.signals.monteCarloRunAdded.connect(self._onMonteCarloRunAdded)

    def _showNoDataPanel(self):
        if not self.sample.hasOptimalAge:
            # Display an error message when there is no optimal age
            error_label = QLabel("There is no optimal age, so no data can be generated.")
            self.layout.addWidget(error_label)
        else:
            # Show the regular data widget
            self.layout.addWidget(self.dataWidget)

    def _onMonteCarloRunAdded(self):
        self._showNoDataPanel()

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
        self.figure = SampleMonteCarloWetherillFigure(self.sample, self.controller)
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
        self.currentRun = run
        self.figure.selectRun(run)
        self.optimalStatistic.update(run.optimal_pb_loss_age, run.optimal_statistic)

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

