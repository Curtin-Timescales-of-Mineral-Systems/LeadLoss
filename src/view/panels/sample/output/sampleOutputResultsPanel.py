from PyQt5.QtWidgets import QFormLayout, QGroupBox, QHBoxLayout, QLabel

from utils import config
from utils.ui.numericInput import FloatInput, AgeInput


class SampleOutputResultsPanel(QGroupBox):

    def __init__(self, sample):
        super().__init__("Result")
        self.sample = sample

        self.optimalAge = AgeInput(defaultValue=None)
        self.optimalAge.setReadOnly(True)

        self.optimalAgeLower = AgeInput(defaultValue=None)
        self.optimalAgeLower.setReadOnly(True)
        self.optimalAgeUpper = AgeInput(defaultValue=None)
        self.optimalAgeUpper.setReadOnly(True)
        boundsLayout = QHBoxLayout()
        boundsLayout.addWidget(self.optimalAgeLower)
        boundsLayout.addWidget(QLabel("-"))
        boundsLayout.addWidget(self.optimalAgeUpper)

        self.dValue = FloatInput(defaultValue=None, sf=config.DISPLAY_SF)
        self.dValue.setReadOnly(True)

        self.pValue = FloatInput(defaultValue=None, sf=config.DISPLAY_SF)
        self.pValue.setReadOnly(True)

        layout = QFormLayout()
        layout.addRow("Optimal Pb-loss age", self.optimalAge)
        layout.addRow("95% confidence interval", boundsLayout)
        layout.addRow("Mean D value", self.dValue)
        layout.addRow("Mean p value", self.pValue)
        self.setLayout(layout)

        sample.signals.processingCleared.connect(self._onProcessingCleared)
        sample.signals.optimalAgeCalculated.connect(self._onOptimalAgeCalculated)

    #############
    ## Actions ##
    #############

    def update(self):
        self.optimalAge.setValue(self.sample.optimalAge)
        self.optimalAgeLower.setValue(self.sample.optimalAgeLowerBound)
        self.optimalAgeUpper.setValue(self.sample.optimalAgeUpperBound)
        self.dValue.setValue(self.sample.optimalAgeDValue)
        self.pValue.setValue(self.sample.optimalAgePValue)

    def clear(self):
        self.optimalAge.setValue(None)
        self.optimalAgeLower.setValue(None)
        self.optimalAgeUpper.setValue(None)
        self.dValue.setValue(None)
        self.pValue.setValue(None)

    ############
    ## Events ##
    ############

    def _onProcessingCleared(self):
        self.clear()

    def _onOptimalAgeCalculated(self):
        self.update()