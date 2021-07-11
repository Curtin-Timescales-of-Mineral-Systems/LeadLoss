from PyQt5.QtWidgets import QVBoxLayout, QGroupBox, QFormLayout, QLabel, QWidget, QCheckBox

from process.dissimilarityTests import DissimilarityTest
from model.settings.type import SettingsType
from src.model.settings.calculation import DiscordanceClassificationMethod
from utils import stringUtils
from model.settings.calculation import LeadLossCalculationSettings
from utils.ui.numericInput import PercentageInput, AgeInput, IntInput
from utils.ui.radioButtons import IntRadioButtonGroup, EnumRadioButtonGroup
from view.dialogs.settings.abstract import AbstractSettingsDialog


class LeadLossCalculationSettingsDialog(AbstractSettingsDialog):

    KEY = SettingsType.CALCULATION

    def __init__(self, samples, defaultSettings, *args, **kwargs):
        self.pointWithNoErrors = any(spot.uPbError == 0.0 and spot.pbPbError == 0.0 for sample in samples for spot in sample.validSpots)
        super().__init__(defaultSettings, *args, **kwargs)
        self.setWindowTitle("Calculation settings")
        self._onDiscordanceTypeChanged()
        self._alignLabels()

    def initMainSettings(self):
        layout = QVBoxLayout()
        layout.addWidget(self._initDiscordanceSettings())
        layout.addWidget(self._initSamplingSettings())
        layout.addWidget(self._initDistributionComparisonSettings())
        layout.addWidget(self._initMonteCarloSettings())

        widget = QWidget()
        widget.setLayout(layout)
        return widget


    def _initDiscordanceSettings(self):
        defaults = self.defaultSettings

        self.discordanceTypeRB = EnumRadioButtonGroup(DiscordanceClassificationMethod, self._validate, defaults.discordanceClassificationMethod)
        self.discordanceTypeRB.group.buttonClicked.connect(self._onDiscordanceTypeChanged)

        self.discordancePercentageCutoffLE = PercentageInput(
            defaultValue=defaults.discordancePercentageCutoff,
            validation=self._validate
        )
        self.discordancePercentageCutoffLabel = QLabel("Percentage cutoff")

        self.discordanceEllipseSigmasRB = IntRadioButtonGroup(stringUtils.ERROR_SIGMA_OPTIONS, self._validate, defaults.discordanceEllipseSigmas)
        self.discordanceEllipseSigmasLabel = QLabel("Ellipse sigmas")

        self.discordanceLayout = QFormLayout()
        self.discordanceLayout.addRow("Classify using", self.discordanceTypeRB)
        self.discordanceLayout.addRow(self.discordancePercentageCutoffLabel, self.discordancePercentageCutoffLE)
        self.discordanceLayout.addRow(self.discordanceEllipseSigmasLabel, self.discordanceEllipseSigmasRB)
        self._registerFormLayoutForAlignment(self.discordanceLayout)

        box = QGroupBox("Discordance")
        box.setLayout(self.discordanceLayout)
        return box

    def _initSamplingSettings(self):
        defaults = self.defaultSettings

        self.minimumRimAgeInput = AgeInput(validation=self._validate, defaultValue=defaults.minimumRimAge)
        self.maximumRimAgeInput = AgeInput(validation=self._validate, defaultValue=defaults.maximumRimAge)
        self.rimAgesSampledInput = IntInput(validation=self._validate, defaultValue=defaults.rimAgesSampled)

        layout = QFormLayout()
        layout.addRow(QLabel("Minimum"), self.minimumRimAgeInput)
        layout.addRow("Maximum", self.maximumRimAgeInput)
        layout.addRow("Number of samples", self.rimAgesSampledInput)
        self._registerFormLayoutForAlignment(layout)

        box = QGroupBox("Time of radiogenic-Pb loss")
        box.setLayout(layout)
        return box

    def _initDistributionComparisonSettings(self):
        defaults = self.defaultSettings

        self.dissimilarityTestRB = EnumRadioButtonGroup(DissimilarityTest, self._validate, defaults.dissimilarityTest, rows=None, cols=1)

        self.penaliseInvalidAgesCB = QCheckBox(self)
        self.penaliseInvalidAgesCB.setChecked(defaults.penaliseInvalidAges)
        self.penaliseInvalidAgesCB.stateChanged.connect(self._validate)

        layout = QFormLayout()
        layout.addRow(QLabel("Dissimilarity test"), self.dissimilarityTestRB)
        layout.addRow(QLabel("Penalise invalid ages"), self.penaliseInvalidAgesCB)
        self._registerFormLayoutForAlignment(layout)

        box = QGroupBox("Distribution comparison")
        box.setLayout(layout)
        return box

    def _initMonteCarloSettings(self):
        defaults = self.defaultSettings

        self.monteCarloRunsInput = IntInput(defaults.monteCarloRuns, self._validate)

        layout = QFormLayout()
        layout.addRow("Runs", self.monteCarloRunsInput)
        self._registerFormLayoutForAlignment(layout)

        box = QGroupBox("Monte Carlo sampling")
        box.setLayout(layout)
        return box

    ############
    ## Events ##
    ############

    def _onDiscordanceTypeChanged(self):
        type = self.discordanceTypeRB.selection()
        percentages = type == DiscordanceClassificationMethod.PERCENTAGE
        self.discordancePercentageCutoffLabel.setVisible(percentages)
        self.discordancePercentageCutoffLE.setVisible(percentages)
        self.discordanceEllipseSigmasLabel.setVisible(not percentages)
        self.discordanceEllipseSigmasRB.setVisible(not percentages)

    #####################
    ## Create settings ##
    #####################

    def getWarning(self, settings):
        if self.pointWithNoErrors and settings.discordanceClassificationMethod == DiscordanceClassificationMethod.ERROR_ELLIPSE:
            return "Warning: there exist points with no associated errors in the data set. " \
                   "Such points will never be classified as concordant using the error ellipse method " \
                   "unless they lie exactly on the concordia curve."

    def _createSettings(self):
        settings = LeadLossCalculationSettings()

        settings.discordanceClassificationMethod = self.discordanceTypeRB.selection()
        if settings.discordanceClassificationMethod == DiscordanceClassificationMethod.PERCENTAGE:
            settings.discordancePercentageCutoff = self.discordancePercentageCutoffLE.value()
        else:
            settings.discordanceEllipseSigmas = self.discordanceEllipseSigmasRB.selection()

        settings.minimumRimAge = self.minimumRimAgeInput.value()
        settings.maximumRimAge = self.maximumRimAgeInput.value()
        settings.rimAgesSampled = self.rimAgesSampledInput.value()

        settings.dissimilarityTestRB = self.dissimilarityTestRB.selection()
        settings.penaliseInvalidAges = self.penaliseInvalidAgesCB.isChecked()

        settings.monteCarloRuns = self.monteCarloRunsInput.value()

        return settings