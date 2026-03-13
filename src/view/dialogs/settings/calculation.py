from PyQt5.QtWidgets import QVBoxLayout, QGroupBox, QFormLayout, QLabel, QWidget, QCheckBox
from process.dissimilarityTests import DissimilarityTest
from model.settings.type import SettingsType
from model.settings.calculation import DiscordanceClassificationMethod
from utils import stringUtils
from model.settings.calculation import LeadLossCalculationSettings
from utils.ui.numericInput import PercentageInput, AgeInput, IntInput
from utils.ui.radioButtons import IntRadioButtonGroup, EnumRadioButtonGroup
from view.dialogs.settings.abstract import AbstractSettingsDialog

class LeadLossCalculationSettingsDialog(AbstractSettingsDialog):

    KEY = SettingsType.CALCULATION

    def __init__(self, samples, defaultSettings, *args, **kwargs):
        self.pointWithNoErrors = any(
            spot.uPbError == 0.0 and spot.pbPbError == 0.0
            for sample in samples for spot in sample.validSpots
        )
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
        layout.addWidget(self._initPeakSettings())
        widget = QWidget(); widget.setLayout(layout)
        return widget

    def _initDiscordanceSettings(self):
        defaults = self.defaultSettings
        self.discordanceTypeRB = EnumRadioButtonGroup(
            DiscordanceClassificationMethod, self._validate, defaults.discordanceClassificationMethod
        )
        self.discordanceTypeRB.group.buttonClicked.connect(self._onDiscordanceTypeChanged)

        self.discordancePercentageCutoffLE = PercentageInput(
            defaultValue=defaults.discordancePercentageCutoff, validation=self._validate
        )
        self.discordancePercentageCutoffLabel = QLabel("Percentage cutoff")

        self.discordanceEllipseSigmasRB = IntRadioButtonGroup(
            stringUtils.ERROR_SIGMA_OPTIONS, self._validate, defaults.discordanceEllipseSigmas
        )
        self.discordanceEllipseSigmasLabel = QLabel("Ellipse sigmas")

        form = QFormLayout()
        form.addRow("Classify using", self.discordanceTypeRB)
        form.addRow(self.discordancePercentageCutoffLabel, self.discordancePercentageCutoffLE)
        form.addRow(self.discordanceEllipseSigmasLabel, self.discordanceEllipseSigmasRB)
        self._registerFormLayoutForAlignment(form)

        box = QGroupBox("Discordance"); box.setLayout(form)
        return box

    def _initSamplingSettings(self):
        defaults = self.defaultSettings
        self.minimumRimAgeInput = AgeInput(validation=self._validate, defaultValue=defaults.minimumRimAge)
        self.maximumRimAgeInput = AgeInput(validation=self._validate, defaultValue=defaults.maximumRimAge)
        self.rimAgesSampledInput = IntInput(validation=self._validate, defaultValue=defaults.rimAgesSampled)

        form = QFormLayout()
        form.addRow(QLabel("Minimum"), self.minimumRimAgeInput)
        form.addRow("Maximum", self.maximumRimAgeInput)
        form.addRow("Number of samples", self.rimAgesSampledInput)
        self._registerFormLayoutForAlignment(form)

        box = QGroupBox("Time of radiogenic-Pb loss"); box.setLayout(form)
        return box

    def _initDistributionComparisonSettings(self):
        defaults = self.defaultSettings
        self.dissimilarityTestRB = EnumRadioButtonGroup(
            DissimilarityTest, self._validate, defaults.dissimilarityTest, rows=None, cols=1
        )
        self.penaliseInvalidAgesCB = QCheckBox(self)
        self.penaliseInvalidAgesCB.setChecked(defaults.penaliseInvalidAges)
        self.penaliseInvalidAgesCB.stateChanged.connect(self._validate)

        form = QFormLayout()
        form.addRow(QLabel("Dissimilarity test"), self.dissimilarityTestRB)
        form.addRow(QLabel("Penalise invalid ages"), self.penaliseInvalidAgesCB)
        self._registerFormLayoutForAlignment(form)

        box = QGroupBox("Distribution comparison"); box.setLayout(form)
        return box

    def _initMonteCarloSettings(self):
        defaults = self.defaultSettings
        self.monteCarloRunsInput = IntInput(defaults.monteCarloRuns, self._validate)

        form = QFormLayout()
        form.addRow("Runs", self.monteCarloRunsInput)
        self._registerFormLayoutForAlignment(form)

        box = QGroupBox("Monte Carlo sampling"); box.setLayout(form)
        return box

    def _initPeakSettings(self):
        d = self.defaultSettings
        box = QGroupBox("Peak extraction")
        v = QVBoxLayout()

        # Ensemble multi-peak finder
        self.enableEnsembleCB = QCheckBox("Ensemble catalogue", self)
        self.enableEnsembleCB.setChecked(bool(getattr(d, "enable_ensemble_peak_picking", True)))
        self.enableEnsembleCB.stateChanged.connect(self._validate)

        v.addWidget(self.enableEnsembleCB)
        v.addStretch(1)

        box.setLayout(v)
        return box

    # events / settings

    def _onDiscordanceTypeChanged(self):
        t = self.discordanceTypeRB.selection()
        perc = (t == DiscordanceClassificationMethod.PERCENTAGE)
        self.discordancePercentageCutoffLabel.setVisible(perc)
        self.discordancePercentageCutoffLE.setVisible(perc)
        self.discordanceEllipseSigmasLabel.setVisible(not perc)
        self.discordanceEllipseSigmasRB.setVisible(not perc)  # set True when not perc

    def getWarning(self, settings):
        if self.pointWithNoErrors and settings.discordanceClassificationMethod == DiscordanceClassificationMethod.ERROR_ELLIPSE:
            return ("Warning: there exist points with no associated errors in the data set. "
                    "Such points will never be classified as concordant using the error ellipse method "
                    "unless they lie exactly on the concordia curve.")

    def _createSettings(self):
        s = LeadLossCalculationSettings()
        s.discordanceClassificationMethod = self.discordanceTypeRB.selection()
        if s.discordanceClassificationMethod == DiscordanceClassificationMethod.PERCENTAGE:
            s.discordancePercentageCutoff = self.discordancePercentageCutoffLE.value()
        else:
            s.discordanceEllipseSigmas = self.discordanceEllipseSigmasRB.selection()

        s.minimumRimAge  = self.minimumRimAgeInput.value()
        s.maximumRimAge  = self.maximumRimAgeInput.value()
        s.rimAgesSampled = self.rimAgesSampledInput.value()

        s.dissimilarityTest   = self.dissimilarityTestRB.selection()
        s.penaliseInvalidAges = self.penaliseInvalidAgesCB.isChecked()
        s.monteCarloRuns      = self.monteCarloRunsInput.value()

        s.enable_ensemble_peak_picking = self.enableEnsembleCB.isChecked()

        # Fixed publication-safe defaults (hidden from GUI to reduce user burden)
        s.conservative_abstain_on_monotonic = True
        s.merge_nearby_peaks                = True

        return s
