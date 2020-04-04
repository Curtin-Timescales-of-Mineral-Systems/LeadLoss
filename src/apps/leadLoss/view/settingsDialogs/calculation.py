from PyQt5.QtWidgets import QVBoxLayout, QGroupBox, QFormLayout, QLabel, QWidget

from apps.leadLoss.process.dissimilarityTests import DissimilarityTest
from utils import stringUtils
from apps.abstract.view.settingsDialogs.calculation import AbstractCalculationSettingsDialog
from apps.leadLoss.model.settings.calculation import LeadLossCalculationSettings
from apps.type import ApplicationType, SettingsType
from utils.ui.numericInput import PercentageInput, AgeInput, IntInput
from utils.ui.radioButtons import RadioButtons, IntRadioButtonGroup, EnumRadioButtonGroup


class LeadLossCalculationSettingsDialog(AbstractCalculationSettingsDialog):

    KEY = (ApplicationType.LEAD_LOSS, SettingsType.CALCULATION)

    def __init__(self, defaultSettings, *args, **kwargs):
        super().__init__(defaultSettings, *args, **kwargs)

        self._onDiscordanceTypeChanged()
        self._alignLabels()

    def initMainSettings(self):
        layout = QVBoxLayout()
        layout.addWidget(self._initDiscordanceSettings())
        layout.addWidget(self._initSamplingSettings())
        layout.addWidget(self._initDistributionComparisonSettings())

        widget = QWidget()
        widget.setLayout(layout)
        return widget


    def _initDiscordanceSettings(self):
        defaults = self.defaultSettings

        self.discordanceTypeRB = RadioButtons(stringUtils.DISCORDANCE_OPTIONS, self._validate, defaults.discordanceType)
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
        layout.addRow(QLabel("Minimum rim age"), self.minimumRimAgeInput)
        layout.addRow("Maximum rim age", self.maximumRimAgeInput)
        layout.addRow("Number of samples", self.rimAgesSampledInput)
        self._registerFormLayoutForAlignment(layout)

        box = QGroupBox("Rim age sampling")
        box.setLayout(layout)
        return box

    def _initDistributionComparisonSettings(self):
        defaults = self.defaultSettings

        self.dissimilarityTestRB = EnumRadioButtonGroup(DissimilarityTest, self._validate, defaults.dissimilarityTest, rows=None, cols=1)

        layout = QFormLayout()
        layout.addRow(QLabel("Dissimilarity test"), self.dissimilarityTestRB)
        self._registerFormLayoutForAlignment(layout)

        box = QGroupBox("Distribution comparison")
        box.setLayout(layout)
        return box

    ############
    ## Events ##
    ############

    def _onDiscordanceTypeChanged(self):
        type = self.discordanceTypeRB.selection()
        percentages = type == "Percentages"
        self.discordancePercentageCutoffLabel.setVisible(percentages)
        self.discordancePercentageCutoffLE.setVisible(percentages)
        self.discordanceEllipseSigmasLabel.setVisible(not percentages)
        self.discordanceEllipseSigmasRB.setVisible(not percentages)

    #####################
    ## Create settings ##
    #####################

    def _createSettings(self):
        settings = LeadLossCalculationSettings()

        settings.discordanceType = self.discordanceTypeRB.selection()
        if settings.discordanceType == "Percentages":
            settings.discordancePercentageCutoff = self.discordancePercentageCutoffLE.value()
        else:
            settings.discordanceEllipseSigmas = self.discordanceEllipseSigmasRB.selection()

        settings.minimumRimAge = self.minimumRimAgeInput.value()
        settings.maximumRimAge = self.maximumRimAgeInput.value()
        settings.rimAgesSampled = self.rimAgesSampledInput.value()

        settings.dissimilarityTestRB = self.dissimilarityTestRB.selection()

        return settings