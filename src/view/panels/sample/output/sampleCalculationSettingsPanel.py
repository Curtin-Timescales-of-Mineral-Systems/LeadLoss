from PyQt5.QtWidgets import QWidget, QFormLayout, QLabel, QGroupBox


class SampleCalculationSettingsPanel(QGroupBox):

    def __init__(self, sample):
        super().__init__(title="Settings")
        self.sample = sample

        self.discordance = QLabel()
        self.ageSamples = QLabel()
        self.dissimilarityTest = QLabel()
        self.monteCarloRuns = QLabel()

        layout = QFormLayout()
        layout.addRow("Discordance:", self.discordance)
        layout.addRow("Sample times:", self.ageSamples)
        layout.addRow("Dissimilarity:", self.dissimilarityTest)
        layout.addRow("Monte Carlo runs:", self.monteCarloRuns)
        self.setLayout(layout)

        sample.signals.concordancyCalculated.connect(self._onConcordanceCalculated)
        sample.signals.processingCleared.connect(self._onProcessingCleared)

    #############
    ## Actions ##
    #############

    def _update(self, settings):
        ageSamples = str(settings.minimumRimAge / (10 ** 6)) + "-" + \
                     str(settings.maximumRimAge / (10 ** 6)) + "[" + \
                     str(settings.rimAgesSampled) + "]";

        self.discordance.setText(settings.discordanceClassificationMethod.value)
        self.ageSamples.setText(ageSamples)
        self.dissimilarityTest.setText(str(settings.dissimilarityTest.value))
        self.monteCarloRuns.setText(str(settings.monteCarloRuns))

    def _clear(self):
        self.discordance.setText("")
        self.ageSamples.setText("")
        self.dissimilarityTest.setText("")
        self.monteCarloRuns.setText("")

    ############
    ## Events ##
    ############

    def _onProcessingCleared(self):
        self._clear()

    def _onConcordanceCalculated(self):
        self._update(self.sample.calculationSettings)