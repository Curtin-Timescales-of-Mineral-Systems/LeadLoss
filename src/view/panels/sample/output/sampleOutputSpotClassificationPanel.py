from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout

from model.settings.calculation import DiscordanceClassificationMethod
from model.settings.type import SettingsType
from utils.settings import Settings
from utils.ui import spotTable, uiUtils
from utils.ui.icons import Icons


class SampleOutputSpotClassificationPanel(QWidget):

    def __init__(self, sample):
        super().__init__()
        self.sample = sample

        self.setLayout(QVBoxLayout())

        self.sample.signals.concordancyCalculated.connect(self._onConcordancyCalculated)

    ########
    ## UI ##
    ########

    def _createSpotsWidget(self):
        invalidWarningWidget = self._createInvalidWarningWidget()

        importSettings = Settings.get(SettingsType.IMPORT)
        calculationSettings = self.sample.calculationSettings
        importHeaders = importSettings.getHeaders()

        if calculationSettings.discordanceClassificationMethod.value == DiscordanceClassificationMethod.PERCENTAGE.value:
            baseHeaders = ["Discordance (%)"]
        else:
            baseHeaders = []

        concordantHeaders = importHeaders + baseHeaders + ["Age (Ma)"]
        concordantSpots = self.sample.concordantSpots()

        discordantHeaders = importHeaders + baseHeaders
        discordantSpots = self.sample.discordantSpots()

        layout = self.layout()
        layout.addWidget(QLabel("Concordant points"))
        layout.addWidget(spotTable.createSpotTable(concordantHeaders, concordantSpots))
        layout.addWidget(QLabel("Discordant points"))
        layout.addWidget(spotTable.createSpotTable(discordantHeaders, discordantSpots))
        if invalidWarningWidget:
            layout.addWidget(invalidWarningWidget)

    def _createInvalidWarningWidget(self):
        n = len(self.sample.invalidSpots)
        if n == 0:
            return None

        if n == 1:
            pointText = str(n) + " invalid point"
        else:
            pointText = str(n) + " invalid points"
        return uiUtils.createIconWithLabel(Icons.warning(), pointText + " excluded from analysis")

    ############
    ## Events ##
    ############

    def _onConcordancyCalculated(self):
        uiUtils.clearChildren(self.layout())
        self.layout().addWidget(self._createSpotsWidget())