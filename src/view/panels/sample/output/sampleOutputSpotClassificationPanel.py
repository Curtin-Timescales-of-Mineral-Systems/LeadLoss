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

        baseHeaders = ["Discordance (%)"] if (
            calculationSettings.discordanceClassificationMethod.value
            == DiscordanceClassificationMethod.PERCENTAGE.value
        ) else []

        concordantHeaders = importHeaders + baseHeaders
        discordantHeaders = importHeaders + baseHeaders

        concordantSpots = self.sample.concordantSpots()
        discordantSpots = self.sample.discordantSpots()

        # IMPORTANT: build a new widget to return
        w = QWidget(self)
        v = QVBoxLayout(w)

        v.addWidget(QLabel("Concordant points"))
        v.addWidget(spotTable.createSpotTable(concordantHeaders, concordantSpots))

        v.addWidget(QLabel("Discordant points"))
        v.addWidget(spotTable.createSpotTable(discordantHeaders, discordantSpots))

        if invalidWarningWidget:
            v.addWidget(invalidWarningWidget)

        return w


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
        w = self._createSpotsWidget()
        if w is not None:
            self.layout().addWidget(w)
