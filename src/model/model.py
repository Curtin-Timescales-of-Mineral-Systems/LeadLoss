from model.row import Row
from model.settings.type import SettingsType
from process import processing

from model.settings.calculation import LeadLossCalculationSettings
from utils.settings import Settings


class LeadLossModel:

    def __init__(self, signals):
        self.signals = signals

        self.headers = []
        self.rows = []
        self.concordantRows = []
        self.discordantRows = []

        self.statistics = {}
        self.reconstructedAges = {}

    def loadRawData(self, importSettings, rawHeaders, rawRows):
        self.headers = rawHeaders
        self.rows = [Row(row, importSettings) for row in rawRows]

        importHeaders = importSettings.getHeaders()
        calculationHeaders = LeadLossCalculationSettings.getDefaultHeaders()
        headers = importHeaders + calculationHeaders

        self.signals.headersUpdated.emit(headers)
        self.signals.allRowsUpdated.emit(self.rows)
        self.signals.taskComplete.emit(True, "Successfully imported CSV file")

    def updateRow(self, i, row):
        self.rows[i] = row

    def resetCalculations(self):
        importSettings = Settings.get(SettingsType.IMPORT)
        calculationSettings = Settings.get(SettingsType.CALCULATION)
        headers = importSettings.getHeaders() + calculationSettings.getHeaders()

        for row in self.rows:
            row.resetCalculatedCells()

        self.signals.headersUpdated.emit(headers)
        self.signals.allRowsUpdated.emit(self.rows)

    def getProcessingFunction(self):
        return processing.process

    def getProcessingData(self):
        return self.rows

    def updateConcordance(self, i, discordance, concordantAge):
        row = self.rows[i]
        row.setConcordantAge(discordance, concordantAge)
        self.signals.rowUpdated.emit(i, row)

    def addProcessingOutput(self, output):
        pass

    def selectAgeToCompare(self, targetRimAge):
        if not self.statistics:
            return

        if targetRimAge is not None:
            chosenRimAge = min(self.statistics, key=lambda a: abs(a-targetRimAge))
        else:
            chosenRimAge = max(self.statistics, key=lambda a: self.statistics[a])

        agesToCompare = []
        for reconstructedAge in self.reconstructedAges[chosenRimAge]:
            if reconstructedAge is not None:
                agesToCompare.append(reconstructedAge.values[0])
        #self.view.displayAgeComparison(chosenRimAge, agesToCompare)

