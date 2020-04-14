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

    def addRimAgeStats(self, rimAge, discordantAges, statistic):
        self.statistics[rimAge] = statistic
        self.reconstructedAges[rimAge] = discordantAges

        if len(self.statistics) % 5 == 0:
            self.signals.statisticsUpdated.emit(self.statistics, ())

    def getAgeRange(self):
        concordantAges = [row.concordantAge for row in self.rows if row.concordant]
        recAges = [recAge for ages in self.reconstructedAges.values() for recAge in ages]
        discordantAges = [recAge.values[0] for recAge in recAges if recAge]
        allAges = concordantAges + discordantAges
        return min(allAges), max(allAges)

    def getNearestSampledAge(self, requestedRimAge):
        if not self.statistics:
            return None, []

        if requestedRimAge is not None:
            actualRimAge = min(self.statistics, key=lambda a: abs(a-requestedRimAge))
        else:
            actualRimAge = max(self.statistics, key=lambda a: self.statistics[a])

        return actualRimAge, self.reconstructedAges[actualRimAge]