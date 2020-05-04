import time

from model.row import Row
from model.settings.type import SettingsType
from process import processing

from model.settings.calculation import LeadLossCalculationSettings
from utils.settings import Settings


class LeadLossModel:

    UPDATE_INTERVAL = 0.5

    def __init__(self, signals):
        self.signals = signals

        self.headers = []
        self.rows = []
        self.concordantRows = []
        self.discordantRows = []

        self.pValuesByAge = {}
        self.dValuesByAge = {}
        self.reconstructedAges = {}

        self.lastUpdateTime = 0

    ################
    ## Input data ##
    ################

    def loadInputData(self, inputFile, importSettings, rawHeaders, rawRows):
        self.headers = rawHeaders
        self.rows = [Row(row, importSettings) for row in rawRows]

        importHeaders = importSettings.getHeaders()
        calculationHeaders = LeadLossCalculationSettings.getDefaultHeaders()
        headers = importHeaders + calculationHeaders

        self.signals.inputDataLoaded.emit(inputFile, headers, self.rows)
        self.signals.taskComplete.emit(True, "Successfully imported CSV file")

    def clearInputData(self):
        self.headers = []
        self.rows = []
        self.concordantRows = []
        self.discordantRows = []

        self.pValuesByAge = {}
        self.dValuesByAge = {}
        self.reconstructedAges = {}

        self.signals.inputDataCleared.emit()

    #################
    ## Calculation ##
    #################

    def resetCalculation(self):
        self.optimalAge = None
        self.pValuesByAge = {}
        self.dValuesByAge = {}

        for row in self.rows:
            row.resetCalculatedCells()
        self.signals.processingCleared.emit()

        importSettings = Settings.get(SettingsType.IMPORT)
        calculationSettings = Settings.get(SettingsType.CALCULATION)
        headers = importSettings.getHeaders() + calculationSettings.getHeaders()
        self.signals.headersUpdated.emit(headers)
        self.signals.allRowsUpdated.emit(self.rows)

        self.lastUpdateTime = time.time()

    def getProcessingFunction(self):
        return processing.process

    def getProcessingData(self):
        return self.rows

    def updateConcordance(self, i, discordance, concordantAge):
        row = self.rows[i]
        row.setConcordantAge(discordance, concordantAge)
        self.signals.rowUpdated.emit(i, row)

    def updateRow(self, i, row):
        self.rows[i] = row

    def addRimAgeStats(self, rimAge, discordantAges, dValue, pValue):
        self.dValuesByAge[rimAge] = dValue
        self.pValuesByAge[rimAge] = pValue
        self.reconstructedAges[rimAge] = discordantAges

        self.signals.statisticUpdated.emit(len(self.dValuesByAge)-1, dValue, pValue)
        now = time.time()
        if now - self.lastUpdateTime > self.UPDATE_INTERVAL:
            self.signals.allStatisticsUpdated.emit(self.dValuesByAge)
            self.lastUpdateTime = now

    def setOptimalAge(self, optimalAge):
        self.optimalAge = optimalAge

    #############
    ## Getters ##
    #############

    def getAgeRange(self):
        concordantAges = [row.concordantAge for row in self.rows if row.concordant]
        recAges = [recAge for ages in self.reconstructedAges.values() for recAge in ages]
        discordantAges = [recAge.values[0] for recAge in recAges if recAge]
        allAges = concordantAges + discordantAges
        return min(allAges), max(allAges)

    def getNearestSampledAge(self, requestedAge):
        if not self.dValuesByAge:
            return None, []

        if requestedAge is not None:
            actualAge = min(self.dValuesByAge, key=lambda a: abs(a-requestedAge))
        else:
            actualAge = self.optimalAge

        return actualAge, self.dValuesByAge[actualAge], self.pValuesByAge[actualAge], self.reconstructedAges[actualAge]