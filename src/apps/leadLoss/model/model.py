from apps.leadLoss.model.row import Row
from apps.leadLoss.process import processing

from apps.abstract.model.model import AbstractModel
from apps.leadLoss.model.settings.calculation import LeadLossCalculationSettings
from apps.type import ApplicationType


class LeadLossModel(AbstractModel):

    #################
    ## Constructor ##
    #################

    def __init__(self, signals):
        super().__init__(ApplicationType.LEAD_LOSS, signals)

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
        """
        self.statistics = {}
        self.reconstructedAges = {}
        for rimAge, reconstructedAges, statistic in output[0]:
            self.statistics[rimAge] = statistic
            self.reconstructedAges[rimAge] = reconstructedAges

        concordantAges = [row.calculatedCells[1].value for row in self.rows if row.concordant]
        discordantAges = [row.calculatedCells[1].value for row in self.rows if not row.concordant]

        self.view.onNewlyClassifiedPoints(self.rows, concordantAges, discordantAges)
        self.view.onNewStatistics(self.statistics)
        """

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

