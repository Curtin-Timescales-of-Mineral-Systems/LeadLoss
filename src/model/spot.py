from model.column import Column
from process import calculations

from utils import stringUtils


class Spot:

    @staticmethod
    def _getFloat(settings, values, column):
        stringRep = values[settings.getColumn(column)]
        try:
            return float(stringRep)
        except:
            return stringRep

    def __init__(self, rawData, settings):
        if settings.multipleSamples:
            self.sampleName = rawData[settings.getColumn(Column.SAMPLE_NAME)]
        else:
            self.sampleName = None

        values = {}
        self.displayStrings = []
        self.invalidColumns = []
        for i, col in enumerate([Column.U_PB_VALUE, Column.U_PB_ERROR, Column.PB_PB_VALUE, Column.PB_PB_ERROR]):
            string = rawData[settings.getColumn(col)]
            try:
                value = float(string)
                string = str(stringUtils.round_to_sf(value))
            except:
                value = None
                self.invalidColumns.append(i)
            values[col] = value
            self.displayStrings.append(string)

        self.uPbValue = values[Column.U_PB_VALUE]
        self.uPbError = values[Column.U_PB_ERROR]
        self.pbPbValue = values[Column.PB_PB_VALUE]
        self.pbPbError = values[Column.PB_PB_ERROR]

        self.valid = not self.invalidColumns
        if self.valid:
            self.uPbStDev = calculations.to1StdDev(self.uPbValue, self.uPbError, settings.uPbErrorType, settings.uPbErrorSigmas)
            self.pbPbStDev = calculations.to1StdDev(self.pbPbValue, self.pbPbError, settings.pbPbErrorType, settings.pbPbErrorSigmas)

        self.processed = False

    def updateConcordance(self, concordantAge, discordance):
        self.concordant = concordantAge is not None

        self.discordance = discordance
        if discordance:
            self.displayStrings.append(stringUtils.round_to_sf(discordance*100))

        if self.concordant:
            self.concordantAge = concordantAge
            self.displayStrings.append(stringUtils.round_to_sf(concordantAge/(10**6)))