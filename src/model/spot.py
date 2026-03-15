import math

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

        # Preserve imported display state so repeated processing runs do not
        # keep appending discordance columns.
        self._baseDisplayStrings = list(self.displayStrings)
        self._baseInvalidColumns = list(self.invalidColumns)

        self.processed = False
        self.reverseDiscordant = False
        self.cluster_id = None


    def clear(self):
        self.processed = False
        self.concordant = None
        self.discordance = None
        self.reverseDiscordant = False
        self.cluster_id = None
        self.displayStrings = list(self._baseDisplayStrings)
        self.invalidColumns = list(self._baseInvalidColumns)

    def updateConcordance(self, concordant, discordance, reverse=False):
        self.processed = True
        self.concordant = None if concordant is None else bool(concordant)
        self.discordance = discordance
        self.reverseDiscordant = bool(reverse)
        base_n = len(self._baseDisplayStrings)
        self.displayStrings = list(self.displayStrings[:base_n])
        if discordance is not None:
            try:
                discordance_pct = float(discordance) * 100.0
            except (TypeError, ValueError):
                discordance_pct = float("nan")
            if math.isfinite(discordance_pct):
                self.displayStrings.append(stringUtils.round_to_sf(discordance_pct))
            else:
                self.displayStrings.append("N/A")
