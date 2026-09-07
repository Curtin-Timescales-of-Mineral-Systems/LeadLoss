import math

from model.column import Column
from model.settings.ratio import ConcordiaRatioSpace, ratio_space_from_value
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
        settings.ensureCompatibility()
        if settings.multipleSamples:
            self.sampleName = rawData[settings.getColumn(Column.SAMPLE_NAME)]
        else:
            self.sampleName = None

        self.inputRatioSpace = settings.getInputRatioSpace()
        self.displayRatioSpace = settings.getDisplayRatioSpace()
        self.errorCorrelation = 0.0
        self._ratioValues = {}
        self._ratioStDevs = {}
        self.inputXStDev = None
        self.inputYStDev = None

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

        rhoColumn = settings.getRhoColumn()
        if rhoColumn is not None:
            try:
                rhoString = rawData[rhoColumn]
            except Exception:
                rhoString = ""
            try:
                self.errorCorrelation = float(rhoString)
                rhoDisplay = str(stringUtils.round_to_sf(self.errorCorrelation))
                if not -1.0 <= self.errorCorrelation <= 1.0:
                    self.invalidColumns.append(4)
            except Exception:
                self.errorCorrelation = 0.0
                rhoDisplay = rhoString
                self.invalidColumns.append(4)
            self.displayStrings.append(rhoDisplay)

        self.inputXValue = values[Column.U_PB_VALUE]
        self.inputXError = values[Column.U_PB_ERROR]
        self.inputYValue = values[Column.PB_PB_VALUE]
        self.inputYError = values[Column.PB_PB_ERROR]

        # Legacy aliases remain TW coordinates after successful parsing.
        self.uPbValue = self.inputXValue
        self.uPbError = self.inputXError
        self.pbPbValue = self.inputYValue
        self.pbPbError = self.inputYError
        self.uPbStDev = None
        self.pbPbStDev = None
        self.pb207u235Value = None
        self.pb207u235StDev = None
        self.pb206u238Value = None
        self.pb206u238StDev = None
        self.wetherillErrorCorrelation = 0.0
        self.twErrorCorrelation = 0.0

        self.valid = not self.invalidColumns
        if self.valid:
            self.inputXStDev = calculations.to1StdDev(
                self.inputXValue,
                self.inputXError,
                settings.uPbErrorType,
                settings.uPbErrorSigmas,
            )
            self.inputYStDev = calculations.to1StdDev(
                self.inputYValue,
                self.inputYError,
                settings.pbPbErrorType,
                settings.pbPbErrorSigmas,
            )
            self._cacheRatioSpaces()

        # Preserve imported display state so repeated processing runs do not
        # keep appending discordance columns.
        self._baseDisplayStrings = list(self.displayStrings)
        self._baseInvalidColumns = list(self.invalidColumns)

        self.processed = False
        self.reverseDiscordant = False

    @staticmethod
    def _ratioKey(ratioSpace):
        return ratio_space_from_value(ratioSpace).value

    def _markRatioConversionInvalid(self):
        for index in (0, 2):
            if index not in self.invalidColumns:
                self.invalidColumns.append(index)
        self.valid = False

    def _cacheRatioSpaces(self):
        source_key = self._ratioKey(self.inputRatioSpace)
        self._ratioValues[source_key] = (self.inputXValue, self.inputYValue)
        self._ratioStDevs[source_key] = (self.inputXStDev, self.inputYStDev, self.errorCorrelation)

        target = (
            ConcordiaRatioSpace.WETHERILL
            if self.inputRatioSpace == ConcordiaRatioSpace.TERA_WASSERBURG
            else ConcordiaRatioSpace.TERA_WASSERBURG
        )
        target_key = self._ratioKey(target)

        try:
            tx, ty = calculations.convert_ratio_xy(
                self.inputXValue,
                self.inputYValue,
                self.inputRatioSpace,
                target,
            )
            tsx, tsy, trho = calculations.convert_ratio_covariance(
                self.inputXValue,
                self.inputYValue,
                self.inputXStDev,
                self.inputYStDev,
                self.errorCorrelation,
                self.inputRatioSpace,
                target,
            )
        except Exception:
            self._markRatioConversionInvalid()
            return

        self._ratioValues[target_key] = (tx, ty)
        self._ratioStDevs[target_key] = (tsx, tsy, trho)

        tw_x, tw_y = self.getRatioValues(ConcordiaRatioSpace.TERA_WASSERBURG)
        tw_sx, tw_sy, tw_rho = self.getRatioStDevs(ConcordiaRatioSpace.TERA_WASSERBURG)
        weth_x, weth_y = self.getRatioValues(ConcordiaRatioSpace.WETHERILL)
        weth_sx, weth_sy, weth_rho = self.getRatioStDevs(ConcordiaRatioSpace.WETHERILL)

        self.uPbValue = tw_x
        self.pbPbValue = tw_y
        self.uPbStDev = tw_sx
        self.pbPbStDev = tw_sy
        self.twErrorCorrelation = tw_rho

        self.pb207u235Value = weth_x
        self.pb206u238Value = weth_y
        self.pb207u235StDev = weth_sx
        self.pb206u238StDev = weth_sy
        self.wetherillErrorCorrelation = weth_rho

    def getInputRatioValues(self):
        return self.inputXValue, self.inputYValue

    def getInputRatioStDevs(self):
        return self.inputXStDev, self.inputYStDev, self.errorCorrelation

    def getRatioValues(self, ratioSpace):
        return self._ratioValues.get(self._ratioKey(ratioSpace), (None, None))

    def getRatioStDevs(self, ratioSpace):
        return self._ratioStDevs.get(self._ratioKey(ratioSpace), (None, None, 0.0))

    def clear(self):
        self.processed = False
        self.concordant = None
        self.discordance = None
        self.reverseDiscordant = False
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
