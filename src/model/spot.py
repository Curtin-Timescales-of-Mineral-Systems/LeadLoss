from model.column import Column
from process import calculations
import math
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

        self.uPbValue = values[Column.U_PB_VALUE] #238U/206Pb
        self.uPbError = values[Column.U_PB_ERROR]
        self.pbPbValue = values[Column.PB_PB_VALUE] # 207Pb/206Pb
        self.pbPbError = values[Column.PB_PB_ERROR]

        self.valid = not self.invalidColumns
        
        if self.valid:
            # Convert user-supplied errors ti 1-sigma absolute for Tera-Wasserburg
            self.uPbStDev = calculations.to1StdDev(self.uPbValue, self.uPbError, settings.uPbErrorType, settings.uPbErrorSigmas)
            self.pbPbStDev = calculations.to1StdDev(self.pbPbValue, self.pbPbError, settings.pbPbErrorType, settings.pbPbErrorSigmas)

            # Compute Wetherill from Tera-Wasserburg
            # 206Pb/238U = 1/(238U/206Pb)
            if self.uPbValue is not None and self.uPbValue != 0:
                self.pb206U238Value = 1.0 / self.uPbValue
            # Propagate error using partial derivative  d(1/x)/dx = -1/x^2
                self.pb206U238Error = abs(1.0 / (self.uPbValue**2)) * self.uPbStDev
            else:
                self.pb206U238Value = None
                self.pb206U238Error = None

            # 207Pb/235U = (137.818) * (207Pb/206Pb) * (238U/206Pb)
            if (self.pbPbValue is not None) and (self.pb206U238Value is not None):
                self.pb207U235Value = 137.818 * self.pbPbValue * self.pb206U238Value

                # error propagation: z = c * x * y
                # var(z) ~ ( ∂z/∂x * σx )^2 + ( ∂z/∂y * σy )^2
                # where x=pbPbValue, y=pb206U238Value, c=137.818
                partial_wrt_pbPb = 137.818 * self.pb206U238Value
                partial_wrt_pb206U238 = 137.818 * self.pbPbValue
                var_207U235 = (
                    (partial_wrt_pbPb     * self.pbPbStDev       )**2 +
                    (partial_wrt_pb206U238 * self.pb206U238Error )**2
                )
                self.pb207U235Error = math.sqrt(var_207U235)
            else:
                self.pb207U235Value = None
                self.pb207U235Error = None            
        else:
            # If invalid, no ratio or errors
            self.uPbStDev = None
            self.pbPbStDev = None

            self.pb206U238Value = None
            self.pb206U238Error = None
            self.pb207U235Value = None
            self.pb207U235Error = None

        self.processed = False

    def clear(self):
        self.processed = False
        self.concordant = None
        self.discordance = None

    def updateConcordance(self, concordant, discordance):
        self.processed = True
        self.concordant = concordant
        self.discordance = discordance
        if discordance is not None:
            self.displayStrings.append(stringUtils.round_to_sf(discordance*100))