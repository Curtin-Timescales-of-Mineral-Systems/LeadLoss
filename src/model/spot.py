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
        # -------------------------
        # Sample name (unchanged)
        # -------------------------
        if settings.multipleSamples:
            self.sampleName = rawData[settings.getColumn(Column.SAMPLE_NAME)]
        else:
            self.sampleName = None

        # -------------------------
        # Determine import mode
        # -------------------------
        mode = getattr(settings, "ratioInputMode", "TW")
        mode_val = getattr(mode, "value", mode)  # Enum -> string
        mode_str = str(mode_val).lower()
        importing_wetherill = ("weth" in mode_str)

        self.importingWetherill = importing_wetherill  # optional debug flag

        # -------------------------
        # Choose which 4 columns we parse into displayStrings
        # -------------------------
        if importing_wetherill:
            parsed_cols = [
                Column.PB207_U235_VALUE,
                Column.PB207_U235_ERROR,
                Column.PB206_U238_VALUE,
                Column.PB206_U238_ERROR,
            ]
        else:
            parsed_cols = [
                Column.U_PB_VALUE,
                Column.U_PB_ERROR,
                Column.PB_PB_VALUE,
                Column.PB_PB_ERROR,
            ]

        values = {}
        self.displayStrings = []
        self.invalidColumns = []

        for i, col in enumerate(parsed_cols):
            string = rawData[settings.getColumn(col)]
            try:
                value = float(string)
                string = str(stringUtils.round_to_sf(value))
            except:
                value = None
                self.invalidColumns.append(i)
            values[col] = value
            self.displayStrings.append(string)

        # -------------------------
        # Initialise ALL fields (so other code never AttributeErrors)
        # -------------------------
        self.uPbValue = None
        self.uPbError = None
        self.uPbStDev = None

        self.pbPbValue = None
        self.pbPbError = None
        self.pbPbStDev = None

        self.pb207U235Value = None
        self.pb207U235Error = None
        self.pb207U235StDev = None

        self.pb206U238Value = None
        self.pb206U238Error = None
        self.pb206U238StDev = None

        # -------------------------
        # Validity: "did the imported 4 columns parse?"
        # -------------------------
        self.valid = not self.invalidColumns

        # Constant used for transforms
        U = getattr(calculations, "U238U235_RATIO", 137.818)

        # -------------------------
        # If importing TW: set TW from CSV, compute TW 1σ, derive Wetherill
        # -------------------------
        if not importing_wetherill:
            self.uPbValue = values.get(Column.U_PB_VALUE)
            self.uPbError = values.get(Column.U_PB_ERROR)
            self.pbPbValue = values.get(Column.PB_PB_VALUE)
            self.pbPbError = values.get(Column.PB_PB_ERROR)

            if self.valid:
                self.uPbStDev = calculations.to1StdDev(
                    self.uPbValue, self.uPbError, settings.uPbErrorType, settings.uPbErrorSigmas
                )
                self.pbPbStDev = calculations.to1StdDev(
                    self.pbPbValue, self.pbPbError, settings.pbPbErrorType, settings.pbPbErrorSigmas
                )

            # Derive Wetherill ratios from TW if possible
            u = self.uPbValue
            v = self.pbPbValue
            if (u is not None) and (v is not None) and (u > 0) and math.isfinite(u) and math.isfinite(v):
                # y = 206/238 = 1/u
                y = 1.0 / u
                # x = 207/235 = v * (206/238) * (238/235) = v * U / u
                x = v * U / u

                self.pb206U238Value = y
                self.pb207U235Value = x

                # Propagate 1σ (no covariance)
                if (self.uPbStDev is not None) and (self.pbPbStDev is not None) and math.isfinite(self.uPbStDev) and math.isfinite(self.pbPbStDev):
                    su = abs(self.uPbStDev)
                    sv = abs(self.pbPbStDev)

                    # y = 1/u  => dy/du = -1/u^2
                    sy = abs(su / (u * u))

                    # x = v*U/u
                    dx_dv = U / u
                    dx_du = -v * U / (u * u)
                    sx = math.sqrt((dx_dv * sv) ** 2 + (dx_du * su) ** 2)

                    self.pb206U238StDev = sy
                    self.pb207U235StDev = sx

                    # Also populate "error" fields (for legacy code that still reads *Error)
                    self.pb206U238Error = calculations.from1StdDev(
                        y, sy, settings.pb206U238ErrorType, settings.pb206U238ErrorSigmas
                    )
                    self.pb207U235Error = calculations.from1StdDev(
                        x, sx, settings.pb207U235ErrorType, settings.pb207U235ErrorSigmas
                    )

        # -------------------------
        # If importing Wetherill: set Wetherill from CSV, compute 1σ, derive TW
        # -------------------------
        else:
            self.pb207U235Value = values.get(Column.PB207_U235_VALUE)
            self.pb207U235Error = values.get(Column.PB207_U235_ERROR)
            self.pb206U238Value = values.get(Column.PB206_U238_VALUE)
            self.pb206U238Error = values.get(Column.PB206_U238_ERROR)

            if self.valid:
                self.pb207U235StDev = calculations.to1StdDev(
                    self.pb207U235Value, self.pb207U235Error, settings.pb207U235ErrorType, settings.pb207U235ErrorSigmas
                )
                self.pb206U238StDev = calculations.to1StdDev(
                    self.pb206U238Value, self.pb206U238Error, settings.pb206U238ErrorType, settings.pb206U238ErrorSigmas
                )

            x = self.pb207U235Value
            y = self.pb206U238Value
            if (x is not None) and (y is not None) and (y > 0) and math.isfinite(x) and math.isfinite(y) and (U > 0):
                # u = 238/206 = 1/y
                u = 1.0 / y
                # v = 207/206 = x / (U*y)
                v = x / (U * y)

                self.uPbValue = u
                self.pbPbValue = v

                # Propagate 1σ (no covariance)
                if (self.pb207U235StDev is not None) and (self.pb206U238StDev is not None) and math.isfinite(self.pb207U235StDev) and math.isfinite(self.pb206U238StDev):
                    sx = abs(self.pb207U235StDev)
                    sy = abs(self.pb206U238StDev)

                    # u = 1/y => du/dy = -1/y^2
                    su = abs(sy / (y * y))

                    # v = x/(U*y)
                    dv_dx = 1.0 / (U * y)
                    dv_dy = -x / (U * y * y)
                    sv = math.sqrt((dv_dx * sx) ** 2 + (dv_dy * sy) ** 2)

                    self.uPbStDev = su
                    self.pbPbStDev = sv

                    # Populate TW error fields from derived 1σ (for legacy code that reads *Error)
                    self.uPbError = calculations.from1StdDev(
                        u, su, settings.uPbErrorType, settings.uPbErrorSigmas
                    )
                    self.pbPbError = calculations.from1StdDev(
                        v, sv, settings.pbPbErrorType, settings.pbPbErrorSigmas
                    )

        # -------------------------
        # Processing flags (unchanged)
        # -------------------------
        self.processed = False
        self.reverseDiscordant = False

    def clear(self):
        self.processed = False
        self.concordant = None
        self.discordance = None
        self.reverseDiscordant = False

    def updateConcordance(self, concordant, discordance, reverse=False):
        self.processed = True
        self.concordant = None if concordant is None else bool(concordant)
        self.discordance = discordance
        self.reverseDiscordant = bool(reverse)
        if discordance is not None:
            self.displayStrings.append(stringUtils.round_to_sf(discordance * 100))
