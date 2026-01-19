from model.column import Column
from process import calculations
from process.concordia_transforms import (
    tw_to_wetherill,
    wetherill_to_tw,
    propagate_tw_to_wetherill_uncertainty,
    propagate_wetherill_to_tw_uncertainty,
)

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

        # Determine which ratio space the CSV provides.
        mode = getattr(settings, "ratioInputMode", "TW")
        if hasattr(mode, "value"):
            mode_str = str(mode.value)
        else:
            mode_str = str(mode)
        mode_str = mode_str.lower()
        csv_is_wetherill = "weth" in mode_str

        values = {}
        self.displayStrings = []
        self.invalidColumns = []

        if csv_is_wetherill:
            displayed = [
                Column.PB207_U235_VALUE,
                Column.PB207_U235_ERROR,
                Column.PB206_U238_VALUE,
                Column.PB206_U238_ERROR,
            ]
        else:
            displayed = [
                Column.U_PB_VALUE,
                Column.U_PB_ERROR,
                Column.PB_PB_VALUE,
                Column.PB_PB_ERROR,
            ]

        for i, col in enumerate(displayed):
            string = rawData[settings.getColumn(col)]
            try:
                value = float(string)
                string = str(stringUtils.round_to_sf(value))
            except:
                value = None
                self.invalidColumns.append(i)
            values[col] = value
            self.displayStrings.append(string)

        # ----------------------------
        # Canonical attributes
        # ----------------------------
        # TW (used by the TW pipeline)
        self.uPbValue = None
        self.uPbError = None
        self.uPbStDev = None
        self.pbPbValue = None
        self.pbPbError = None
        self.pbPbStDev = None

        # Wetherill (used by Wetherill plot/pipeline)
        self.pb207U235Value = None
        self.pb207U235Error = None
        self.pb207U235StDev = None
        self.pb206U238Value = None
        self.pb206U238Error = None
        self.pb206U238StDev = None

        self.valid = not self.invalidColumns
        if self.valid:
            U = calculations.U238U235_RATIO

            if csv_is_wetherill:
                # --- imported in Wetherill ---
                self.pb207U235Value = values[Column.PB207_U235_VALUE]
                self.pb207U235Error = values[Column.PB207_U235_ERROR]
                self.pb206U238Value = values[Column.PB206_U238_VALUE]
                self.pb206U238Error = values[Column.PB206_U238_ERROR]

                self.pb207U235StDev = calculations.to1StdDev(
                    self.pb207U235Value,
                    self.pb207U235Error,
                    getattr(settings, "pb207U235ErrorType", "Absolute"),
                    getattr(settings, "pb207U235ErrorSigmas", 2),
                )
                self.pb206U238StDev = calculations.to1StdDev(
                    self.pb206U238Value,
                    self.pb206U238Error,
                    getattr(settings, "pb206U238ErrorType", "Absolute"),
                    getattr(settings, "pb206U238ErrorSigmas", 2),
                )

                # Derive TW ratios and 1σ uncertainties (no covariance).
                self.uPbValue, self.pbPbValue = wetherill_to_tw(
                    self.pb207U235Value,
                    self.pb206U238Value,
                    u238u235_ratio=U,
                )

                self.uPbStDev, self.pbPbStDev = propagate_wetherill_to_tw_uncertainty(
                    self.pb207U235Value,
                    self.pb207U235StDev,
                    self.pb206U238Value,
                    self.pb206U238StDev,
                    u238u235_ratio=U,
                )

                # Store TW errors in the user's configured representation
                # (mainly for consistency; the TW pipeline uses StDev).
                self.uPbError = calculations.from1StdDev(
                    self.uPbValue,
                    self.uPbStDev,
                    settings.uPbErrorType,
                    settings.uPbErrorSigmas,
                )
                self.pbPbError = calculations.from1StdDev(
                    self.pbPbValue,
                    self.pbPbStDev,
                    settings.pbPbErrorType,
                    settings.pbPbErrorSigmas,
                )

            else:
                # --- imported in TW ---
                self.uPbValue = values[Column.U_PB_VALUE]
                self.uPbError = values[Column.U_PB_ERROR]
                self.pbPbValue = values[Column.PB_PB_VALUE]
                self.pbPbError = values[Column.PB_PB_ERROR]

                self.uPbStDev = calculations.to1StdDev(
                    self.uPbValue,
                    self.uPbError,
                    settings.uPbErrorType,
                    settings.uPbErrorSigmas,
                )
                self.pbPbStDev = calculations.to1StdDev(
                    self.pbPbValue,
                    self.pbPbError,
                    settings.pbPbErrorType,
                    settings.pbPbErrorSigmas,
                )

                # Derive Wetherill ratios and 1σ uncertainties (no covariance).
                self.pb207U235Value, self.pb206U238Value = tw_to_wetherill(
                    self.uPbValue,
                    self.pbPbValue,
                    u238u235_ratio=U,
                )
                self.pb207U235StDev, self.pb206U238StDev = propagate_tw_to_wetherill_uncertainty(
                    self.uPbValue,
                    self.uPbStDev,
                    self.pbPbValue,
                    self.pbPbStDev,
                    u238u235_ratio=U,
                )
                self.pb207U235Error = calculations.from1StdDev(
                    self.pb207U235Value,
                    self.pb207U235StDev,
                    getattr(settings, "pb207U235ErrorType", "Absolute"),
                    getattr(settings, "pb207U235ErrorSigmas", 2),
                )
                self.pb206U238Error = calculations.from1StdDev(
                    self.pb206U238Value,
                    self.pb206U238StDev,
                    getattr(settings, "pb206U238ErrorType", "Absolute"),
                    getattr(settings, "pb206U238ErrorSigmas", 2),
                )

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
