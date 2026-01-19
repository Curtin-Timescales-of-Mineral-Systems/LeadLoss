from enum import Enum

from model.column import Column
from utils import stringUtils, csvUtils

from model.settings.type import SettingsType
from utils.csvUtils import ColumnReferenceType


class LeadLossImportSettings:

    KEY = SettingsType.IMPORT

    class RatioInputMode(Enum):
        TW = "TW"
        WETHERILL = "Wetherill"

    @staticmethod
    def getImportedColumnNames():
        # Superset: we allow either TW or Wetherill columns to be mapped.
        return [
            Column.SAMPLE_NAME,

            # TW
            Column.U_PB_VALUE,
            Column.U_PB_ERROR,
            Column.PB_PB_VALUE,
            Column.PB_PB_ERROR,

            # Wetherill
            Column.PB207_U235_VALUE,
            Column.PB207_U235_ERROR,
            Column.PB206_U238_VALUE,
            Column.PB206_U238_ERROR,
        ]

    def __init__(self):
        self.delimiter = ","
        self.hasHeaders = True
        self.columnReferenceType = ColumnReferenceType.LETTERS
        self._columnRefs = {name: i for i, name in enumerate(LeadLossImportSettings.getImportedColumnNames())}

        # TW error settings
        self.uPbErrorType = "Absolute"
        self.uPbErrorSigmas = 2

        self.pbPbErrorType = "Absolute"
        self.pbPbErrorSigmas = 2

        # Wetherill error settings
        self.pb207U235ErrorType = "Absolute"
        self.pb207U235ErrorSigmas = 2

        self.pb206U238ErrorType = "Absolute"
        self.pb206U238ErrorSigmas = 2

        # NEW: which ratios are present in the CSV file
        self.ratioInputMode = LeadLossImportSettings.RatioInputMode.TW

        self.multipleSamples = True
        self.sampleNameColumn = 0

    def _mode(self):
        # Backward-safe: older pickles may not have ratioInputMode
        mode = getattr(self, "ratioInputMode", LeadLossImportSettings.RatioInputMode.TW)
        if isinstance(mode, str):
            mode_l = mode.lower()
            return LeadLossImportSettings.RatioInputMode.WETHERILL if "weth" in mode_l else LeadLossImportSettings.RatioInputMode.TW
        return mode

    def getRequiredColumnNames(self):
        if self._mode() == LeadLossImportSettings.RatioInputMode.WETHERILL:
            cols = [
                Column.SAMPLE_NAME,
                Column.PB207_U235_VALUE,
                Column.PB207_U235_ERROR,
                Column.PB206_U238_VALUE,
                Column.PB206_U238_ERROR,
            ]
        else:
            cols = [
                Column.SAMPLE_NAME,
                Column.U_PB_VALUE,
                Column.U_PB_ERROR,
                Column.PB_PB_VALUE,
                Column.PB_PB_ERROR,
            ]

        if not getattr(self, "multipleSamples", True) and Column.SAMPLE_NAME in cols:
            cols.remove(Column.SAMPLE_NAME)

        return cols

    def getUPbErrorStr(self):
        return stringUtils.get_error_str(getattr(self, "uPbErrorSigmas", 2), getattr(self, "uPbErrorType", "Absolute"))

    def getPbPbErrorStr(self):
        return stringUtils.get_error_str(getattr(self, "pbPbErrorSigmas", 2), getattr(self, "pbPbErrorType", "Absolute"))

    def getPb207U235ErrorStr(self):
        return stringUtils.get_error_str(getattr(self, "pb207U235ErrorSigmas", 2), getattr(self, "pb207U235ErrorType", "Absolute"))

    def getPb206U238ErrorStr(self):
        return stringUtils.get_error_str(getattr(self, "pb206U238ErrorSigmas", 2), getattr(self, "pb206U238ErrorType", "Absolute"))

    def getHeaders(self):
        # These are headers for the "value/error" columns (sample name is handled separately).
        if self._mode() == LeadLossImportSettings.RatioInputMode.WETHERILL:
            return [
                "207Pb/235U",
                "±" + self.getPb207U235ErrorStr(),
                "206Pb/238U",
                "±" + self.getPb206U238ErrorStr(),
            ]

        return [
            stringUtils.U_PB_STR,
            "±" + self.getUPbErrorStr(),
            stringUtils.PB_PB_STR,
            "±" + self.getPbPbErrorStr()
        ]

    def getDisplayColumns(self):
        # Used to validate the CSV has all required columns.
        required = self.getRequiredColumnNames()
        numbers = [self._columnRefs[c] for c in required if c in self._columnRefs]
        numbers.sort()
        return numbers

    def getColumn(self, column):
        return csvUtils.columnLettersToNumber(self._columnRefs[column], zeroIndexed=True)

    def getDisplayColumnsWithRefs(self):
        required = self.getRequiredColumnNames()
        pairs = []
        for col in required:
            if col == Column.SAMPLE_NAME:
                continue
            colRef = self._columnRefs.get(col)
            if colRef is None:
                continue
            pairs.append((col, csvUtils.columnLettersToNumber(colRef, zeroIndexed=True)))
        pairs.sort(key=lambda v: v[0].value)
        return pairs

    def getDisplayColumnsByRefs(self):
        return self._columnRefs

    def validate(self):
        required = self.getRequiredColumnNames()
        if not all([self._columnRefs.get(k) is not None for k in required]):
            return "Must enter a value for each required column"

        columnsByRef = set()
        for col in self.getDisplayColumns():
            if col not in columnsByRef:
                columnsByRef.add(col)
                continue

            if self.columnReferenceType == ColumnReferenceType.LETTERS:
                col = csvUtils.columnNumberToLetters(col, zeroIndexed=True)
            else:
                col = str(col + 1)

            return "Column " + col + " is used more than once"

        return None
