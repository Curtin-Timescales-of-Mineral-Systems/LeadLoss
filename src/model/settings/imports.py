from model.column import Column
from utils import stringUtils, csvUtils
from model.settings.ratio import (
    ConcordiaRatioSpace,
    ConcordiaSpaceSelection,
    ratio_space_from_value,
    resolve_space_selection,
    space_selection_from_value,
)

from model.settings.type import SettingsType
from utils.csvUtils import ColumnReferenceType


class LeadLossImportSettings:

    KEY = SettingsType.IMPORT

    @staticmethod
    def getImportedColumnNames():
        return [
            Column.SAMPLE_NAME,
            Column.U_PB_VALUE,
            Column.U_PB_ERROR,
            Column.PB_PB_VALUE,
            Column.PB_PB_ERROR,
        ]

    def __init__(self):
        self.delimiter = ","
        self.hasHeaders = True
        self.columnReferenceType = ColumnReferenceType.LETTERS
        self._columnRefs = {name: i for i, name in enumerate(LeadLossImportSettings.getImportedColumnNames())}
        self.rhoColumn = None
        self.inputRatioSpace = ConcordiaRatioSpace.TERA_WASSERBURG
        self.displayRatioSpace = ConcordiaSpaceSelection.SAME_AS_INPUT

        self.uPbErrorType = "Absolute"
        self.uPbErrorSigmas = 2

        self.pbPbErrorType = "Absolute"
        self.pbPbErrorSigmas = 2

        self.multipleSamples = True
        self.sampleNameColumn = 0

    def getUPbErrorStr(self):
        return stringUtils.get_error_str(self.uPbErrorSigmas, self.uPbErrorType)

    def getPbPbErrorStr(self):
        return stringUtils.get_error_str(self.pbPbErrorSigmas, self.pbPbErrorType)

    def getHeaders(self):
        x_label, y_label = stringUtils.getRatioLabels(self.getInputRatioSpace(), True)
        headers = [
            x_label,
            "±" + self.getUPbErrorStr(),
            y_label,
            "±" + self.getPbPbErrorStr()
        ]
        if self.getRhoColumn() is not None:
            headers.append("rho")
        return headers

    def getInputRatioSpace(self):
        self.inputRatioSpace = ratio_space_from_value(
            getattr(self, "inputRatioSpace", ConcordiaRatioSpace.TERA_WASSERBURG)
        )
        return self.inputRatioSpace

    def getDisplayRatioSpace(self):
        self.displayRatioSpace = space_selection_from_value(
            getattr(self, "displayRatioSpace", ConcordiaSpaceSelection.SAME_AS_INPUT)
        )
        return resolve_space_selection(self.displayRatioSpace, self.getInputRatioSpace())

    def getRhoColumn(self):
        return getattr(self, "rhoColumn", None)

    def setRhoColumn(self, value):
        self.rhoColumn = value

    def ensureCompatibility(self):
        if not hasattr(self, "rhoColumn"):
            self.rhoColumn = None
        if not hasattr(self, "inputRatioSpace"):
            self.inputRatioSpace = ConcordiaRatioSpace.TERA_WASSERBURG
        else:
            self.inputRatioSpace = ratio_space_from_value(self.inputRatioSpace)
        if not hasattr(self, "displayRatioSpace"):
            self.displayRatioSpace = ConcordiaSpaceSelection.SAME_AS_INPUT
        else:
            self.displayRatioSpace = space_selection_from_value(self.displayRatioSpace)
        return self

    def getLegacyHeaders(self):
        return [
            stringUtils.U_PB_STR,
            "±" + self.getUPbErrorStr(),
            stringUtils.PB_PB_STR,
            "±" + self.getPbPbErrorStr()
        ]

    def getDisplayColumns(self):
        numbers = [
            v for k, v in self._columnRefs.items()
            if v is not None and not (k == Column.SAMPLE_NAME and not self.multipleSamples)
        ]
        rho_col = self.getRhoColumn()
        if rho_col is not None:
            numbers.append(rho_col)
        numbers.sort()
        return numbers

    def getColumn(self, column):
        return csvUtils.columnLettersToNumber(self._columnRefs[column], zeroIndexed=True)

    def getDisplayColumnsWithRefs(self):
        numbers = [(col, csvUtils.columnLettersToNumber(colRef, zeroIndexed=True)) for col, colRef in
                   self._columnRefs.items() if col != Column.SAMPLE_NAME]
        if self.getRhoColumn() is not None:
            numbers.append((Column.ERROR_CORRELATION, csvUtils.columnLettersToNumber(self.getRhoColumn(), zeroIndexed=True)))
        numbers.sort(key=lambda v: v[0].value)
        return numbers

    def getDisplayColumnsByRefs(self):
        return self._columnRefs

    def validate(self):
        self.ensureCompatibility()
        if not all([v is not None for v in self._columnRefs.values()]):
            return "Must enter a value for each column"

        columnsByRef = set()
        for col in self.getDisplayColumns():
            if col not in columnsByRef:
                columnsByRef.add(col)
                continue

            if self.columnReferenceType == ColumnReferenceType.LETTERS:
                col = csvUtils.columnNumberToLetters(col, zeroIndexed=True)
            else:
                col = str(col+1)

            return "Column " + col + " is used more than once"

        return None
