from model.column import Column
from utils import stringUtils, csvUtils

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
        return [
            stringUtils.U_PB_STR,
            "±" + self.getUPbErrorStr(),
            stringUtils.PB_PB_STR,
            "±" + self.getPbPbErrorStr()
        ]

    def getDisplayColumns(self):
        numbers = [v for k,v in self._columnRefs.items() if not (k == Column.SAMPLE_NAME and not self.multipleSamples)]
        numbers.sort()
        return numbers

    def getColumn(self, column):
        return csvUtils.columnLettersToNumber(self._columnRefs[column], zeroIndexed=True)

    def getDisplayColumnsWithRefs(self):
        numbers = [(col, csvUtils.columnLettersToNumber(colRef, zeroIndexed=True)) for col, colRef in
                   self._columnRefs.items() if col != Column.SAMPLE_NAME]
        numbers.sort(key=lambda v: v[0].value)
        return numbers

    def getDisplayColumnsByRefs(self):
        return self._columnRefs

    def validate(self):
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
