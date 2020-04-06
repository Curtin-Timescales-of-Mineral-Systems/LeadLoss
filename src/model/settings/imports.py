from model.columnSpec import ColumnSpec
from model.column import Column
from utils import stringUtils, csvUtils

from model.settings.type import SettingsType
from utils.csvUtils import ColumnReferenceType


class LeadLossImportSettings:

    KEY = SettingsType.IMPORT

    @staticmethod
    def getImportedColumnSpecs():
        return [
            ColumnSpec(Column.U_PB_VALUE),
            ColumnSpec(Column.U_PB_ERROR),
            ColumnSpec(Column.PB_PB_VALUE),
            ColumnSpec(Column.PB_PB_ERROR),
        ]

    def __init__(self):
        self.delimiter = ","
        self.hasHeaders = True
        self.columnReferenceType = ColumnReferenceType.LETTERS
        self._columnRefs = {spec.type: i for i, spec in enumerate(LeadLossImportSettings.getImportedColumnSpecs())}


        self.uPbErrorType = "Absolute"
        self.uPbErrorSigmas = 2

        self.pbPbErrorType = "Absolute"
        self.pbPbErrorSigmas = 2

        self.discordanceThreshold = 0.1

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
        numbers = list(self._columnRefs.values())
        numbers.sort()
        return numbers

    def getDisplayColumnsWithRefs(self):
        numbers = [(col, csvUtils.columnLettersToNumber(colRef, zeroIndexed=True)) for col, colRef in
                   self._columnRefs.items()]
        numbers.sort(key=lambda v: v[0].value)
        return numbers

    def getDisplayColumnsByRefs(self):
        return self._columnRefs

    def validate(self):
        if not all([v is not None for v in self._columnRefs.values()]):
            return "Must enter a value for each column"

        displayColumns = self.getDisplayColumns()
        if len(set(displayColumns)) != len(displayColumns):
            return "Columns should not contain duplicates"

        return None
