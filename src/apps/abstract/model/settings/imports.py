from utils import csvUtils
from utils.csvUtils import ColumnReferenceType


class AbstractImportSettings:

    def __init__(self, importColumnSpecs):
        self.delimiter = ","
        self.hasHeaders = True
        self.columnReferenceType = ColumnReferenceType.LETTERS
        self._columnRefs = {spec.type: i for i, spec in enumerate(importColumnSpecs)}

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
