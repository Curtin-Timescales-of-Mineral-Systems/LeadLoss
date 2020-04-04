from utils import stringUtils


class Cell:
    def __init__(self, value, isImported):
        self.value = value
        self._isImported = isImported

    def isValid(self):
        return self.value is not None

    def isImported(self):
        return self._isImported

    def isCalculated(self):
        return not self._isImported


class ImportedCell(Cell):
    def __init__(self, rawValue):
        self._rawValue = rawValue
        try:
            value = float(rawValue)
        except:
            value = None
        super().__init__(value, True)

    def getDisplayString(self):
        if self.value is None:
            return self._rawValue
        return str(stringUtils.round_to_sf(self.value, 5))

class UncalculatedCell(Cell):
    def __init__(self):
        super().__init__(None, False)

    def isValid(self):
        True

    def getDisplayString(self):
        return ""

class CalculatedCell(Cell):
    def __init__(self, value):
        super().__init__(value, False)

    def isValid(self):
        return self.value is not None

    def getDisplayString(self):
        if self.value is None:
            return ""
        return str(stringUtils.round_to_sf(self.value, 5))