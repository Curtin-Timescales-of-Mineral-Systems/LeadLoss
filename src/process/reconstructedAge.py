
class ReconstructedAge():
    def __init__(self, values, minValues, maxValues):
        self.values = list(values)
        self.minValues = list(minValues) if minValues else [None]*3
        self.maxValues = list(maxValues) if maxValues else [None]*3
        self.fullyValid = minValues is not None and maxValues is not None

        # Swap uPb values as they are inverted
        t1 = self.minValues[1]
        self.minValues[1] = self.maxValues[1]
        self.maxValues[1] = t1

    def getAge(self):
        return self._getValuesAndError(0)

    def getUPb(self):
        return self._getValuesAndError(1)

    def getPbPb(self):
        return self._getValuesAndError(2)

    def _getValuesAndError(self, i):
        scale = (10 ** -6) if i == 0 else 1

        value = self.values[i]
        if value:
            value *= scale

        minValue = self.minValues[i]
        if minValue:
            minValue = value - scale*minValue

        maxValue = self.maxValues[i]
        if maxValue:
            maxValue = scale*self.maxValues[i] - value

        return value, minValue, maxValue