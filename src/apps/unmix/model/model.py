from apps.type import ApplicationType
from apps.unmix.model.row import Row
from apps.abstract.model.model import AbstractModel
from apps.unmix.model.settings.calculation import UnmixCalculationSettings

class UnmixModel(AbstractModel):

    def __init__(self, signals):
        super().__init__(ApplicationType.UNMIX, signals)
        self.rows = None

    def loadRawData(self, importSettings, rawHeaders, rawRows):
        self.rawHeaders = rawHeaders
        self.rows = [Row(row, importSettings) for row in rawRows]

        importHeaders = importSettings.getHeaders()
        calculationHeaders = UnmixCalculationSettings.getDefaultHeaders()
        headers = importHeaders + calculationHeaders

        self.view.onHeadersUpdated(headers)
        self.view.onAllRowsUpdated(self.rows)

    def getProcessingFunction(self):
        return process

    def getProcessingData(self):
        return self.rows

    def addProcessingOutput(self, *args):
        pass

    def getExportData(self, calculationSettings):
        headers = self.rawHeaders + calculationSettings.getExportHeaders()
        rows = [row.rawImportedValues + [cell.value for cell in row.calculatedCells] for row in self.rows]
        return headers, rows


def process(signals, rows, importSettings, calculationSettings):
    for i, row in enumerate(rows):
        if signals.halt():
            signals.cancelled()
            return
        row.process(importSettings, calculationSettings)
        signals.progress((i+1)/len(rows), i, row)
    signals.completed(None)