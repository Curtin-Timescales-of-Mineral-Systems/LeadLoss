from apps.type import SettingsType
from utils.settings import Settings

class AbstractModel:

    def __init__(self, applicationType, signals):
        self.applicationType = applicationType
        self.signals = signals

    def updateRow(self, i, row):
        self.rows[i] = row

    def resetCalculations(self):
        importSettings = Settings.get(self.applicationType, SettingsType.IMPORT)
        calculationSettings = Settings.get(self.applicationType, SettingsType.CALCULATION)
        headers = importSettings.getHeaders() + calculationSettings.getHeaders()

        for row in self.rows:
            row.resetCalculatedCells()

        self.signals.headersUpdated.emit(headers)
        self.signals.allRowsUpdated.emit(self.rows)


