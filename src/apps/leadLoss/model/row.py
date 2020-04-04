from apps.abstract.model.cell import CalculatedCell
from apps.abstract.model.row import AbstractRow
from apps.leadLoss.model.column import Column
from apps.leadLoss.model.settings.calculation import LeadLossCalculationSettings


class Row(AbstractRow):

    def __init__(self, importedValues, importSettings):
        super().__init__(importedValues, importSettings, LeadLossCalculationSettings.getDefaultHeaders())

    def uPbValue(self):
        return self.importedCellsByCol[Column.U_PB_VALUE].value

    def pbPbValue(self):
        return self.importedCellsByCol[Column.U_PB_VALUE].value

    def setConcordantAge(self, discordance, concordantAge):
        self.processed = True
        self.concordant = concordantAge is not None
        self.calculatedCells[0] = CalculatedCell(discordance*100)
        self.calculatedCells[1] = CalculatedCell(None if concordantAge is None else concordantAge/(10**6))