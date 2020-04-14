from model.cell import CalculatedCell, ImportedCell, UncalculatedCell
from model.column import Column
from model.settings.calculation import LeadLossCalculationSettings
from model.settings.type import SettingsType
from process import calculations
from utils.settings import Settings


class Row:

    def __init__(self, importedValues, importSettings):

        self.rawImportedValues = importedValues
        self.calculatedValues = None
        self.processed = False
        self.calculatedCellSpecs = LeadLossCalculationSettings.getDefaultHeaders()

        displayedImportedColumns = importSettings.getDisplayColumnsWithRefs()
        self.importedCells = [ImportedCell(importedValues[i]) for _, i in displayedImportedColumns]
        self.importedCellsByCol = {col: self.importedCells[j] for j, (col, _) in
                                   enumerate(displayedImportedColumns)}
        self.validImports = all(cell.isValid() for cell in self.importedCells)
        self.resetCalculatedCells()


    def setCalculatedValues(self, calculatedValues):
        self.calculatedValues = calculatedValues

    def getExportedValues(self, exportSettings):
        return

    def getDisplayCells(self):
        return self.importedCells + self.calculatedCells

    def resetCalculatedCells(self):
        self.calculatedCells = [UncalculatedCell() for _ in self.calculatedCellSpecs]
        self.processed = False


    def uPbValue(self):
        return self.importedCellsByCol[Column.U_PB_VALUE].value

    def uPbError(self):
        return self.importedCellsByCol[Column.U_PB_ERROR].value

    def uPbStDev(self, importSettings):
        return calculations.convert_to_stddev(
            self.uPbValue(),
            self.uPbError(),
            importSettings.uPbErrorType,
            importSettings.uPbErrorSigmas
        )

    def pbPbValue(self):
        return self.importedCellsByCol[Column.PB_PB_VALUE].value

    def pbPbError(self):
        return self.importedCellsByCol[Column.PB_PB_ERROR].value

    def pbPbStDev(self, importSettings):
        return calculations.convert_to_stddev(
            self.pbPbValue(),
            self.pbPbError(),
            importSettings.pbPbErrorType,
            importSettings.pbPbErrorSigmas
        )

    def setConcordantAge(self, discordance, concordantAge):
        self.processed = True
        self.concordant = concordantAge is not None
        self.concordantAge = concordantAge
        self.calculatedCells = [
            CalculatedCell("Yes" if self.concordant else "No"),
            CalculatedCell(discordance * 100 if discordance else None),
            CalculatedCell(None if concordantAge is None else concordantAge / (10 ** 6))
        ]