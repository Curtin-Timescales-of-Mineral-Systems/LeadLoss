from apps.abstract.model.cell import UncalculatedCell, ImportedCell


class AbstractRow:

    def __init__(self, importedValues, importSettings, calculatedCellSpecs):
        self.rawImportedValues = importedValues
        self.calculatedValues = None
        self.processed = False
        self.calculatedCellSpecs = calculatedCellSpecs

        displayedImportedColumns = importSettings.getDisplayColumnsWithRefs()
        self.importedCells = [ImportedCell(importedValues[i]) for _, i in displayedImportedColumns]
        self.importedCellsByCol = {col:self.importedCells[j] for j, (col, _) in enumerate(displayedImportedColumns)}
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