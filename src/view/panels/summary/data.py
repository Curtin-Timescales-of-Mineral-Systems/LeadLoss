from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QTableWidget, QWidget, QTableWidgetItem, QVBoxLayout, QPushButton

from utils import stringUtils
from utils.ui.icons import Icons


class SummaryDataPanel(QWidget):

    def __init__(self, controller, samples):
        super().__init__()
        self.controller = controller
        self.samples = samples

        self._createUI(samples)

        app = lambda f, x: lambda: f(x)
        for sample in samples:
            sample.signals.concordancyCalculated.connect(app(self._onSampleConcordancyCalculated, sample))
            sample.signals.optimalAgeCalculated.connect(app(self._onOptimalAgeCalculated, sample))

        self.dataTable.selectionModel().selectionChanged.connect(self._onSelectionChanged)

    #############
    ## Widgets ##
    #############

    def _createUI(self, samples):
        headers = [
            "Sample",
            "Concordant\npoints",
            "Discordant\npoints",
            "95%\nlower\nbound",
            "Pb-loss\nage (Ma)",
            "95%\nupper\nbound",
            "d-value",
            "p-value",
            "Score"
        ]

        self.dataTable = QTableWidget(len(samples), len(headers))
        self.dataTable.setHorizontalHeaderLabels(headers)

        for i, sample in enumerate(samples):
            self.dataTable.setItem(i, 0, self._createTableCellWidget(sample.name))
            self.dataTable.setItem(i, 1, self._createTableCellWidget(""))
            self.dataTable.setItem(i, 2, self._createTableCellWidget(""))
            self.dataTable.setItem(i, 3, self._createTableCellWidget(""))
            self.dataTable.setItem(i, 4, self._createTableCellWidget(""))
            self.dataTable.setItem(i, 5, self._createTableCellWidget(""))
            self.dataTable.setItem(i, 6, self._createTableCellWidget(""))
            self.dataTable.setItem(i, 7, self._createTableCellWidget(""))
            self.dataTable.setItem(i, 8, self._createTableCellWidget(""))
        self.dataTable.resizeColumnsToContents()
        self.dataTable.resizeRowsToContents()

        self.exportButton = QPushButton("  Export table")
        self.exportButton.clicked.connect(self._onExportClicked)
        self.exportButton.setIcon(Icons.export())

        layout = QVBoxLayout()
        layout.addWidget(self.dataTable)
        layout.addWidget(self.exportButton)
        self.setLayout(layout)

    def _createTableCellWidget(self, content):
        cell = QTableWidgetItem(content)
        cell.setTextAlignment(Qt.AlignHCenter)
        cell.setFlags(cell.flags() ^ Qt.ItemIsEditable)
        return cell

    ############
    ## Events ##
    ############

    def _onSampleConcordancyCalculated(self, sample):
        self.dataTable.setItem(sample.id, 1, self._createTableCellWidget(str(len(sample.concordantSpots()))))
        self.dataTable.setItem(sample.id, 2, self._createTableCellWidget(str(len(sample.discordantSpots()))))
        self.dataTable.update()
        self.dataTable.resizeColumnsToContents()

    def _onOptimalAgeCalculated(self, sample):
        def create(value):
            return self._createTableCellWidget(str(stringUtils.round_to_sf(value)))

        self.dataTable.setItem(sample.id, 3, create(sample.optimalAgeLowerBound/(10**6)))
        self.dataTable.setItem(sample.id, 4, create(sample.optimalAge/(10**6)))
        self.dataTable.setItem(sample.id, 5, create(sample.optimalAgeUpperBound/(10**6)))
        self.dataTable.setItem(sample.id, 6, create(sample.optimalAgeDValue))
        self.dataTable.setItem(sample.id, 7, create(sample.optimalAgePValue))
        self.dataTable.setItem(sample.id, 8, create(sample.optimalAgeScore))
        self.dataTable.update()
        self.dataTable.resizeColumnsToContents()

    def _onExportClicked(self):
        n = self.dataTable.columnCount()
        headers = [self.dataTable.horizontalHeaderItem(i).text() for i in range(n)]
        headers = [header.replace("\n", " ") for header in headers]
        data = [[self.dataTable.item(row, col).text() for col in range(n)] for row in range(self.dataTable.rowCount())]
        self.controller.exportCSV(headers, data)

    def _onSelectionChanged(self):
        selectedRows = set(index.row() for index in self.dataTable.selectedIndexes())
        self.controller.selectSamples(selectedRows)