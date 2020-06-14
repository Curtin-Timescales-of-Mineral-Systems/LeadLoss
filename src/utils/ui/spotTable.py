from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem

from utils import config


def _createTableCellWidget(content):
    cell = QTableWidgetItem(str(content))
    cell.setTextAlignment(Qt.AlignHCenter)
    cell.setFlags(cell.flags() ^ Qt.ItemIsEditable)
    return cell


def createSpotTable(headers, spots):
    tableWidget = QTableWidget(len(spots), len(headers))
    tableWidget.setHorizontalHeaderLabels(headers)
    invalidColour = QColor(*config.INVALID_COLOUR_255)
    for i, spot in enumerate(spots):
        for j, value in enumerate(spot.displayStrings):
            cellWidget = _createTableCellWidget(value)
            if j in spot.invalidColumns:
                cellWidget.setBackground(invalidColour)
            tableWidget.setItem(i, j, cellWidget)

        rowHeaderWidget = QTableWidgetItem(str(i + 1))
        if not spot.valid:
            rowHeaderWidget.setBackground(invalidColour)
        tableWidget.setVerticalHeaderItem(i, rowHeaderWidget)

    tableWidget.resizeColumnsToContents()
    tableWidget.resizeRowsToContents()
    tableWidget.viewport().update()
    return tableWidget