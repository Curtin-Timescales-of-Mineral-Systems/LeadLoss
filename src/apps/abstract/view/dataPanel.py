from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

from utils.ui import uiUtils


class AbstractDataPanel(QGroupBox):

    def __init__(self, controller, *args, **kwargs):
        super(AbstractDataPanel, self).__init__("Data", *args, **kwargs)

        self.controller = controller

        self.controller.signals.headersUpdated.connect(self.onHeadersUpdated)
        self.controller.signals.rowUpdated.connect(self.onRowUpdated)
        self.controller.signals.allRowsUpdated.connect(self.onAllRowsUpdated)

        self.controller.signals.processingStarted.connect(self.onProcessingStart)
        self.controller.signals.processingCompleted.connect(self.onProcessingEnd)
        self.controller.signals.processingErrored.connect(self.onProcessingEnd)
        self.controller.signals.processingCancelled.connect(self.onProcessingEnd)

    #############
    ## UI spec ##
    #############

    def _initImportWidget(self):

        self.importButton = QPushButton("  Import CSV")
        self.importButton.clicked.connect(self.controller.importCSV)
        self.importButton.setIcon(self.style().standardIcon(QStyle.SP_DialogOpenButton))

        self.importFileText = QLineEdit("")
        self.importFileText.setReadOnly(True)

        self.helpButton = QPushButton("  Help")
        self.helpButton.clicked.connect(self.controller.showHelp)
        self.helpButton.setIcon(self.style().standardIcon(QStyle.SP_MessageBoxQuestion))

        self.importWidget = QWidget()
        layout = QHBoxLayout()
        layout.addWidget(self.importButton)
        layout.addWidget(self.importFileText)
        layout.addWidget(self.helpButton)
        layout.setContentsMargins(0,0,0,5)
        self.importWidget.setLayout(layout)

    def _initDataTable(self):
        self.dataTable = QTableWidget(1, 1)
        self.dataTable.resizeColumnsToContents()
        self.dataTable.resizeRowsToContents()
        self.dataTable.hide()
        self.dataTable.itemSelectionChanged.connect(self._selectionChanged)
        uiUtils.retainSizeWhenHidden(self.dataTable)

    def _initTableWidgetItem(self, content):
        cell = QTableWidgetItem(str(content))
        cell.setTextAlignment(Qt.AlignHCenter)
        cell.setFlags(cell.flags() ^ Qt.ItemIsEditable)
        return cell

    def _initActionButtonsWidget(self):
        self.processButton = QPushButton("  Process")
        self.processButton.clicked.connect(self.controller.process)
        self.processButton.setIcon(self.style().standardIcon(QStyle.SP_ArrowForward))

        self.exportButton = QPushButton("  Export CSV")
        self.exportButton.clicked.connect(self.controller.exportCSV)
        self.exportButton.setEnabled(False)
        self.exportButton.setIcon(self.style().standardIcon(QStyle.SP_DialogSaveButton))

        self.actionButtonsWidget = QWidget()
        layout = QHBoxLayout()
        layout.addWidget(self.processButton)
        layout.addWidget(self.exportButton)
        self.actionButtonsWidget.setLayout(layout)

    #############
    ## Updates ##
    #############

    def afterSuccessfulCSVImport(self, inputFile):
        self.importFileText.setText(inputFile)

        self.dataTable.show()
        self.actionButtonsWidget.show()

    def onProcessingStart(self):
        for button in self.getActionButtons():
            button.setEnabled(button == self.processButton)
        self.processButton.setEnabled(True)
        self.processButton.setText("  Cancel processing")
        self.processButton.setIcon(self.style().standardIcon(QStyle.SP_DialogCancelButton))
        self.processButton.clicked.disconnect(self.controller.process)
        self.processButton.clicked.connect(self.controller.cancelProcessing)

    def onProcessingEnd(self):
        for button in self.getActionButtons():
            button.setEnabled(True)
        self.processButton.setText("  Process")
        self.processButton.setIcon(self.style().standardIcon(QStyle.SP_ArrowForward))
        self.processButton.clicked.disconnect(self.controller.cancelProcessing)
        self.processButton.clicked.connect(self.controller.process)

    def onHeadersUpdated(self, headers):
        self.dataTable.setColumnCount(len(headers))
        self.dataTable.setHorizontalHeaderLabels(headers)

    def onAllRowsUpdated(self, rows):
        self.dataTable.setRowCount(len(rows))
        for index, row in enumerate(rows):
            self.onRowUpdated(index, row)

    def onRowUpdated(self, i, row):
        self._refreshRowHeader(i, row)
        self._refreshRowData(i, row)

        self.dataTable.resizeColumnsToContents()
        self.dataTable.resizeRowsToContents()

    def _refreshRowData(self, i, row):
        for j, cell in enumerate(row.getDisplayCells()):
            tableCell = self._initTableWidgetItem(cell.getDisplayString())
            if not cell.isValid():
                if cell.isImported():
                    tableCell.setBackground(QColor(255, 0, 0, 27))
                elif row.validImports and row.processed:
                    tableCell.setBackground(QColor(255, 165, 0, 27))
            self.dataTable.setItem(i, j, tableCell)
        self.dataTable.viewport().update()
        self.dataTable.resizeColumnsToContents()
