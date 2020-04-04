from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

from apps.abstract.view.dataPanel import AbstractDataPanel


class UnmixDataPanel(AbstractDataPanel):

    def __init__(self, controller, *args, **kwargs):
        super(UnmixDataPanel, self).__init__(controller, *args, **kwargs)
        self._initUI()

    #############
    ## UI spec ##
    #############

    def _initUI(self):
        self._initImportWidget()
        self._initDataTable()
        self._initActionButtonsWidget()

        layout = QVBoxLayout()
        layout.addWidget(self.importWidget)
        layout.addWidget(self.dataTable)
        layout.addWidget(self.actionButtonsWidget)
        self.setLayout(layout)

        self.actionButtonsWidget.hide()


    ######################
    ## Event processing ##
    ######################

    def _selectionChanged(self):
        rowIndices = sorted(set(index.row() for index in self.dataTable.selectedIndexes()))
        self.controller.selectRows(rowIndices)

    ###############
    ## Utilities ##
    ###############

    def getActionButtons(self):
        return [self.importButton, self.processButton, self.exportButton]

    def afterSuccessfulCSVImport(self, inputFile):
        super().afterSuccessfulCSVImport(inputFile)
        self.exportButton.show()

    def _colourTableRowHeader(self, i):
        row = self.rows[i]

    def _refreshRowHeader(self, i, row):
        header = self._initTableWidgetItem(i+1)
        self.dataTable.setVerticalHeaderItem(i, header)

        if not row.validImports:
            header.setBackground(QColor(255, 0, 0, 27))
            return
        if not row.processed:
            return
        if not row.validOutput: 
            header.setBackground(QColor(255, 165, 0, 27))
            return
        header.setBackground(QColor(0, 255, 0, 27))

