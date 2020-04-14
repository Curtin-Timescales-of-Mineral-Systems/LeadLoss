from PyQt5.QtWidgets import QGridLayout, QWidget, QGroupBox, QLineEdit, QCheckBox, QFormLayout

from model.column import Column
from utils import stringUtils
from model.settings.imports import LeadLossImportSettings
from utils.csvUtils import ColumnReferenceType
from utils.ui import uiUtils
from utils.ui.columnReferenceInput import ColumnReferenceInput
from utils.ui.columnReferenceTypeInput import ColumnReferenceTypeInput
from utils.ui.errorTypeInput import ErrorTypeInput


class LeadLossImportSettingsDialog:

    def __init__(self, defaultSettings, *args, **kwargs):
        super().__init__(defaultSettings, *args, **kwargs)
        self.setWindowTitle("CSV import settings")

    def _onColumnRefChange(self, button):
        newRefType = ColumnReferenceType(button.option)
        self._updateColumnRefs(newRefType)
        self._validate()

    ###############
    ## UI layout ##
    ###############

    def initMainSettings(self):
        defaults = self.defaultSettings
        columnRefs = defaults.getDisplayColumnsByRefs()

        self._generalSettingsWidget = GeneralSettingsWidget(self._validate, defaults)
        self._generalSettingsWidget.columnRefChanged.connect(self._onColumnRefChange)

        self._uPbWidget = ImportedValueErrorWidget(
            stringUtils.getUPbStr(True),
            self._validate,
            defaults.columnReferenceType,
            columnRefs[Column.U_PB_VALUE],
            columnRefs[Column.U_PB_ERROR],
            defaults.uPbErrorType,
            defaults.uPbErrorSigmas
        )

        self._pbPbWidget = ImportedValueErrorWidget(
            stringUtils.getPbPbStr(True),
            self._validate,
            defaults.columnReferenceType,
            columnRefs[Column.PB_PB_VALUE],
            columnRefs[Column.PB_PB_ERROR],
            defaults.pbPbErrorType,
            defaults.pbPbErrorSigmas
        )

        self._updateColumnRefs(defaults.columnReferenceType)

        layout = QGridLayout()
        layout.setHorizontalSpacing(15)
        layout.setVerticalSpacing(15)
        layout.addWidget(self._generalSettingsWidget, 0, 0)
        layout.addWidget(self._uPbWidget, 1, 0)
        layout.addWidget(self._pbPbWidget, 2, 0)

        widget = QWidget()
        widget.setLayout(layout)
        return widget

    ################
    ## Validation ##
    ################

    def _updateColumnRefs(self, newRefType):
        self._uPbWidget.changeColumnReferenceType(newRefType)
        self._pbPbWidget.changeColumnReferenceType(newRefType)

    def _createSettings(self):
        settings = LeadLossImportSettings()
        settings.delimiter = self._generalSettingsWidget.getDelimiter()
        settings.hasHeaders = self._generalSettingsWidget.getHasHeaders()
        settings.columnReferenceType = self._generalSettingsWidget.getColumnReferenceType()

        settings._columnRefs = {
            Column.U_PB_VALUE: self._uPbWidget.getValueColumn(),
            Column.U_PB_ERROR: self._uPbWidget.getErrorColumn(),
            Column.PB_PB_VALUE: self._pbPbWidget.getValueColumn(),
            Column.PB_PB_ERROR: self._pbPbWidget.getErrorColumn()
        }

        settings.uPbErrorType = self._uPbWidget.getErrorType()
        settings.uPbErrorSigmas = self._uPbWidget.getErrorSigmas()
        settings.pbPbErrorType = self._pbPbWidget.getErrorType()
        settings.pbPbErrorSigmas = self._pbPbWidget.getErrorSigmas()
        return settings

    def getWarning(self):
        return None

# Widget for displaying general CSV import settings
class GeneralSettingsWidget(QGroupBox):

    def __init__(self, validation, defaultSettings):
        super().__init__("General settings")

        self._delimiterEntry = QLineEdit(defaultSettings.delimiter)
        self._delimiterEntry.textChanged.connect(validation)
        self._delimiterEntry.setFixedWidth(30)
        self._delimiterEntry.setAlignment(Qt.AlignCenter)

        self._hasHeadersCB = QCheckBox()
        self._hasHeadersCB.setChecked(defaultSettings.hasHeaders)
        self._hasHeadersCB.stateChanged.connect(validation)

        self._columnRefType = ColumnReferenceTypeInput(validation, defaultSettings.columnReferenceType)
        self.columnRefChanged = self._columnRefType.group.buttonReleased

        layout = QFormLayout()
        layout.setHorizontalSpacing(uiUtils.FORM_HORIZONTAL_SPACING)
        layout.addRow("File headers", self._hasHeadersCB)
        layout.addRow("Column separator", self._delimiterEntry)
        layout.addRow("Refer to columns by", self._columnRefType)
        self.setLayout(layout)

    def getHasHeaders(self):
        return self._hasHeadersCB.isChecked()

    def getDelimiter(self):
        return self._delimiterEntry.text()

    def getColumnReferenceType(self):
        return self._columnRefType.selection()

# A widget for importing an (value, error) pair
class ImportedValueErrorWidget(QGroupBox):
    width = 30

    def __init__(self, title, validation, defaultReferenceType, defaultValueColumn, defaultErrorColumn, defaultErrorType, defaultErrorSigmas):
        super().__init__(title)

        self._valueColumn = ColumnReferenceInput(validation, defaultReferenceType, defaultValueColumn)
        self._errorColumn = ColumnReferenceInput(validation, defaultReferenceType, defaultErrorColumn)
        self._errorType = ErrorTypeInput(validation, defaultErrorType, defaultErrorSigmas)

        layout = QFormLayout()
        layout.setHorizontalSpacing(uiUtils.FORM_HORIZONTAL_SPACING)
        layout.addRow("Value column", self._valueColumn)
        layout.addRow("Error column", self._errorColumn)
        layout.addRow("Error type", self._errorType)
        self.setLayout(layout)

    def getValueColumn(self):
        return self._valueColumn.text()

    def getErrorColumn(self):
        return self._errorColumn.text()

    def getErrorType(self):
        return self._errorType.getErrorType()

    def getErrorSigmas(self):
        return self._errorType.getErrorSigmas()

    def changeColumnReferenceType(self, newReferenceType):
        self._valueColumn.changeColumnReferenceType(newReferenceType)
        self._errorColumn.changeColumnReferenceType(newReferenceType)