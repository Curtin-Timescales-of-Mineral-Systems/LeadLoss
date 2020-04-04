from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

from utils.csvUtils import ColumnReferenceType
from utils.ui import uiUtils
from apps.abstract.view.settingsDialogs.general import AbstractSettingsDialog
from utils.ui.columnReferenceInput import ColumnReferenceInput
from utils.ui.columnReferenceTypeInput import ColumnReferenceTypeInput
from utils.ui.errorTypeInput import ErrorTypeInput


class AbstractImportSettingsDialog(AbstractSettingsDialog):

    def __init__(self, defaultSettings, *args, **kwargs):
        super().__init__(defaultSettings, *args, **kwargs)
        self.setWindowTitle("CSV import settings")

    def _onColumnRefChange(self, button):
        newRefType = ColumnReferenceType(button.option)
        self._updateColumnRefs(newRefType)
        self._validate()

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