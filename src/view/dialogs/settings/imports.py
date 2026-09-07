from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QGridLayout, QWidget, QGroupBox, QLineEdit, QCheckBox, QFormLayout, QLabel

from model.column import Column
from utils import stringUtils
from model.settings.imports import LeadLossImportSettings
from model.settings.ratio import ConcordiaRatioSpace, ConcordiaSpaceSelection
from utils.csvUtils import ColumnReferenceType
from utils.ui import uiUtils
from utils.ui.columnReferenceInput import ColumnReferenceInput
from utils.ui.columnReferenceTypeInput import ColumnReferenceTypeInput
from utils.ui.errorTypeInput import ErrorTypeInput
from utils.ui.radioButtons import EnumRadioButtonGroup
from view.dialogs.settings.abstract import AbstractSettingsDialog


class LeadLossImportSettingsDialog(AbstractSettingsDialog):

    def __init__(self, defaultSettings):
        super().__init__(defaultSettings)
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
        defaults.ensureCompatibility()
        columnRefs = defaults.getDisplayColumnsByRefs()

        self._generalSettingsWidget = GeneralSettingsWidget(self._validate, defaults)
        self._ratioSpaceWidget = RatioSpaceSettingsWidget(self._validate, defaults)
        self._sampleSettingsWidget = SampleSettingsWidget(self._validate, defaults)

        x_label, y_label = stringUtils.getRatioLabels(defaults.getInputRatioSpace(), True)
        self._uPbWidget = ImportedValueErrorWidget(
            x_label,
            self._validate,
            defaults.columnReferenceType,
            columnRefs[Column.U_PB_VALUE],
            columnRefs[Column.U_PB_ERROR],
            defaults.uPbErrorType,
            defaults.uPbErrorSigmas
        )

        self._pbPbWidget = ImportedValueErrorWidget(
            y_label,
            self._validate,
            defaults.columnReferenceType,
            columnRefs[Column.PB_PB_VALUE],
            columnRefs[Column.PB_PB_ERROR],
            defaults.pbPbErrorType,
            defaults.pbPbErrorSigmas
        )

        self._generalSettingsWidget.columnRefChanged.connect(self._onColumnRefChange)
        self._ratioSpaceWidget.inputSpaceChanged.connect(self._onInputRatioSpaceChanged)
        self._updateColumnRefs(defaults.columnReferenceType)
        self._onInputRatioSpaceChanged()

        layout = QGridLayout()
        layout.setHorizontalSpacing(15)
        layout.setVerticalSpacing(15)
        layout.addWidget(self._generalSettingsWidget, 0, 0)
        layout.addWidget(self._sampleSettingsWidget, 0, 1)
        layout.addWidget(self._ratioSpaceWidget, 1, 0, 1, 2)
        layout.addWidget(self._uPbWidget, 2, 0)
        layout.addWidget(self._pbPbWidget, 2, 1)

        widget = QWidget()
        widget.setLayout(layout)
        return widget

    ################
    ## Validation ##
    ################

    def _updateColumnRefs(self, newRefType):
        self._uPbWidget.changeColumnReferenceType(newRefType)
        self._pbPbWidget.changeColumnReferenceType(newRefType)
        self._ratioSpaceWidget.changeColumnReferenceType(newRefType)
        self._sampleSettingsWidget.changeColumnReferenceType(newRefType)

    def _onInputRatioSpaceChanged(self, *args):
        x_label, y_label = stringUtils.getRatioLabels(self._ratioSpaceWidget.getInputRatioSpace(), True)
        self._uPbWidget.setTitle(x_label)
        self._pbPbWidget.setTitle(y_label)
        self._ratioSpaceWidget.updateRhoVisibility()
        if hasattr(self, "okButton"):
            self._validate()

    def _createSettings(self):
        settings = LeadLossImportSettings()
        settings.delimiter = self._generalSettingsWidget.getDelimiter()
        settings.hasHeaders = self._generalSettingsWidget.getHasHeaders()
        settings.columnReferenceType = self._generalSettingsWidget.getColumnReferenceType()
        settings.inputRatioSpace = self._ratioSpaceWidget.getInputRatioSpace()
        settings.displayRatioSpace = self._ratioSpaceWidget.getDisplayRatioSpace()
        settings.rhoColumn = self._ratioSpaceWidget.getRhoColumn()

        settings.multipleSamples = self._sampleSettingsWidget.getMultipleSamples()

        settings._columnRefs = {
            Column.SAMPLE_NAME: self._sampleSettingsWidget.getSampleColumn(),
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

    def getWarning(self, settings):
        if (
            settings.getInputRatioSpace() == ConcordiaRatioSpace.WETHERILL
            and settings.getRhoColumn() is None
        ):
            return (
                "Native Wetherill import without a rho column assumes independent "
                "207Pb/235U and 206Pb/238U uncertainties."
            )
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


class RatioSpaceSettingsWidget(QGroupBox):
    def __init__(self, validation, defaultSettings):
        super().__init__("Concordia spaces")

        self._inputRatioSpace = EnumRadioButtonGroup(
            ConcordiaRatioSpace,
            validation,
            defaultSettings.getInputRatioSpace(),
            rows=None,
            cols=1,
        )
        self._displayRatioSpace = EnumRadioButtonGroup(
            ConcordiaSpaceSelection,
            validation,
            getattr(defaultSettings, "displayRatioSpace", ConcordiaSpaceSelection.SAME_AS_INPUT),
            rows=None,
            cols=1,
        )
        self.inputSpaceChanged = self._inputRatioSpace.group.buttonReleased

        self._rhoColumnLabel = QLabel("Error correlation (rho) column (optional)")
        self._rhoColumn = ColumnReferenceInput(
            validation,
            defaultSettings.columnReferenceType,
            defaultSettings.getRhoColumn(),
            allowEmpty=True,
        )

        layout = QFormLayout()
        layout.setHorizontalSpacing(uiUtils.FORM_HORIZONTAL_SPACING)
        layout.addRow("Input ratios", self._inputRatioSpace)
        layout.addRow("Display concordia", self._displayRatioSpace)
        layout.addRow(self._rhoColumnLabel, self._rhoColumn)
        self.setLayout(layout)

    def getInputRatioSpace(self):
        return self._inputRatioSpace.selection()

    def getDisplayRatioSpace(self):
        return self._displayRatioSpace.selection()

    def getRhoColumn(self):
        return self._rhoColumn.text()

    def updateRhoVisibility(self):
        # Both coordinate systems may include correlated uncertainties.
        # Existing TW files may leave this field blank.
        self._rhoColumnLabel.setVisible(True)
        self._rhoColumn.setVisible(True)

    def changeColumnReferenceType(self, newReferenceType):
        self._rhoColumn.changeColumnReferenceType(newReferenceType)


class SampleSettingsWidget(QGroupBox):

    def __init__(self, validation, defaultSettings):
        super().__init__("Sample settings")

        self._multipleSamplesCB = QCheckBox()
        self._multipleSamplesCB.setChecked(defaultSettings.multipleSamples)
        self._multipleSamplesCB.stateChanged.connect(validation)
        self._multipleSamplesCB.stateChanged.connect(self._onMultipleSamplesChanged)

        self._sampleColumnLabel = QLabel("Sample name column")
        self._sampleColumnLabel.setVisible(defaultSettings.multipleSamples)
        uiUtils.retainSizeWhenHidden(self._sampleColumnLabel)

        self._sampleColumn = ColumnReferenceInput(validation, defaultSettings.columnReferenceType,
                                                  defaultSettings.sampleNameColumn)
        self._sampleColumn.setVisible(defaultSettings.multipleSamples)

        layout = QFormLayout()
        layout.setHorizontalSpacing(uiUtils.FORM_HORIZONTAL_SPACING)
        layout.addRow("Multiple samples", self._multipleSamplesCB)
        layout.addRow(self._sampleColumnLabel, self._sampleColumn)
        self.setLayout(layout)

    def getMultipleSamples(self):
        return self._multipleSamplesCB.isChecked()

    def getSampleColumn(self):
        return self._sampleColumn.text()

    def changeColumnReferenceType(self, newReferenceType):
        self._sampleColumn.changeColumnReferenceType(newReferenceType)

    def _onMultipleSamplesChanged(self):
        self._sampleColumnLabel.setVisible(self._multipleSamplesCB.isChecked())
        self._sampleColumn.setVisible(self._multipleSamplesCB.isChecked())

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
