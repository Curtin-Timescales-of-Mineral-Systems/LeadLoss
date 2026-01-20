from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QGridLayout, QWidget, QGroupBox, QLineEdit, QCheckBox, QFormLayout, QLabel, QComboBox

from model.column import Column
from utils import stringUtils
from model.settings.imports import LeadLossImportSettings
from utils.csvUtils import ColumnReferenceType
from utils.ui import uiUtils
from utils.ui.columnReferenceInput import ColumnReferenceInput
from utils.ui.columnReferenceTypeInput import ColumnReferenceTypeInput
from utils.ui.errorTypeInput import ErrorTypeInput
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
        columnRefs = defaults.getDisplayColumnsByRefs()

        self._generalSettingsWidget = GeneralSettingsWidget(self._validate, defaults)
        self._sampleSettingsWidget = SampleSettingsWidget(self._validate, defaults)

        # TW widgets
        self._uPbWidget = ImportedValueErrorWidget(
            stringUtils.getUPbStr(True),
            self._validate,
            defaults.columnReferenceType,
            columnRefs.get(Column.U_PB_VALUE, 1),
            columnRefs.get(Column.U_PB_ERROR, 2),
            defaults.uPbErrorType,
            defaults.uPbErrorSigmas
        )

        self._pbPbWidget = ImportedValueErrorWidget(
            stringUtils.getPbPbStr(True),
            self._validate,
            defaults.columnReferenceType,
            columnRefs.get(Column.PB_PB_VALUE, 3),
            columnRefs.get(Column.PB_PB_ERROR, 4),
            defaults.pbPbErrorType,
            defaults.pbPbErrorSigmas
        )

        # Wetherill widgets
        self._pb207U235Widget = ImportedValueErrorWidget(
            "207Pb/235U",
            self._validate,
            defaults.columnReferenceType,
            columnRefs.get(Column.PB207_U235_VALUE, 5),
            columnRefs.get(Column.PB207_U235_ERROR, 6),
            getattr(defaults, "pb207U235ErrorType", "Absolute"),
            getattr(defaults, "pb207U235ErrorSigmas", 2),
        )

        self._pb206U238Widget = ImportedValueErrorWidget(
            "206Pb/238U",
            self._validate,
            defaults.columnReferenceType,
            columnRefs.get(Column.PB206_U238_VALUE, 7),
            columnRefs.get(Column.PB206_U238_ERROR, 8),
            getattr(defaults, "pb206U238ErrorType", "Absolute"),
            getattr(defaults, "pb206U238ErrorSigmas", 2),
        )

        self._generalSettingsWidget.columnRefChanged.connect(self._onColumnRefChange)
        self._updateColumnRefs(defaults.columnReferenceType)

        layout = QGridLayout()
        layout.setHorizontalSpacing(15)
        layout.setVerticalSpacing(15)
        layout.addWidget(self._generalSettingsWidget, 0, 0)
        layout.addWidget(self._sampleSettingsWidget, 0, 1)
        layout.addWidget(self._uPbWidget, 1, 0)
        layout.addWidget(self._pbPbWidget, 1, 1)
        layout.addWidget(self._pb207U235Widget, 2, 0)
        layout.addWidget(self._pb206U238Widget, 2, 1)

        widget = QWidget()
        widget.setLayout(layout)

        # Show/hide TW vs Wetherill widgets based on current selection
        self._generalSettingsWidget.ratioModeChanged.connect(self._onRatioModeChanged)
        self._onRatioModeChanged()

        return widget

    def _onRatioModeChanged(self, *_args):
        mode = self._generalSettingsWidget.getRatioInputMode()
        is_wetherill = (mode == LeadLossImportSettings.RatioInputMode.WETHERILL)

        self._uPbWidget.setVisible(not is_wetherill)
        self._pbPbWidget.setVisible(not is_wetherill)

        self._pb207U235Widget.setVisible(is_wetherill)
        self._pb206U238Widget.setVisible(is_wetherill)

        # IMPORTANT: during initMainSettings(), okButton doesn't exist yet.
        # AbstractSettingsDialog calls _validate() after buttons are created.
        if hasattr(self, "okButton"):
            self._validate()


    ################
    ## Validation ##
    ################

    def _updateColumnRefs(self, newRefType):
        self._uPbWidget.changeColumnReferenceType(newRefType)
        self._pbPbWidget.changeColumnReferenceType(newRefType)
        self._pb207U235Widget.changeColumnReferenceType(newRefType)
        self._pb206U238Widget.changeColumnReferenceType(newRefType)
        self._sampleSettingsWidget.changeColumnReferenceType(newRefType)

    def _createSettings(self):
        settings = LeadLossImportSettings()
        settings.delimiter = self._generalSettingsWidget.getDelimiter()
        settings.hasHeaders = self._generalSettingsWidget.getHasHeaders()
        settings.columnReferenceType = self._generalSettingsWidget.getColumnReferenceType()
        settings.ratioInputMode = self._generalSettingsWidget.getRatioInputMode()

        settings.multipleSamples = self._sampleSettingsWidget.getMultipleSamples()

        settings._columnRefs = {
            Column.SAMPLE_NAME: self._sampleSettingsWidget.getSampleColumn(),

            # TW
            Column.U_PB_VALUE: self._uPbWidget.getValueColumn(),
            Column.U_PB_ERROR: self._uPbWidget.getErrorColumn(),
            Column.PB_PB_VALUE: self._pbPbWidget.getValueColumn(),
            Column.PB_PB_ERROR: self._pbPbWidget.getErrorColumn(),

            # Wetherill
            Column.PB207_U235_VALUE: self._pb207U235Widget.getValueColumn(),
            Column.PB207_U235_ERROR: self._pb207U235Widget.getErrorColumn(),
            Column.PB206_U238_VALUE: self._pb206U238Widget.getValueColumn(),
            Column.PB206_U238_ERROR: self._pb206U238Widget.getErrorColumn(),
        }

        settings.uPbErrorType = self._uPbWidget.getErrorType()
        settings.uPbErrorSigmas = self._uPbWidget.getErrorSigmas()
        settings.pbPbErrorType = self._pbPbWidget.getErrorType()
        settings.pbPbErrorSigmas = self._pbPbWidget.getErrorSigmas()

        # Wetherill settings (only used when ratioInputMode=WETHERILL)
        settings.pb207U235ErrorType = self._pb207U235Widget.getErrorType()
        settings.pb207U235ErrorSigmas = self._pb207U235Widget.getErrorSigmas()
        settings.pb206U238ErrorType = self._pb206U238Widget.getErrorType()
        settings.pb206U238ErrorSigmas = self._pb206U238Widget.getErrorSigmas()
        return settings

    def getWarning(self, settings):
        return None


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

        # Ratio input mode (TW vs Wetherill)
        self._ratioMode = QComboBox()
        self._ratioMode.addItem("Tera–Wasserburg (238U/206Pb, 207Pb/206Pb)", LeadLossImportSettings.RatioInputMode.TW)
        self._ratioMode.addItem("Wetherill (207Pb/235U, 206Pb/238U)", LeadLossImportSettings.RatioInputMode.WETHERILL)

        default_mode = getattr(defaultSettings, "ratioInputMode", LeadLossImportSettings.RatioInputMode.TW)
        if isinstance(default_mode, str):
            default_mode = LeadLossImportSettings.RatioInputMode.WETHERILL if "weth" in default_mode.lower() else LeadLossImportSettings.RatioInputMode.TW

        idx = 0 if default_mode == LeadLossImportSettings.RatioInputMode.TW else 1
        self._ratioMode.setCurrentIndex(idx)
        self._ratioMode.currentIndexChanged.connect(validation)
        self.ratioModeChanged = self._ratioMode.currentIndexChanged

        layout = QFormLayout()
        layout.setHorizontalSpacing(uiUtils.FORM_HORIZONTAL_SPACING)
        layout.addRow("File headers", self._hasHeadersCB)
        layout.addRow("Column separator", self._delimiterEntry)
        layout.addRow("Refer to columns by", self._columnRefType)
        layout.addRow("Ratios in CSV", self._ratioMode)
        self.setLayout(layout)

    def getHasHeaders(self):
        return self._hasHeadersCB.isChecked()

    def getDelimiter(self):
        return self._delimiterEntry.text()

    def getColumnReferenceType(self):
        return self._columnRefType.selection()

    def getRatioInputMode(self):
        return self._ratioMode.currentData()


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
