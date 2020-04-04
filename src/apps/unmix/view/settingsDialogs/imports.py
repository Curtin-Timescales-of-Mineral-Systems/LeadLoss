from PyQt5.QtWidgets import *

from apps.unmix.model.column import Column
from apps.unmix.model.settings.imports import UnmixImportSettings
from apps.abstract.view.settingsDialogs.imports import AbstractImportSettingsDialog, ImportedValueErrorWidget, \
    GeneralSettingsWidget

from utils import stringUtils


class UnmixImportSettingsDialog(AbstractImportSettingsDialog):

    def __init__(self, defaultSettings, *args, **kwargs):
        super().__init__(defaultSettings, *args, **kwargs)

    ###############
    ## UI layout ##
    ###############

    def initMainSettings(self):
        # Defaults
        defaults = self.defaultSettings
        columnRefs = self.defaultSettings.getDisplayColumnsByRefs()

        self._generalSettingsWidget = GeneralSettingsWidget(self._validate, defaults)
        self._generalSettingsWidget.columnRefChanged.connect(self._onColumnRefChange)

        self._rimAgeWidget = ImportedValueErrorWidget(
            "Rim age",
            self._validate,
            defaults.columnReferenceType,
            columnRefs[Column.RIM_AGE_VALUE],
            columnRefs[Column.RIM_AGE_ERROR],
            defaults.rimAgeErrorType,
            defaults.rimAgeErrorSigmas
        )

        self._mixedUPbWidget = ImportedValueErrorWidget(
            "Mixed " + stringUtils.getUPbStr(True),
            self._validate,
            defaults.columnReferenceType,
            columnRefs[Column.MIXED_U_PB_VALUE],
            columnRefs[Column.MIXED_U_PB_ERROR],
            defaults.mixedUPbErrorType,
            defaults.mixedUPbErrorSigmas
        )

        self._mixedPbPbWidget = ImportedValueErrorWidget(
            "Mixed " + stringUtils.getPbPbStr(True),
            self._validate,
            defaults.columnReferenceType,
            columnRefs[Column.MIXED_PB_PB_VALUE],
            columnRefs[Column.MIXED_PB_PB_ERROR],
            defaults.mixedPbPbErrorType,
            defaults.mixedPbPbErrorSigmas
        )

        self._updateColumnRefs(defaults.columnReferenceType)

        layout = QGridLayout()
        layout.setHorizontalSpacing(15)
        layout.setVerticalSpacing(15)
        layout.addWidget(self._generalSettingsWidget, 0, 0)
        layout.addWidget(self._rimAgeWidget, 0, 1)
        layout.addWidget(self._mixedUPbWidget, 1, 0)
        layout.addWidget(self._mixedPbPbWidget, 1, 1)

        widget = QWidget()
        widget.setLayout(layout)
        return widget

    ################
    ## Validation ##
    ################

    def _updateColumnRefs(self, newRefType):
        self._rimAgeWidget.changeColumnReferenceType(newRefType)
        self._mixedUPbWidget.changeColumnReferenceType(newRefType)
        self._mixedPbPbWidget.changeColumnReferenceType(newRefType)

    def _createSettings(self):
        settings = UnmixImportSettings()
        settings.delimiter = self._generalSettingsWidget.getDelimiter()
        settings.hasHeaders = self._generalSettingsWidget.getHasHeaders()
        settings.columnReferenceType = self._generalSettingsWidget.getColumnReferenceType()

        settings._columnRefs = {
            Column.RIM_AGE_VALUE: self._rimAgeWidget.getValueColumn(),
            Column.RIM_AGE_ERROR: self._rimAgeWidget.getErrorColumn(),
            Column.MIXED_U_PB_VALUE: self._mixedUPbWidget.getValueColumn(),
            Column.MIXED_U_PB_ERROR: self._mixedUPbWidget.getErrorColumn(),
            Column.MIXED_PB_PB_VALUE: self._mixedPbPbWidget.getValueColumn(),
            Column.MIXED_PB_PB_ERROR: self._mixedPbPbWidget.getErrorColumn()
        }

        settings.rimAgeErrorType = self._rimAgeWidget.getErrorType()
        settings.rimAgeErrorSigmas = self._rimAgeWidget.getErrorSigmas()
        settings.mixedUPbErrorType = self._mixedUPbWidget.getErrorType()
        settings.mixedUPbErrorSigmas = self._mixedUPbWidget.getErrorSigmas()
        settings.mixedPbPbErrorType = self._mixedPbPbWidget.getErrorType()
        settings.mixedPbPbErrorSigmas = self._mixedPbPbWidget.getErrorSigmas()

        return settings
