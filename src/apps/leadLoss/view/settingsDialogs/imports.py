from PyQt5.QtWidgets import QGridLayout, QWidget

from apps.leadLoss.model.column import Column
from utils import stringUtils
from apps.abstract.view.settingsDialogs.imports import AbstractImportSettingsDialog, ImportedValueErrorWidget, \
    GeneralSettingsWidget
from apps.leadLoss.model.settings.imports import LeadLossImportSettings


class LeadLossImportSettingsDialog(AbstractImportSettingsDialog):

    def __init__(self, defaultSettings, *args, **kwargs):
        super().__init__(defaultSettings, *args, **kwargs)

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

        print(settings._columnRefs)

        settings.uPbErrorType = self._uPbWidget.getErrorType()
        settings.uPbErrorSigmas = self._uPbWidget.getErrorSigmas()
        settings.pbPbErrorType = self._pbPbWidget.getErrorType()
        settings.pbPbErrorSigmas = self._pbPbWidget.getErrorSigmas()
        return settings