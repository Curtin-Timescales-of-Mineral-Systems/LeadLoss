from utils.settings import Settings
from apps.abstract.view.view import AbstractView
from apps.type import ApplicationType, SettingsType
from apps.unmix.view.graphPanel import UnmixGraphPanel
from apps.unmix.view.dataPanel import UnmixDataPanel
from apps.unmix.view.settingsDialogs.calculation import UnmixCalculationSettingsDialog

from apps.unmix.view.settingsDialogs.imports import UnmixImportSettingsDialog


class UnmixView(AbstractView):

    def __init__(self, controller):
        super().__init__(controller)

    def createDataWidget(self):
        return UnmixDataPanel(self.controller)

    def createGraphWidget(self):
        return UnmixGraphPanel(self.controller)

    def getSettingsDialog(self, settingsType):
        defaultSettings = Settings.get(ApplicationType.UNMIX, settingsType)
        if settingsType == SettingsType.IMPORT:
            return UnmixImportSettingsDialog(defaultSettings)
        if settingsType == SettingsType.CALCULATION:
            return UnmixCalculationSettingsDialog(defaultSettings)

        raise Exception("Unknown settingsDialogs " + str(type(settingsType)))