from apps.abstract.view.settingsDialogs.general import AbstractSettingsDialog


class AbstractCalculationSettingsDialog(AbstractSettingsDialog):

    def __init__(self, defaultSettings, *args, **kwargs):
        super().__init__(defaultSettings, *args, **kwargs)
        self.setWindowTitle("Calculation settings")
