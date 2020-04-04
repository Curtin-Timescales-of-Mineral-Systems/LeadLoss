from apps.abstract.model.settings.exports import AbstractExportSettings
from apps.type import ApplicationType, SettingsType


class UnmixExportSettings(AbstractExportSettings):
    KEY = (ApplicationType.UNMIX, SettingsType.EXPORT)

