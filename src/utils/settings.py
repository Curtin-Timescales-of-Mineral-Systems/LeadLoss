import pickle
from os import path

from utils import stringUtils
from apps.leadLoss.model.settings.exports import LeadLossExportSettings
from apps.leadLoss.model.settings.imports import LeadLossImportSettings
from apps.leadLoss.model.settings.calculation import LeadLossCalculationSettings
from apps.unmix.model.settings.exports import UnmixExportSettings
from apps.unmix.model.settings.imports import UnmixImportSettings
from apps.unmix.model.settings.calculation import UnmixCalculationSettings


class Settings:
    __instance = None

    def __init__(self):
        self.contents = {
            UnmixImportSettings.KEY: UnmixImportSettings(),
            UnmixCalculationSettings.KEY: UnmixCalculationSettings(),
            UnmixExportSettings.KEY: UnmixExportSettings(),

            LeadLossImportSettings.KEY: LeadLossImportSettings(),
            LeadLossCalculationSettings.KEY: LeadLossCalculationSettings(),
            LeadLossExportSettings.KEY: LeadLossExportSettings(),
        }

    @classmethod
    def get(cls, tabType, settingsType):
        cls.__ensureInstance()
        return cls.__instance.contents[(tabType, settingsType)]
        # raise Exception("Unknown tab and settingsDialogs: " + type(tabType) + " " + type(settingsType))]

    @classmethod
    def update(cls, newSettings):
        cls.__ensureInstance()
        cls.__instance.contents[newSettings.KEY] = newSettings
        cls.__instance.__save()

    @classmethod
    def __ensureInstance(cls):
        if cls.__instance is None:
            cls.__instance = Settings()
            loadedInstance = cls.load()
            for key, value in loadedInstance.contents.items():
                cls.__instance.contents[key] = value
        return cls.__instance

    def __save(self):
        with open(stringUtils.SAVE_FILE, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load():
        if path.exists(stringUtils.SAVE_FILE):
            with open(stringUtils.SAVE_FILE, 'rb') as input:
                try:
                    return pickle.load(input)
                except Exception as e:
                    print(e)

        return Settings()
