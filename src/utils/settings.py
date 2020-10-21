import pickle
from os import path

from utils import stringUtils
from model.settings.exports import LeadLossExportSettings
from model.settings.imports import LeadLossImportSettings
from model.settings.calculation import LeadLossCalculationSettings


import pickle
from os import path

from model.settings.type import SettingsType
from utils import stringUtils, config


class Settings:
    __instance = None
    __currentFile = None

    def __init__(self):
        self.per_file_settings = {}
        self.version = config.VERSION

    def __ensureCompatibility(self):
        if self.version is None: # Pre v0.5 or before
            self.per_file_settings = {}
        self.version = config.VERSION

    @classmethod
    def setCurrentFile(cls, file):
        cls.__currentFile = file
        cls.__ensureInstance()
        if file not in cls.__instance.per_file_settings:
            cls.__instance.per_file_settings[file] = {
                SettingsType.IMPORT: LeadLossImportSettings(),
                SettingsType.CALCULATION: LeadLossCalculationSettings(),
                SettingsType.EXPORT: LeadLossExportSettings()
            }

    @classmethod
    def get(cls, settingsType):
        cls.__ensureInstance()
        per_file_settings = cls.__instance.per_file_settings
        return per_file_settings[cls.__currentFile][settingsType]

    @classmethod
    def update(cls, newSettings):
        cls.__ensureInstance()
        cls.__instance.per_file_settings[cls.__currentFile][newSettings.KEY] = newSettings
        cls.__instance.__save()

    @classmethod
    def __ensureInstance(cls):
        if cls.__instance is None:
            cls.__instance = Settings()
            loadedInstance = cls.load()
            for file, perFileSettings in loadedInstance.per_file_settings.items():
                cls.__instance.per_file_settings[file] = perFileSettings
        return cls.__instance

    def __save(self):
        with open(stringUtils.SAVE_FILE, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load():
        if path.exists(stringUtils.SAVE_FILE):
            with open(stringUtils.SAVE_FILE, 'rb') as inputFile:
                try:
                    result = pickle.load(inputFile)
                    result.__ensureCompatibility()
                    return result
                except Exception as e:
                    print(e)

        return Settings()
"""
class Settings:
    __instance = None

    def __init__(self):
        self.contents = {
            LeadLossImportSettings.KEY: LeadLossImportSettings(),
            LeadLossCalculationSettings.KEY: LeadLossCalculationSettings(),
            LeadLossExportSettings.KEY: LeadLossExportSettings(),
        }

    @classmethod
    def get(cls, settingsType):
        cls.__ensureInstance()
        return cls.__instance.contents[settingsType]

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
"""