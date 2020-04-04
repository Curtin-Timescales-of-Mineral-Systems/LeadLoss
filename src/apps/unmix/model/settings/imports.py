from apps.abstract.model.column import ColumnSpec
from apps.unmix.model.column import Column
from utils import stringUtils

from apps.abstract.model.settings.imports import AbstractImportSettings
from apps.type import ApplicationType, SettingsType

class UnmixImportSettings(AbstractImportSettings):

    KEY = (ApplicationType.UNMIX, SettingsType.IMPORT)

    @staticmethod
    def getImportedColumnSpecs():
        return [
            ColumnSpec(Column.RIM_AGE_VALUE),
            ColumnSpec(Column.RIM_AGE_ERROR),
            ColumnSpec(Column.MIXED_U_PB_VALUE),
            ColumnSpec(Column.MIXED_U_PB_ERROR),
            ColumnSpec(Column.MIXED_PB_PB_VALUE),
            ColumnSpec(Column.MIXED_PB_PB_ERROR)
        ]

    def __init__(self):
        super().__init__(self.getImportedColumnSpecs())

        self.rimAgeErrorType = "Absolute"
        self.rimAgeErrorSigmas = 2

        self.mixedUPbErrorType = "Absolute"
        self.mixedUPbErrorSigmas = 2

        self.mixedPbPbErrorType = "Absolute"
        self.mixedPbPbErrorSigmas = 2

    ## Error getters ##

    def getRimAgeErrorStr(self):
        return stringUtils.get_error_str(self.rimAgeErrorSigmas, self.rimAgeErrorType)

    def getMixedUPbErrorStr(self):
        return stringUtils.get_error_str(self.mixedUPbErrorSigmas, self.mixedUPbErrorType)

    def getMixedPbPbErrorStr(self):
        return stringUtils.get_error_str(self.mixedPbPbErrorSigmas, self.mixedPbPbErrorType)

    ## Headers ##

    def getHeaders(self):
        return [
            "Rim age (Ma)",
            "±" + self.getRimAgeErrorStr(),
            "Mixed " + stringUtils.U_PB_STR,
            "±" + self.getMixedUPbErrorStr(),
            "Mixed " + stringUtils.PB_PB_STR,
            "±" + self.getMixedPbPbErrorStr()
        ]

    ## Validation ##

    def validate(self):
        return super().validate()
