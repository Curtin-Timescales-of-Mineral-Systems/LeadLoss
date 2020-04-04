from apps.abstract.model.column import ColumnSpec
from apps.leadLoss.model.column import Column
from utils import stringUtils
from apps.abstract.model.settings.imports import AbstractImportSettings

from apps.type import ApplicationType, SettingsType


class LeadLossImportSettings(AbstractImportSettings):

    KEY = (ApplicationType.LEAD_LOSS, SettingsType.IMPORT)

    @staticmethod
    def getImportedColumnSpecs():
        return [
            ColumnSpec(Column.U_PB_VALUE),
            ColumnSpec(Column.U_PB_ERROR),
            ColumnSpec(Column.PB_PB_VALUE),
            ColumnSpec(Column.PB_PB_ERROR),
        ]

    def __init__(self):
        super().__init__(LeadLossImportSettings.getImportedColumnSpecs())

        self.uPbErrorType = "Absolute"
        self.uPbErrorSigmas = 2

        self.pbPbErrorType = "Absolute"
        self.pbPbErrorSigmas = 2

        self.discordanceThreshold = 0.1

    def getUPbErrorStr(self):
        return stringUtils.get_error_str(self.uPbErrorSigmas, self.uPbErrorType)

    def getPbPbErrorStr(self):
        return stringUtils.get_error_str(self.pbPbErrorSigmas, self.pbPbErrorType)

    def getHeaders(self):
        return [
            stringUtils.U_PB_STR,
            "±" + self.getUPbErrorStr(),
            stringUtils.PB_PB_STR,
            "±" + self.getPbPbErrorStr()
        ]


    def validate(self):
        return super().validate()

