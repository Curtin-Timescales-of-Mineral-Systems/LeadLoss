
from apps.type import ApplicationType, SettingsType

class LeadLossExportSettings():

    KEY = (ApplicationType.LEAD_LOSS, SettingsType.EXPORT)

    def __init__(self):
        outputSF = 3
