from apps.abstract.view.helpDialog import AbstractHelpDialog
from utils import config


class LeadLossHelpDialog(AbstractHelpDialog):
    def __init__(self):
        super().__init__()

    def getTitle(self):
        return config.LEAD_LOSS_TITLE

    def getInputsHelpText(self):
        return \
            "Input data required: <ul>" \
            "<li> measured ²³⁸U/²⁰⁶Pb and ²⁰⁷Pb/²⁰⁶Pb ratios" \
            "<li> uncertainties for all of the above" \
            "</ul>" + \
            self._getStandardInputHelp()

    def getProcessingHelpText(self):
        return \
            "PROCESSING HELP" \
            "<br><br>" + \
            self._getStandardProcessingHelp()

    def getOutputsHelpText(self):
        return \
            "OUTPUTS HELP" \
            "<br><br>" + \
            self._getStandardOutputsHelp()
