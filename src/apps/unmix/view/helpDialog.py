from apps.abstract.view.helpDialog import AbstractHelpDialog
from utils import config


class UnmixHelpDialog(AbstractHelpDialog):
    def __init__(self):
        super().__init__()

    def getTitle(self):
        return config.U_PB_UNMIXER_TITLE

    def getInputsHelpText(self):
        return \
            "Input data required: <ul>" \
            "<li> known age (in Ma) of rim (younger) component in mixture" \
            "<li> measured ²³⁸U/²⁰⁶Pb and ²⁰⁷Pb/²⁰⁶Pb ratios" \
            "<li> uncertainties for all of the above" \
            "</ul>" + \
            self._getStandardInputHelp()

    def getProcessingHelpText(self):
        return \
            "Processing the data will attempt to calculate a reconstructed core age with associated uncertainty values." \
            "<br><br>" \
            "Uncertainties are calculated using second order error propagation. " \
            "Due to the non-linearity of the concordia curve, the resulting uncertainties are not symmetric. " \
            "In particular the upper uncertainty will be larger than the lower uncertainty." \
            "<br><br>" \
            "Rows with successful calculations will be highlighted in <font color='green'>GREEN</font>." \
            "<br><br>" \
            "Rows with partially successful calculations will be highlighted in <font color='orange'>ORANGE</font>." \
            "<br><br>"  + \
            self._getStandardProcessingHelp()

    def getOutputsHelpText(self):
        return \
            "The following values are output:" \
            "<ul>" \
            "<li> a reconstructed core age (in Ma)" \
            "<li> reconstructed core ²³⁸U/²⁰⁶Pb and ²⁰⁷Pb/²⁰⁶Pb ratios " \
            "<li> uncertainties for all of the above" \
            "</ul>" \
            "Clicking on an individual row permits visual inspection of the full solution " \
            "on the inverse Concordia plot." \
            "<br><br>" \
            "Selecting multiple rows permits visual inspection of the final calculated ages of " \
            "all the rows in the selection." \
            "<br><br>"  + \
            self._getStandardOutputsHelp()