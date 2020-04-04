from utils import stringUtils
from utils.ui.radioButtons import RadioButtons


class ErrorSigmasInput(RadioButtons):

    def __init__(self, validation, defaultSigmas):
        default = stringUtils.get_error_sigmas_str(defaultSigmas)
        super().__init__(stringUtils.SIGMA_OPTIONS_STR, validation, default)

    def selection(self):
        return stringUtils.SIGMA_OPTIONS_STR.index(super().selection()) + 1