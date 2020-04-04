import math

###############
## Constants ##
###############
from PyQt5.QtCore import QRegExp

U_PB_STR = "²³⁸U/²⁰⁶Pb"
PB_PB_STR = "²⁰⁷Pb/²⁰⁶Pb"

ERROR_SIGMA_OPTIONS = [2, 1]
ERROR_TYPE_OPTIONS = ["Absolute", "Percentage"]
DISCORDANCE_OPTIONS = ["Percentages", "Error ellipse"]

SAVE_FILE = "../concordia_save_data.pkl"

###############
## Functions ##
###############

def getConstantStr(constant):#
    return '%.6g' % constant

def getUPbStr(useSuperscripts):
    if useSuperscripts:
        return U_PB_STR
    return "238U/206Pb"

def getPbPbStr(useSuperscripts):
    if useSuperscripts:
        return PB_PB_STR
    return "207Pb/206Pb"

def print_warning(message):
    print("\033[93m" + message + "\033[0m")

def round_to_sf(x, sf=2):
    if isinstance(x, str):
        try:
            x = float(x)
        except ValueError:
            return x

    if x == 0:
        return 0
    return round(x, sf-int(math.floor(math.log10(abs(x))))-1)

def print_progress_bar (iteration, total, prefix = 'Progress', suffix = '', decimals = 1, length = 50, fill = '█', printEnd = "\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    if iteration == total: 
        print()

def get_error_sigmas_str(sigmas):
    return str(sigmas) + "σ"

def get_error_str(sigmas, type):
    return get_error_sigmas_str(sigmas) + " " + error_symbol(type, brackets=True)

def error_symbol(type, brackets=False):
    if type != "Percentage":
        return ""
    if brackets:
        return "(%)"
    return "%"