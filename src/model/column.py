from enum import Enum


class Column(Enum):
    SAMPLE_NAME = 0
    U_PB_VALUE = 1
    U_PB_ERROR = 2
    PB_PB_VALUE = 3
    PB_PB_ERROR = 4
    # Optional Wetherill input columns
    #   x = 207Pb/235U
    #   y = 206Pb/238U
    PB207_U235_VALUE = 5
    PB207_U235_ERROR = 6
    PB206_U238_VALUE = 7
    PB206_U238_ERROR = 8
