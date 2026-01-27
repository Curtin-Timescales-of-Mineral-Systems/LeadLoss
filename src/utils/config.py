######################
## General settings ##
######################

LEAD_LOSS_TITLE = "Pb-loss"
VERSION = "2.0"

# Size of the confidence interval -- 1 or 2
SIGMAS_MIXED_POINT_ERROR = 1
SIGMAS_RIM_AGE_ERROR = 1
SIGMAS_OUTPUT_ERROR = 2

# How the error is specified -- either "Absolute" or "Percentage"
ERROR_TYPE_MIXED_POINT = "Percentage"
ERROR_TYPE_RIM_AGE = "Percentage"
ERROR_TYPE_OUTPUT = "Absolute"

# Other
DISPLAY_SF = 5
CONSTANT_SF = 6

##################
## CSV dialogs ##
##################

# Delimiter
CSV_DELIMITER = ","

# Column for the data in the CSV file
# These can either be numbers (with the first column = 0) or letters (A, B, C, up to Z)
COLUMN_MIXED_POINT_U238Pb206 = "H"
COLUMN_MIXED_POINT_U238Pb206_ERROR = "I"
COLUMN_MIXED_POINT_Pb207Pb206 = "J"
COLUMN_MIXED_POINT_Pb207Pb206_ERROR = "K"
COLUMN_RIM_AGE = "S"
COLUMN_RIM_AGE_ERROR = "T"

# Column used to generate the name for the figures when run in CSV mode using the `-f` flag
COLUMN_SAMPLE_NAME = 0

##################
## GUI dialogs ##
##################

# Graph labels
LABEL_RIM_AGE = "Rim age (Ma)"
LABEL_RIM_AGE_ERROR = "Rim age (Ma) error"
LABEL_U238Pb206 = "238U/206Pb"
LABEL_U238Pb206_ERROR = LABEL_U238Pb206 + " error"
LABEL_Pb207Pb206 = "207Pb/206Pb"
LABEL_Pb207Pb206_ERROR = LABEL_Pb207Pb206 + " error"
LABEL_RECONSTRUCTED_AGE = "Reconstructed age"
LABEL_RECONSTRUCTED_AGE_ERROR = "Reconstructed age error"

# Graph colors

INVALID_COLOUR_255              = (239,  68,  68,  70)   # soft red highlight (Qt table background)

UNCLASSIFIED_COLOUR_255         = (153, 153, 153, 140)   # grey
CONCORDANT_COLOUR_255           = (  0, 158, 115, 140)   # bluish green
DISCORDANT_COLOUR_255           = (230, 159,   0, 140)   # orange
REVERSE_DISCORDANT_COLOUR_255   = (213,  94,   0, 140)   # vermillion (NOT pure red)
OPTIMAL_COLOUR_255              = (204, 121, 167, 200)   # purple (Pb-loss age)
PREDICTION_COLOUR_255           = ( 86, 180, 233, 180)   # sky blue (predicted/selected markers)

# Matplotlib RGB (0..1)
UNCLASSIFIED_COLOUR_1       = tuple(v/255.0 for v in UNCLASSIFIED_COLOUR_255[:3])
CONCORDANT_COLOUR_1         = tuple(v/255.0 for v in CONCORDANT_COLOUR_255[:3])
DISCORDANT_COLOUR_1         = tuple(v/255.0 for v in DISCORDANT_COLOUR_255[:3])
REVERSE_DISCORDANT_COLOUR_1 = tuple(v/255.0 for v in REVERSE_DISCORDANT_COLOUR_255[:3])
OPTIMAL_COLOUR_1            = tuple(v/255.0 for v in OPTIMAL_COLOUR_255[:3])
PREDICTION_COLOUR_1         = tuple(v/255.0 for v in PREDICTION_COLOUR_255[:3])

HEATMAP_RESOLUTION = 100