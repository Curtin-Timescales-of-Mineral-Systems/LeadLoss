setwd("/Users/lucymathieson/Desktop/reimink_discordance_dating")
#setwd("/Users/lucymathieson/Desktop")
############################# U-Pb modeling inputs  #############################
library(IsoplotR)
library(dplyr)
library(ggplot2)

source("UPb_Constants_Functions_Libraries.R")

## read in the data file and save the sample name
sample.name <- "6a" ## set the sample name

## where to save the new 200-bootstrap results
out_dir <- "/Users/lucymathieson/Desktop/alta_out"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

data.raw <- read.csv(paste(sample.name, ".csv", sep = "")) # read in sample file

## read data into IsoplotR and do concordia calcs
data.concordia <- IsoplotR::read.data(data.raw[, 2:6], ierr = 2, method = "U-Pb", format = 1)
concordia <- IsoplotR::concordia(data.concordia, type = 1)

## now run our modeling reduction for the dataset
## Define node spacing
node.spacing <- 10

#############################  SWITCHES #############################
## normalize uncertainties to the median value?
normalize.uncertainty <- "N"

## detrital weighting against concordant analyses?
data.type <- "single"

## trim the input data by ratio cuts?
cut.data.by.ratios <- "N"
startcut.r75 <- 0
endcut.r75   <- 20
startcut.r68 <- 0
endcut.r68   <- 0.8

## zoom the analysis into a certain age window (saves time outside)
zoom.analysis <- "Y"
startcut.age.lower <- 1     ## Ma
endcut.age.lower   <- 2500  ## Ma
startcut.age.upper <- 1     ## Ma
endcut.age.upper   <- 4000  ## Ma

## bootstrap resampling
subsample.iteration <- 200

## storage
resample.datapoints    <- list() # store bootstrapped U-Pb data
resample.results.data  <- list() # store results from discordance dating

## resampling loop (no timing / runlog)
for (j in 1:subsample.iteration) {
  data.test <- dplyr::slice_sample(.data = data.raw, n = nrow(data.raw), replace = TRUE)
  resample.datapoints[[j]] <- data.test
  
  source("UPb_Reduction_Resample.R")   # heavy step; creates lowerdisc.sum.total
  resample.results.data[[j]] <- lowerdisc.sum.total
  
  print(j)
}

## now use the resampled data and plot it
## peak finding
list.maxposition <- lapply(resample.results.data, function(df) which.max(df$normalized.sum.likelihood))
list.maxlikelihood <- lapply(resample.results.data, function(df) max(df$normalized.sum.likelihood))
data.max.age <- data.frame(max.position = unlist(list.maxposition),
                           max.likelihood = unlist(list.maxlikelihood))
data.max.age$age <- data.max.age$max.position * node.spacing + startcut.age.lower

## expand all bootstrap curves into one data.frame
resampled.results.expanded <- bind_rows(resample.results.data, .id = "column_label")
resampled.results.expanded$run.number <- resampled.results.expanded$column_label

## reimport raw data and do the reduction once on the original data
data.test <- data.raw
source("UPb_Reduction_Resample.R")  ## produces lowerdisc.sum.total for the original data

## build peak labels for the main (non-bootstrap) curve
ages <- lowerdisc.sum.total$`Lower Intercept`/1e6
liks <- lowerdisc.sum.total$normalized.sum.likelihood
n    <- length(liks)

## global max
gm_idx <- which.max(liks)

## interior local maxima by three-point test
if (n >= 3) {
  interior_idx <- 2:(n - 1)
  left  <- liks[interior_idx] > liks[interior_idx - 1]
  right <- liks[interior_idx] > liks[interior_idx + 1]
  int_peaks <- interior_idx[left & right]
} else {
  int_peaks <- integer(0)
}
all_peaks <- unique(c(gm_idx, int_peaks))

## keep only peaks with age > 1 Ma
valid_peaks <- all_peaks[ages[all_peaks] > 1]

peaks.df <- data.frame(
  age        = ages[valid_peaks],
  likelihood = liks[valid_peaks]
)

print(peaks.df)

## plot: all bootstrap curves (light), main curve (black), and peak markers
ggplot(bind_rows(resample.results.data, .id = "data_frame"),
       aes(x = `Lower Intercept`/1e6, y = normalized.sum.likelihood, color = data_frame)) +
  geom_line(show.legend = FALSE, alpha = 0.2) +
  geom_line(data = lowerdisc.sum.total,
            aes(x = `Lower Intercept`/1e6, y = normalized.sum.likelihood),
            color = "black", size = 1) +
  geom_vline(data = peaks.df, aes(xintercept = age), colour = "red", linetype = "dashed") +
  geom_point(data = peaks.df, aes(x = age, y = likelihood), color = "red", size = 2) +
  geom_text(data = peaks.df,
            aes(x = age, y = likelihood, label = paste0(round(age, 0), " Ma")),
            vjust = -0.6, color = "red", size = 3) +
  labs(x = "Intercept Age (Ma)", y = "Normalized Likelihood") +
  fte_theme_white() +
  scale_colour_grey(start = 0.3, end = 0.31)

## histogram of bootstrap maximum ages
ggplot(data.max.age, aes(x = age)) +
  geom_histogram(bins = 50) +
  xlim(0, 2000) +
  xlab("Maximum Age (Ma)") +
  ylab("Number") +
  fte_theme_white()

## scatter of bootstrap maximum vs likelihood
ggplot(data.max.age, aes(x = age, y = max.likelihood)) +
  geom_point() +
  xlim(0, 2000) +
  xlab("Maximum Age (Ma)") +
  ylab("Likelihood") +
  fte_theme_white()

## ── export CSVs ─────────────────────────────────────────────────────
csv_main <- file.path(out_dir, paste0(sample.name, "_lowerdisc_curve_boot200.csv"))
write.csv(lowerdisc.sum.total, csv_main, row.names = FALSE)

csv_boot <- file.path(out_dir, paste0(sample.name, "_bootstrap_curves_boot200.csv"))
write.csv(resampled.results.expanded, csv_boot, row.names = FALSE)
