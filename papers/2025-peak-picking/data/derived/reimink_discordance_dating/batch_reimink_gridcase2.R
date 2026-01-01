#!/usr/bin/env Rscript

## Usage:
##   Rscript reimink_runlog.R /path/to/Reimink_main.R
## (no edits to the Reimink script required)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  stop("Provide the path to the Reimink script, e.g. Rscript reimink_runlog.R /path/to/main.R")
}
reimink_script <- args[[1]]

# Where to append the row (same path you use for CDC)
RUNLOG     <- Sys.getenv("RUNLOG",
                         "/Users/lucymathieson/Desktop/Peak-Picking-Manuscript-Python/runtime_log.csv")
RUN_FIELDS <- c("method","phase","sample","tier","R","n_grid","elapsed_s",
                "per_run_median_s","per_run_p95_s","rss_peak_mb","python","numpy")

infer_tier <- function(s) {
  s <- trimws(s)
  if (nchar(s) && grepl("[A-Za-z]$", s)) toupper(sub(".*([A-Za-z])$","\\1", s)) else ""
}

append_runlog <- function(sample.name, subsample.iteration, lowerdisc.sum.total, elapsed_s) {
  # Try to get grid length; if not available, leave blank
  n_grid <- tryCatch(length(lowerdisc.sum.total$`Lower Intercept`), error = function(e) "")
  tier   <- infer_tier(sample.name)
  
  row <- data.frame(
    method           = "Reimink",
    phase            = "e2e_runtime",
    sample           = sample.name,
    tier             = tier,
    R                = subsample.iteration,
    n_grid           = n_grid,
    elapsed_s        = round(elapsed_s, 3),
    per_run_median_s = "",
    per_run_p95_s    = "",
    rss_peak_mb      = "",   # left blank on purpose (no code changes inside)
    python           = "",
    numpy            = "",
    stringsAsFactors = FALSE
  )
  
  header <- !file.exists(RUNLOG)
  dir.create(dirname(RUNLOG), showWarnings = FALSE, recursive = TRUE)
  utils::write.table(row[, RUN_FIELDS], RUNLOG, sep = ",", row.names = FALSE,
                     col.names = header, append = !header, quote = FALSE)
}

t_start <- Sys.time()

## Run your Reimink script UNCHANGED; its objects will appear in the global env
## so we can read sample.name, subsample.iteration, lowerdisc.sum.total afterwards.
source(reimink_script, local = FALSE)

elapsed_s <- as.numeric(difftime(Sys.time(), t_start, units = "secs"))

## These are created by your script:
##   - sample.name
##   - subsample.iteration
##   - lowerdisc.sum.total
append_runlog(sample.name, subsample.iteration, lowerdisc.sum.total, elapsed_s)

cat(sprintf("[runlog] appended: sample=%s R=%s elapsed=%.3fs -> %s\n",
            sample.name, subsample.iteration, elapsed_s, RUNLOG))
