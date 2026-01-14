## ======================================================================
## Reimink discordance dating: RUNLOG-ONLY with visible progress
## ======================================================================

setwd("/Users/lucymathieson/Desktop/reimink_discordance_dating")

options(stringsAsFactors = FALSE, warn = 1)  # print warnings immediately

library(IsoplotR)
library(dplyr)
# No ggplot2 here (avoids graphics warnings)

source("UPb_Constants_Functions_Libraries.R")

## ------------------------- RUNLOG HELPERS -------------------------------
RUNLOG <- Sys.getenv("RUNLOG")
if (RUNLOG == "") RUNLOG <- "/Users/lucymathieson/Desktop/Peak-Picking-Manuscript-Python/runtime_log_reimink_2.csv"
dir.create(dirname(RUNLOG), recursive = TRUE, showWarnings = FALSE)

.parse_case_tier <- function(x) {
  m  <- regexec("^\\s*([0-9]+)\\s*([A-Za-z]+)\\s*$", x)
  mt <- regmatches(x, m)[[1]]
  case <- if (length(mt) >= 2) mt[2] else NA_character_
  tier <- if (length(mt) >= 3) tolower(mt[3]) else NA_character_
  list(case = case, tier = tier)
}

log_line <- function(step, status = "ok", note = "", elapsed_sec = NA_real_) {
  sname <- if (exists("sample.name", inherits = TRUE)) get("sample.name", inherits = TRUE) else NA_character_
  pt    <- if (is.na(sname)) list(case = NA_character_, tier = NA_character_) else .parse_case_tier(sname)
  ts <- format(Sys.time(), "%Y-%m-%d %H:%M:%S%z")
  row <- data.frame(
    timestamp   = ts,
    case        = pt$case,
    tier        = pt$tier,
    sample      = sname,
    step        = step,
    elapsed_sec = ifelse(is.na(elapsed_sec), NA_real_, round(as.numeric(elapsed_sec), 3)),
    status      = status,
    note        = as.character(note),
    stringsAsFactors = FALSE
  )
  write.table(row, RUNLOG, sep = ",", row.names = FALSE,
              col.names = !file.exists(RUNLOG), append = TRUE)
}

time_it <- function(step, expr) {
  t0 <- Sys.time()
  status <- "ok"; note <- ""
  on.exit({
    dt <- difftime(Sys.time(), t0, units = "secs")
    log_line(step, status = status, note = note, elapsed_sec = as.numeric(dt))
  }, add = TRUE)
  tryCatch(
    eval.parent(substitute(expr)),
    error = function(e) { status <<- "error"; note <<- conditionMessage(e); stop(e) }
  )
}
## ----------------------- END RUNLOG HELPERS -----------------------------

## <<< CHANGE THIS EACH RUN
sample.name <- "1b"   # e.g., "1a", "1b", "1c", "2a", ... "4c", "5c"

## Quick sanity: input file must exist where you're running
input_csv <- file.path(getwd(), paste0(sample.name, ".csv"))
if (!file.exists(input_csv)) {
  avail <- list.files(getwd(), pattern = "\\.csv$", ignore.case = TRUE)
  stop(sprintf("Input CSV not found:\n  %s\nWorking dir: %s\nCSV files here (%d): %s",
               input_csv, getwd(), length(avail), paste(avail, collapse = ", ")))
}

overall_t0 <- Sys.time()
status_all <- "ok"; note_all <- ""
log_line("script_start")

# visible console cue
message(sprintf("[%s] %s: started", format(Sys.time(), "%H:%M:%S"), sample.name)); flush.console()

tryCatch({
  
  ## --------------------- READ + PREP ---------------------
  data.raw <- time_it("read_input", read.csv(input_csv))
  
  data.concordia <- time_it("isoplot_read_data",
                            IsoplotR::read.data(data.raw[, 2:6],
                                                ierr = 2, method = "U-Pb", format = 1))
  
  invisible(time_it("isoplot_concordia",
                    IsoplotR::concordia(data.concordia, type = 1)))
  
  ## --------------------- SETTINGS -----------------------
  node.spacing <- 10
  subsample.iteration <- 200        # adjust if you want a quicker test run
  LOG_EVERY <- 10                   # print a console tick every N iterations
  
  ## --------------------- BOOTSTRAP LOOP -----------------
  # progress bar for visible progress
  pb <- txtProgressBar(min = 0, max = subsample.iteration, style = 3)
  on.exit(close(pb), add = TRUE)
  
  # We don't keep big results; we just do the heavy step to time it.
  for (j in 1:subsample.iteration) {
    time_it(sprintf("bootstrap_%03d", j), {
      data.test <- dplyr::slice_sample(.data = data.raw, n = nrow(data.raw), replace = TRUE)
      # heavy calculation; creates lowerdisc.sum.total, but we don't store it
      source("UPb_Reduction_Resample.R")
      invisible(NULL)
    })
    setTxtProgressBar(pb, j)
    if (j == 1 || (j %% LOG_EVERY) == 0 || j == subsample.iteration) {
      message(sprintf("[%s] %s: bootstrap %d/%d",
                      format(Sys.time(), "%H:%M:%S"), sample.name, j, subsample.iteration))
      flush.console()
      # optional lightweight marker in the runlog for human readability
      log_line("progress", note = sprintf("bootstrap %d/%d", j, subsample.iteration))
    }
  }
  
  ## --------------------- MAIN REDUCTION ------------------
  time_it("main_reduction", {
    data.test <- data.raw
    source("UPb_Reduction_Resample.R")
  })
  
}, error = function(e) {
  status_all <<- "error"; note_all <<- conditionMessage(e)
  log_line("script_error", status = status_all, note = note_all)
  stop(e)
}, finally = {
  log_line("script_end",
           status = status_all,
           note = note_all,
           elapsed_sec = as.numeric(difftime(Sys.time(), overall_t0, units = "secs")))
  message(sprintf("[%s] %s: done", format(Sys.time(), "%H:%M:%S"), sample.name)); flush.console()
})
