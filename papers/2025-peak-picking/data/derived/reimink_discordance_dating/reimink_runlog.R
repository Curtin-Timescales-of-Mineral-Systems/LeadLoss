#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly=TRUE)
if (length(args) < 1) stop("Usage: Rscript reimink_runlog.R /path/to/main.R [optional_sample_name]")
reimink_script <- args[[1]]

## (optional) allow passing a sample name without editing the main script
if (length(args) >= 2) assign("sample.name", args[[2]], envir=.GlobalEnv)

RUNLOG     <- Sys.getenv("RUNLOG","/Users/lucymathieson/Desktop/Peak-Picking-Manuscript-Python/runtime_log.csv")
RUN_FIELDS <- c("method","phase","sample","tier","R","n_grid","elapsed_s",
                "per_run_median_s","per_run_p95_s","rss_peak_mb","python","numpy")

infer_tier <- function(s){
  s <- trimws(as.character(s))
  if (nchar(s) && grepl("[A-Za-z]$", s)) toupper(sub(".*([A-Za-z])$","\\1", s)) else ""
}
rss_mb <- function() {
  if (requireNamespace("ps", quietly = TRUE)) {
    info <- try(ps::ps_memory_info(ps::ps_handle(Sys.getpid())), silent = TRUE)
    if (!inherits(info, "try-error")) {
      val <- tryCatch(unname(as.numeric(info[["rss"]])), error = function(e) NA_real_)
      if (is.finite(val)) return(val / (1024^2))  # MB
    }
  }
  if (.Platform$OS.type == "unix") {
    v <- suppressWarnings(system(sprintf("ps -o rss= -p %d", Sys.getpid()), intern = TRUE))
    return(suppressWarnings(as.numeric(v) / 1024))  # MB
  }
  NA_real_
}

safe_get <- function(sym, default = NULL) {
  if (exists(sym, envir = .GlobalEnv, inherits = FALSE)) get(sym, envir = .GlobalEnv) else default
}

append_runlog <- function(sample.name, subsample.iteration, lowerdisc.sum.total,
                          elapsed_s, per_med = NA_real_, per_p95 = NA_real_, rss = NA_real_) {
  n_grid <- tryCatch(length(lowerdisc.sum.total$`Lower Intercept`), error = function(e) "")
  tier   <- infer_tier(sample.name)
  row <- data.frame(
    method="Reimink", phase="e2e_runtime", sample=sample.name, tier=tier,
    R=subsample.iteration, n_grid=n_grid, elapsed_s=round(elapsed_s,3),
    per_run_median_s=ifelse(is.na(per_med),"",round(per_med,3)),
    per_run_p95_s   =ifelse(is.na(per_p95),"",round(per_p95,3)),
    rss_peak_mb     =ifelse(is.na(rss),"",round(rss,1)),
    python="", numpy="", stringsAsFactors=FALSE
  )
  header <- !file.exists(RUNLOG)
  dir.create(dirname(RUNLOG), showWarnings=FALSE, recursive=TRUE)
  utils::write.table(row[, RUN_FIELDS], RUNLOG, sep=",", row.names=FALSE,
                     col.names=header, append=!header, quote=FALSE)
}

t_start <- Sys.time()
status <- "ok"; err_msg <- NA_character_

## Run your Reimink script UNCHANGED; its objects will appear in .GlobalEnv
tryCatch(
  source(reimink_script, local = FALSE),
  error = function(e) { status <<- "error"; err_msg <<- conditionMessage(e) }
)

elapsed_s <- as.numeric(difftime(Sys.time(), t_start, units="secs"))

## Pull optional metrics out of the global env if present
iter_times            <- safe_get("iter_times", numeric(0))
subsample.iteration   <- safe_get("subsample.iteration", NA_integer_)
lowerdisc.sum.total   <- safe_get("lowerdisc.sum.total", NULL)
sample.name           <- safe_get("sample.name", "")

per_med <- if (length(iter_times)) stats::median(iter_times, na.rm=TRUE) else NA_real_
per_p95 <- if (length(iter_times)) as.numeric(stats::quantile(iter_times, 0.95, na.rm=TRUE, type=7)) else NA_real_
work_s  <- if (length(iter_times)) sum(iter_times, na.rm=TRUE) else NA_real_

append_runlog(sample.name, subsample.iteration, lowerdisc.sum.total,
              elapsed_s = elapsed_s, per_med = per_med, per_p95 = per_p95, rss = rss_mb())

cat(sprintf("[runlog] sample=%s R=%s wall=%.1fs work=%.1fs med=%.3fs p95=%.3fs -> %s\n",
            sample.name, as.character(subsample.iteration),
            elapsed_s, ifelse(is.na(work_s), NA_real_, work_s),
            ifelse(is.na(per_med), NA_real_, per_med),
            ifelse(is.na(per_p95), NA_real_, per_p95),
            RUNLOG))

if (identical(status, "error")) {
  message(sprintf("Reimink script failed: %s", err_msg))
  quit(status = 1L)
}
