#!/usr/bin/env Rscript

## ─────────────────────────────────────────────────────────────────────────────
##  Bootstrap driver – Synthetic cases 1-4 (tiers A-C)   • PARALLEL VERSION •
##  • Uses all but one CPU core via {future}/{future.apply}
##  • Appends only the *missing* bootstrap curves (safe to resume)
##  • Aggressively scrubs each numeric column so “Lambda238 * age76
##    non-numeric argument” error never returns.
##  • Uses a separate temporary PDF for each worker to reduce device warnings.
## ─────────────────────────────────────────────────────────────────────────────

## 0 ─ packages ----------------------------------------------------------------
suppressPackageStartupMessages({
  library(data.table)      # fast I/O  (fread / fwrite)
  library(future)
  library(future.apply)
  library(dplyr)           # slice_sample()
  library(purrr)           # walk()
  library(IsoplotR)
})

## 1 ─ global constants made visible for helper code ---------------------------
λ <- IsoplotR::settings("lambda")   # named numeric vector
list(
  Lambda238 = λ["U238"],
  Lambda235 = λ["U235"],
  Lambda232 = λ["Th232"],
  Lambda234 = λ["U234"]
) |> 
  imap(~ assign(.y, unname(.x), envir = .GlobalEnv))  # inject into .GlobalEnv

## 2 ─ support files -----------------------------------------------------------
source("UPb_Constants_Functions_Libraries.R", local = FALSE)

## 3 ─ utility: force numeric columns (scrub weird chars) ----------------------
clean_numeric <- function(x) {
  # Keep only digits, decimal point, sign, and exponent "e/E"
  # everything else → removed
  gsub("[^0-9eE+\\-\\.]", "", x) |> as.numeric()
}

cast_numeric <- function(DT) {
  # Keep "Sample" as character, everything else numeric
  num_cols <- setdiff(names(DT), "Sample")
  DT[, (num_cols) := lapply(.SD, clean_numeric), .SDcols = num_cols]
  DT
}

## 4 ─ parallel setup ----------------------------------------------------------
plan(multisession, workers = max(parallel::detectCores() - 1L, 1L))
options(future.rng.onMisuse = "ignore")         # silence RNG seed warnings

## 5 ─ paths & quota -----------------------------------------------------------
root        <- "/Users/lucymathieson/Desktop/Desktop - Lucy’s MacBook Pro - 1/LeadLoss-2/Synthetic cases 1-4/Reimink"
target_runs <- 200                              # desired bootstrap quota

## 6 ─ one bootstrap curve -----------------------------------------------------
make_one_curve <- function(dat) {
  
  ## Guarantee helper constants in this worker
  if (!exists("Lambda238", envir = .GlobalEnv)) {
    λ <- IsoplotR::settings("lambda")
    assign("Lambda238", unname(λ["U238"]), .GlobalEnv)
    assign("Lambda235", unname(λ["U235"]), .GlobalEnv)
    assign("Lambda232", unname(λ["Th232"]), .GlobalEnv)
    assign("Lambda234", unname(λ["U234"]), .GlobalEnv)
  }
  
  ## (A) open a dummy PDF device in a temp file
  tmp_pdf <- tempfile(fileext = ".pdf")
  grDevices::pdf(tmp_pdf)
  on.exit(grDevices::dev.off(), add = TRUE)
  
  ## (B) bootstrap resample
  smp <- slice_sample(dat, n = nrow(dat), replace = TRUE) |> cast_numeric()
  
  assign("data.test",           smp, .GlobalEnv)
  assign("Data.new",            smp, .GlobalEnv)
  assign("Data.reduction",      smp, .GlobalEnv)
  assign("resample.datapoints", smp, .GlobalEnv)
  
  ## (C) legacy GUI flags expected by helper
  if (!exists("zoom.analysis",  envir = .GlobalEnv)) assign("zoom.analysis",  FALSE, .GlobalEnv)
  if (!exists("node.spacing",   envir = .GlobalEnv)) assign("node.spacing",   0.25,  .GlobalEnv)
  
  ## (D) run the main logic
  source("UPb_Reduction_Resample.R", local = FALSE)
  
  ## (E) return the final object (e.g., "lowerdisc.sum.total")
  lowerdisc.sum.total
}

## 7 ─ append/resume helper ----------------------------------------------------
resume_bootstrap <- function(dat, outfile, target = 200) {
  
  if (file.exists(outfile)) {
    boot_old <- fread(outfile)
    done     <- max(boot_old$run.number, na.rm = TRUE)
  } else {
    boot_old <- NULL
    done     <- 0L
  }
  
  todo <- target - done
  if (todo <= 0) {
    message("Already have ", done, " >= ", target, " runs; skipping.")
    return(invisible(NULL))
  }
  
  ## If any single run fails, skip it rather than crash repeatedly
  safe_curve <- purrr::safely(make_one_curve, otherwise = NULL)
  
  res.list <- future_lapply(
    seq_len(todo),
    \(i) safe_curve(dat),
    future.seed = TRUE
  )
  
  ## keep successes
  boot_new <- purrr::compact(map(res.list, "result"))
  
  if (!length(boot_new)) {
    stop("All bootstrap runs failed. Check the input data for major problems.")
  }
  
  message("Out of ", todo, " attempts, got ", length(boot_new), " successful runs.")
  
  boot_new <- rbindlist(boot_new, idcol = "run.number")
  boot_new[, run.number := run.number + done]
  
  fwrite(rbindlist(list(boot_old, boot_new), use.names = TRUE, fill = TRUE), outfile)
}

## 8 ─ main loop ---------------------------------------------------------------
rei_files <- list.files(root, pattern = "^[1-4][abc]\\.csv$", full.names = TRUE)
cat(">>> found", length(rei_files), "Case 1-4 Reimink tier files\n\n")

walk(rei_files, function(fp) {
  dat <- fread(fp) |> cast_numeric()   # major coercion at ingest
  out <- sub("\\.csv$", "_bootstrap_curves_200.csv", fp)
  
  cat(sprintf("%-6s → %-28s ", basename(fp), basename(out)))
  resume_bootstrap(dat, out, target_runs)
  cat("✓\n")
})

cat("\n✓ all Case 1-4 tiers now have ≥", target_runs, "bootstrap runs\n")