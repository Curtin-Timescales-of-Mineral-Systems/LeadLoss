# ─────────────────────────────────────────────────────────────────────────────
#  Batch driver – Case-2 parameter sweep   (Δ × W × Tier) • PARALLEL VERSION •
# ─────────────────────────────────────────────────────────────────────────────
#  • One baseline lower-disc curve + 200 bootstrap curves per tier
#  • Resumes cleanly if a *_bootstrap_curves.csv already exists
#  • Uses all but one CPU core via {future}/{future.apply}
# ─────────────────────────────────────────────────────────────────────────────

suppressPackageStartupMessages({
  library(data.table)    # fast fread / fwrite
  library(dplyr)         # slice_sample
  library(purrr)         # walk
  library(future)
  library(future.apply)
  library(IsoplotR)
})

## 1 ─ constants that legacy helper expects in .GlobalEnv ---------------------
λ <- IsoplotR::settings("lambda")        # named numeric vector
list(
  Lambda238 = λ["U238"],
  Lambda235 = λ["U235"],
  Lambda232 = λ["Th232"],
  Lambda234 = λ["U234"]
) |>
  imap(~ assign(.y, unname(.x), envir = .GlobalEnv))

## 2 ─ external helper code ---------------------------------------------------
source("UPb_Constants_Functions_Libraries.R", local = FALSE)  # unchanged

## 3 ─ numeric coercion helper (avoids char → numeric errors) -----------------
cast_numeric <- function(DT) {
  num_cols <- setdiff(names(DT), "Sample")   # leave Sample character
  DT[, (num_cols) := lapply(.SD, \(x) as.numeric(trimws(x))),
     .SDcols = num_cols]
  DT
}

## 4 ─ parallel setup ---------------------------------------------------------
plan(multisession, workers = max(parallel::detectCores() - 1L, 1L))
options(future.rng.onMisuse = "ignore")      # silence RNG seed warnings

## 5 ─ user paths & settings --------------------------------------------------
root_dir   <- "/Users/lucymathieson/Desktop/Desktop - Lucy’s MacBook Pro - 1/LeadLoss-2/Synthetic case 2 parameter sweep"
B          <- 200      ## target bootstrap replicates per tier

## 6 ─ one bootstrap curve ----------------------------------------------------
make_curve <- function(dat) {
  
  ## make sure constants exist inside this worker ---------------------------
  if (!exists("Lambda238", envir = .GlobalEnv)) {
    λ <- IsoplotR::settings("lambda")
    assign("Lambda238", unname(λ["U238"]), .GlobalEnv)
    assign("Lambda235", unname(λ["U235"]), .GlobalEnv)
    assign("Lambda232", unname(λ["Th232"]), .GlobalEnv)
    assign("Lambda234", unname(λ["U234"]), .GlobalEnv)
  }
  
  ## headless graphics device (IsoplotR won’t pop up) -----------------------
  grDevices::pdf(NULL); on.exit(grDevices::dev.off(), add = TRUE)
  
  ## build IsoplotR internals (no plot) -------------------------------------
  IsoplotR::concordia(
    IsoplotR::read.data(dat[, 2:6], ierr = 2, method = "U-Pb", format = 1),
    type = 1
  )
  Lambda238 <- as.numeric(Lambda238) 
  
  ## legacy helper expects these globals ------------------------------------
  assign("data.test",           dat, .GlobalEnv)
  assign("Data.new",            dat, .GlobalEnv)
  assign("Data.reduction",      dat, .GlobalEnv)
  assign("resample.datapoints", dat, .GlobalEnv)
  
  ## legacy switches – create harmless defaults if absent -------------------
  if (!exists("zoom.analysis", envir = .GlobalEnv))
    assign("zoom.analysis", FALSE, .GlobalEnv)
  if (!exists("node.spacing",  envir = .GlobalEnv))
    assign("node.spacing", 0.25,  .GlobalEnv)
  
  ## run the old reduction script ------------------------------------------
  source("UPb_Reduction_Resample.R", local = FALSE)
  
  lowerdisc.sum.total           # <- returned to caller
}

## 7 ─ helper: append missing bootstrap curves -------------------------------
resume_bootstrap <- function(dat, outfile, target = 200) {
  
  if (file.exists(outfile)) {
    boot_old <- fread(outfile)
    done     <- max(boot_old$run.number, na.rm = TRUE)
  } else {
    boot_old <- NULL
    done     <- 0L
  }
  
  todo <- target - done
  if (todo <= 0) return(invisible(NULL))
  
  ## ── NEW: progressor that knows how many tasks we’ll run ────────────────
  p <- progressr::progressor(steps = todo)
  
  res.list <- future_lapply(
    seq_len(todo),
    future.seed = TRUE,
    FUN = function(i) {
      ## tick the bar *inside* the worker
      p(sprintf("bootstrap %d / %d", i, todo))
      
      smp <- slice_sample(dat, n = nrow(dat), replace = TRUE) |> cast_numeric()
      make_curve(smp)
    }
  )
  
  boot_new <- rbindlist(res.list, idcol = "run.number")
  boot_new[, run.number := run.number + done]
  
  fwrite(rbindlist(list(boot_old, boot_new), use.names = TRUE), outfile)
}

## 8 ─ main driver ------------------------------------------------------------
rei_files <- list.files(root_dir, pattern = "_Rei\\.csv$", recursive = TRUE,
                        full.names = TRUE)

cat(">>> Found", length(rei_files), "Reimink panels for Case-2 sweep\n\n")

walk(rei_files, function(fp) {
  
  panel   <- sub("_Rei\\.csv$", "", basename(fp))          # Case2_d600_W70_30
  dat_all <- cast_numeric(fread(fp))                       # whole panel
  
  for (samp in unique(dat_all$Sample)) {        # 2A / 2B / 2C
    dat <- dat_all[Sample == samp] |> cast_numeric()   # ← add cast_numeric()
    if (nrow(dat) == 0) next
    
    tag     <- paste0(panel, "_Tier", substr(samp, 2, 2))  # …_TierA/B/C
    out_dir <- dirname(fp)
    lk_csv  <- file.path(out_dir, paste0(tag, "_lowerdisc_curve.csv"))
    bt_csv  <- file.path(out_dir, paste0(tag, "_bootstrap_curves.csv"))
    
    ## write baseline (once) ------------------------------------------------
    if (!file.exists(lk_csv))
      fwrite(make_curve(dat), lk_csv) 
    
    ## bootstrap top-up (parallel) -----------------------------------------
    cat(sprintf("%-35s → %-28s ", basename(fp), basename(bt_csv)))
    resume_bootstrap(dat, bt_csv, B)
    cat("✓\n")
  }
}, .progress = "time")

cat("\n✓ Case-2 sweep: every tier now has ≥", B, "bootstraps\n")