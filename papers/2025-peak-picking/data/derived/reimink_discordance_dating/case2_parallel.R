# Case-2 parameter sweep – PARALLEL & optimised
# ------------------------------------------------
# * 1 baseline lower-disc curve + 200 bootstraps per tier
# * Tier-level parallelism via future/furrr
# * Graphics disabled (plot = FALSE)
# * Single write per tier (data.table::fwrite)
# ------------------------------------------------

suppressPackageStartupMessages({
  library(IsoplotR)
  library(dplyr)
  library(data.table)
  library(future)
  library(furrr)
  library(filelock)
  library(compiler)
})

## ---- user settings ------------------------------------------------------
root_dir    <- "/Users/lucymathieson/Desktop/Desktop - Lucy’s MacBook Pro - 1/LeadLoss-2/Synthetic case 2 parameter sweep"   
target_runs <- 200
cores       <- max(1, parallel::detectCores() - 1)
file_suffix <- "_parallel"          # << everything we create ends in this
options(future.globals.maxSize = Inf)

## ---- shared constants ---------------------------------------------------
# Define critical globals *before* sourcing helper scripts so they exist at
# parse‑time if the files reference them at top level.
CONST_ENV <- list(
  zoom.analysis      = FALSE,   # plotting off, still defined
  node.spacing       = 10,
  startcut.age.lower = 1,
  endcut.age.lower   = 2500,
  startcut.age.upper = 1,
  endcut.age.upper   = 4000
)
list2env(CONST_ENV, envir = .GlobalEnv)   # set once for master

# Now it is safe to source helpers — they can see zoom.analysis, etc.
source("UPb_Constants_Functions_Libraries.R", local = FALSE)

## ---- concordia wrapper --------------------------------------------------
make_curve <- function(dat){
  list2env(CONST_ENV, envir = .GlobalEnv)          # make sure globals exist
  
  # 1· concordia numerics (no plotting)
  cc <- IsoplotR::concordia(
    IsoplotR::read.data(dat[, 2:6], ierr = 2,
                        method = "U-Pb", format = 1),
    type = 1, plot = FALSE)
  
  # 2· objects the helper expects
  assign("z",                   cc$z, .GlobalEnv)
  assign("data.test",           dat,  .GlobalEnv)
  assign("Data.new",            dat,  .GlobalEnv)
  assign("Data.reduction",      dat,  .GlobalEnv)
  assign("resample.datapoints", dat,  .GlobalEnv)
  
  # 3· **RE-RUN** the reduction for this bootstrap
  source("UPb_Reduction_Resample.R", local = FALSE)
  
  lowerdisc.sum.total           # fresh curve returned to caller
}

make_curve <- compiler::cmpfun(.make_curve)
rm(.make_curve)
## ------------------------------------------------------------------------
resume_bootstrap <- function(dat, outfile, target = 200){
  lockfile <- paste0(outfile, ".lock")
  lock     <- filelock::lock(lockfile, timeout = 0)
  if (is.null(lock)) return(invisible(NULL))
  on.exit(filelock::unlock(lock), add = TRUE)
  
  if (file.exists(outfile)){
    boot_old <- fread(outfile)
    boot_old[, run.number := as.integer(run.number)]
    done <- max(boot_old$run.number, na.rm = TRUE)
  } else {
    boot_old <- NULL
    done     <- 0L
  }
  todo <- target - done
  if (todo <= 0) return(invisible(NULL))
  
  res.list <- vector("list", todo)
  for (j in seq_len(todo))
    res.list[[j]] <- make_curve(dat[sample(.N, .N, replace = TRUE)])
  
  boot_new <- rbindlist(res.list, idcol = "run.number")
  boot_new[, run.number := as.integer(run.number) + done]
  fwrite(rbindlist(list(boot_old, boot_new)), outfile)
}
## ------------------------------------------------------------------------

## ---- per‑file worker ----------------------------------------------------
run_file <- function(fp){
  # replicate environment in each worker **before** sourcing helpers
  list2env(CONST_ENV, envir = .GlobalEnv)
  library(IsoplotR)
  source("UPb_Constants_Functions_Libraries.R", local = FALSE)

  panel   <- sub("_Rei.csv$", "", basename(fp))
  dat_all <- fread(fp)
  
  for (samp in unique(dat_all$Sample)){
    dat <- dat_all[Sample == samp]
    if (nrow(dat) == 0) next
    
    tag     <- paste0(panel, "_Tier", substr(samp, 2, 2))
    out_dir <- dirname(fp)
    lk_csv  <- file.path(out_dir, paste0(tag, "_lowerdisc_curve", file_suffix, ".csv"))
    bt_csv  <- file.path(out_dir, paste0(tag, "_bootstrap_curves", file_suffix, ".csv"))
    
    if (!file.exists(lk_csv)) fwrite(make_curve(dat), lk_csv)
    
    resume_bootstrap(dat, bt_csv, target_runs)
  }
}
## ------------------------------------------------------------------------


