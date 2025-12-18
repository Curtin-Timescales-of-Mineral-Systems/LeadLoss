#!/usr/bin/env python3
"""
Summarize Reimink/DD results for Cases 1–4 (tiers a/b/c) *without* CDC.

Reads:
  REI_DIR/{case}{tier}_bootstrap_curves_boot200.csv
  REI_DIR/{case}{tier}_lowerdisc_curve_boot200.csv

Writes (to dd_out_cases1_4/):
  - dd_by_event.csv           (per-event rows)
  - dd_summary.csv            (tier-level summary)
  - dd_by_event.tex           (LaTeX table)
  - dd_summary.tex            (LaTeX table)
  - dd_app_like.csv / .tex    (optional, app-style 1 result per dataset)

Author: you
"""

from pathlib import Path
import numpy as np
import pandas as pd

# ----------------------- CONFIG -----------------------
REI_DIR     = Path("/Users/lucymathieson/Desktop/reimink_discordance_dating") 
OUT_DIR = Path("./dd_out_new")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Truth ages in Ma (edit if needed)
CASES_TRUE = {
    "1": [700],
    "2": [300, 1800],
    "3": [400],
    "4": [500, 1800],
}
TIERS = ["a", "b", "c"]

# Truth-based window clamp (Ma)
HALF_MIN = 50.0
HALF_CAP = 120.0

# Minimum number of bootstrap picks for using 95% CI
MIN_N_BOOT_95 = 5

# ---------------------- HELPERS -----------------------
def _to_ma(x):
    x = np.asarray(x, float)
    return (x/1e6) if np.nanmax(x) > 1e6 else x

def _truth_half_window_factory(true_ages, half_min=HALF_MIN, half_cap=HALF_CAP):
    t = np.array(sorted(true_ages), float)
    def half_for(tt):
        if t.size <= 1:
            return float(half_min)
        i = int(np.argmin(np.abs(t - tt)))
        left_gap  = (t[i] - t[i-1]) if i > 0 else np.inf
        right_gap = (t[i+1] - t[i]) if i < t.size - 1 else np.inf
        half = 0.5 * float(min(left_gap, right_gap))
        return float(min(half_cap, max(half_min, half)))
    return half_for

def load_reimink(case: str, tier: str, rei_dir: Path):
    """
    Returns (x_grid_Ma, boot_2d, y_med_norm)
      boot_2d shape: (n_boot, n_grid), each row normalized to peak=1
      y_med_norm: aggregate curve normalized to max=1, interpolated to x_grid if needed
    """
    boot_path = rei_dir / f"{case}{tier}_bootstrap_curves_boot200.csv"
    agg_path  = rei_dir / f"{case}{tier}_lowerdisc_curve_boot200.csv"
    if not boot_path.exists() or not agg_path.exists():
        raise FileNotFoundError(f"Missing Reimink files for {case}{tier}: {boot_path.name}, {agg_path.name}")

    boot_df = pd.read_csv(boot_path)
    agg_df  = pd.read_csv(agg_path)

    piv = (boot_df
           .pivot(index="Lower Intercept", columns="run.number",
                  values="normalized.sum.likelihood")
           .sort_index())
    x_boot = _to_ma(piv.index.values)
    boot   = piv.values.T  # (n_boot, n_grid)

    # normalize each bootstrap curve to peak=1
    with np.errstate(invalid="ignore"):
        rowmax = np.nanmax(boot, axis=1, keepdims=True)
        rowmax[~np.isfinite(rowmax)] = 1.0
        rowmax[rowmax <= 0] = 1.0
        boot = np.divide(boot, rowmax, out=np.zeros_like(boot), where=(rowmax > 0))

    x_agg = _to_ma(agg_df["Lower Intercept"].values)
    y_agg = agg_df["normalized.sum.likelihood"].astype(float).values
    ymax  = np.nanmax(y_agg) if np.isfinite(y_agg).any() else 1.0
    y_agg = (y_agg / ymax) if ymax > 0 else y_agg

    # Interpolate aggregate onto the bootstrap grid if needed
    if (x_agg.shape == x_boot.shape) and np.allclose(x_agg, x_boot):
        y_med = y_agg
    else:
        y_med = np.interp(x_boot, x_agg, y_agg, left=np.nan, right=np.nan)
        if np.isnan(y_med).any():
            # fallback: replace edge NaNs with bootstrap median
            y_med_alt = np.nanmedian(boot, axis=0)
            m = np.isnan(y_med)
            y_med[m] = y_med_alt[m]

    return x_boot, boot, y_med

def _bootstrap_max_ages(x_grid, boot_2d):
    """Return the age at the global maximum for each bootstrap curve."""
    valid = np.isfinite(boot_2d).any(axis=1)
    if not valid.any():
        return np.array([], float)
    boot_v = boot_2d[valid]
    with np.errstate(invalid="ignore"):
        j = np.nanargmax(np.where(np.isfinite(boot_v), boot_v, -np.inf), axis=1)
    return x_grid[j]

def _ci_from_samples(arr, step_guess):
    """Median and CI from picked ages with small-sample fallbacks."""
    arr = np.asarray(arr, float)
    med = float(np.median(arr))
    n   = arr.size
    if n >= MIN_N_BOOT_95:
        lo, hi = np.percentile(arr, [2.5, 97.5])
    elif n >= 3:
        lo, hi = np.percentile(arr, [5, 95])
    elif n == 2:
        lo, hi = med - step_guess, med + step_guess
    else:  # n == 1
        lo = hi = med
    return med, float(lo), float(hi), int(n)

def _grid_step(x):
    d = np.diff(np.asarray(x, float))
    return float(np.nanmedian(d)) if d.size else 10.0

# -------------------- MAIN LOGIC ----------------------
def per_event_rows_for_dataset(case: str, tier: str, true_ages, rei_dir: Path):
    x, boot, y_med = load_reimink(case, tier, rei_dir)
    picks = _bootstrap_max_ages(x, boot)
    step  = _grid_step(x)
    tol_for = _truth_half_window_factory(true_ages, HALF_MIN, HALF_CAP)

    rows = []
    for ev, atrue in enumerate(true_ages, 1):
        half = tol_for(atrue)
        sel  = picks[np.abs(picks - atrue) <= half]
        if sel.size == 0:
            rows.append(dict(case=case, tier=tier.upper(), event=ev,
                             true_age=float(atrue), age_med=np.nan,
                             ci_lo=np.nan, ci_hi=np.nan, n_runs=0))
            continue
        med, lo, hi, n = _ci_from_samples(sel, step)
        rows.append(dict(case=case, tier=tier.upper(), event=ev,
                         true_age=float(atrue), age_med=med,
                         ci_lo=lo, ci_hi=hi, n_runs=n))
    return rows

def app_like_row_for_dataset(case: str, tier: str, rei_dir: Path):
    x, boot, _ = load_reimink(case, tier, rei_dir)
    picks = _bootstrap_max_ages(x, boot)
    step  = _grid_step(x)
    if picks.size == 0:
        return dict(case=case, tier=tier.upper(),
                    age_med=np.nan, ci_lo=np.nan, ci_hi=np.nan, n_runs=0)
    med, lo, hi, n = _ci_from_samples(picks, step)
    return dict(case=case, tier=tier.upper(),
                age_med=med, ci_lo=lo, ci_hi=hi, n_runs=n)

def write_latex_by_event(df: pd.DataFrame, path: Path):
    df2 = (df.copy()
             .sort_values(["case","tier","event"])
             [["case","tier","event","true_age","age_med","ci_lo","ci_hi","n_runs","covers_truth","abs_bias"]])
    # neat formatting
    def f0(x): return "" if pd.isna(x) else f"{x:.0f}"
    def f1(x): return "" if pd.isna(x) else f"{x:.1f}"
    df2["true_age"] = df2["true_age"].map(f0)
    df2["age_med"]  = df2["age_med"].map(f0)
    df2["ci_lo"]    = df2["ci_lo"].map(f0)
    df2["ci_hi"]    = df2["ci_hi"].map(f0)
    df2["abs_bias"] = df2["abs_bias"].map(f0)
    df2["covers_truth"] = df2["covers_truth"].map(lambda v: "yes" if bool(v) else "no")
    cols = ["case","tier","event","true_age","age_med","ci_lo","ci_hi","n_runs","covers_truth","abs_bias"]
    with open(path, "w") as fh:
        fh.write(df2[cols].to_latex(index=False,
                                    column_format="llr r r r r r c r".replace(" ", ""),
                                    escape=False))

def write_latex_summary(df_sum: pd.DataFrame, path: Path):
    df2 = df_sum.copy().sort_values(["tier"])
    df2["cover_pct"] = (100.0*df2["cover"]).round(1)
    def f1(x): return "" if pd.isna(x) else f"{x:.1f}"
    def f0(x): return "" if pd.isna(x) else f"{x:.0f}"
    df2["mae"] = df2["mae"].map(f0)
    df2["max_abs"] = df2["max_abs"].map(f0)
    df2 = df2.rename(columns={"cover_pct":"coverage (%)"})
    with open(path, "w") as fh:
        fh.write(df2[["tier","mae","max_abs","coverage (%)"]]
                 .to_latex(index=False,
                           column_format="l r r r",
                           escape=False))

def main():
    # A) Collect per-event rows
    all_rows = []
    for case, truths in CASES_TRUE.items():
        for tier in TIERS:
            try:
                rows = per_event_rows_for_dataset(case, tier, truths, REI_DIR)
                all_rows.extend(rows)
            except FileNotFoundError as e:
                print(f"[skip] {e}")

    if not all_rows:
        raise SystemExit("No datasets found. Check REI_DIR and filenames.")

    df = pd.DataFrame(all_rows)
    df["bias"] = df["age_med"] - df["true_age"]
    df["abs_bias"] = df["bias"].abs()
    df["covers_truth"] = (df["ci_lo"] <= df["true_age"]) & (df["true_age"] <= df["ci_hi"])

    df_out = df.sort_values(["case","tier","event"])
    df_out.to_csv(OUT_DIR / "dd_by_event.csv", index=False)
    write_latex_by_event(df_out, OUT_DIR / "dd_by_event.tex")
    print("✓ wrote dd_by_event.csv / .tex")

    # B) Tier-level summary
    summary = (df_out.groupby(["tier"], as_index=False)
               .agg(mae=("abs_bias","median"),
                    max_abs=("abs_bias","max"),
                    cover=("covers_truth","mean")))
    summary.to_csv(OUT_DIR / "dd_summary.csv", index=False)
    write_latex_summary(summary, OUT_DIR / "dd_summary.tex")
    print("✓ wrote dd_summary.csv / .tex")

    # C) Optional: App-like one-per-dataset rows (no truth split)
    app_rows = []
    for case in CASES_TRUE:
        for tier in TIERS:
            try:
                app_rows.append(app_like_row_for_dataset(case, tier, REI_DIR))
            except FileNotFoundError:
                pass
    if app_rows:
        df_app = (pd.DataFrame(app_rows)
                    .sort_values(["case","tier"]))
        df_app.to_csv(OUT_DIR / "dd_app_like.csv", index=False)
        with open(OUT_DIR / "dd_app_like.tex", "w") as fh:
            fh.write(df_app[["case","tier","age_med","ci_lo","ci_hi","n_runs"]]
                     .to_latex(index=False,
                               column_format="l l r r r r",
                               escape=False))
        print("✓ wrote dd_app_like.csv / .tex")

if __name__ == "__main__":
    main()
