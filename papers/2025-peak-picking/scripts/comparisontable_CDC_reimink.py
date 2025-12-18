#!/usr/bin/env python3
import os, re
from pathlib import Path
import numpy as np
import pandas as pd

# ── config ───────────────────────────────────────────────────────────────────
RUNLOG_FILE = Path("/Users/lucymathieson/Desktop/LeadLossOutputs/runtime_log_28Nov.csv")
REI_DIR     = Path("/Users/lucymathieson/Desktop/reimink_discordance_dating")
CAT_BASE    = Path("/Users/lucymathieson/Desktop/Peak Picking Manuscript Files/Cases 1-7 Pb loss Outputs/ensemble_catalogue_28Nov.csv")


CDC_R_DEFAULT = 200
CDC_RESULTS_SURFACE = os.environ.get("CDC_RESULTS_SURFACE", "PEN").strip().upper()
DD_ASSIGN = os.environ.get("DD_ASSIGN", "WINDOW").strip().upper()   # WINDOW|NEAREST
ONLY_CASES = set(s for s in os.environ.get("ONLY_CASES", "1,2,3,4,5,6,7").replace(" ","").split(",") if s)

ADAPTIVE_MIN, ADAPTIVE_CAP = 50.0, 120.0
MIN_N_BOOT = 5
TIERS = ["a","b","c"]

CASES_TRUE = {
    "1":[700],
    "2":[300,1800],
    "3":[400],
    "4":[500,1800],
    "5":[500,1500],
    "6":[500,1500],
    "7":[500,1500],
}

OUT_DIR = Path(f"./Dec_comparison_out_{CDC_RESULTS_SURFACE.lower()}_{DD_ASSIGN.lower()}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── optional SciPy (required for PEAKS widths). Nice error if missing. ───────
try:
    from scipy.signal import find_peaks, peak_widths
    _HAVE_SCIPY = True
except Exception as e:
    _HAVE_SCIPY = False
    print("[warn] SciPy not available; Reimink(PEAKS) will be skipped.")
    # You can `pip install scipy` in the same Python env to enable it.

# ── small helpers ────────────────────────────────────────────────────────────
def _norm_tag(s: str) -> str:
    t = str(s).strip().upper()
    return re.split(r'[^0-9A-Z]+', t)[0]

def _cdc_r_for_tag(tag: str) -> int:
    try:
        df = pd.read_csv(RUNLOG_FILE)
        mc = df[df["phase"].astype(str).str.upper().eq("MC")]
        if mc.empty: return CDC_R_DEFAULT
        tagcol = mc["sample"].astype(str).str.upper().str.replace(r'[^0-9A-Z]+.*', '', regex=True)
        r_val = pd.to_numeric(mc.loc[tagcol.eq(tag), "R"], errors="coerce").dropna()
        return int(r_val.iloc[-1]) if not r_val.empty else CDC_R_DEFAULT
    except Exception:
        return CDC_R_DEFAULT

def _pick_catalogue_file() -> Path:
    p = CAT_BASE
    if p.exists(): return p
    # fallback variants
    cands = [
        p.with_name(p.name.replace(".csv","_pen.csv")),
        p.with_name(p.name.replace(".csv","_new.csv")),
        p.with_name(p.name.replace(".csv","_raw.csv")),
        p.with_name(p.name.replace(".csv","_np.csv")),
    ]
    for q in cands:
        if q.exists(): return q
    raise SystemExit(f"[CDC] no catalogue found. Tried: {p}, {', '.join(map(str,cands))}")

def load_catalogue_table(path: Path):
    if not path.exists(): raise SystemExit(f"[CDC] catalogue not found: {path}")
    df = pd.read_csv(path, comment="#", encoding="utf-8-sig")
    cols = {c.lower(): c for c in df.columns}
    for k in ["sample","age_ma","ci_low","ci_high"]:
        if k not in cols:
            raise SystemExit(f"[CDC] catalogue missing column: {k} (got {list(df.columns)})")
    df = df.rename(columns={cols[c]: c for c in cols})
    for k in ["age_ma","ci_low","ci_high","support"]:
        if k in df.columns: df[k] = pd.to_numeric(df[k], errors="coerce")
    out = {}
    for _, r in df.dropna(subset=["sample","age_ma","ci_low","ci_high"]).iterrows():
        tag = _norm_tag(r["sample"])
        out.setdefault(tag, []).append(dict(
            age=float(r["age_ma"]), lo=float(r["ci_low"]), hi=float(r["ci_high"]),
            support=(float(r["support"]) if "support" in df.columns and pd.notna(r["support"]) else np.nan)
        ))
    for k in out: out[k].sort(key=lambda e: e["age"])
    print(f"[CDC] loaded catalogue entries for {len(out)} sample tags from {path.name}")
    return out

def _to_Ma(x):
    x = np.asarray(x, float);  return (x/1e6) if np.nanmax(x) > 1e6 else x

def load_reimink(case, tier):
    boot_path = REI_DIR / f"{case}{tier}_bootstrap_curves_boot200.csv"
    agg_path  = REI_DIR / f"{case}{tier}_lowerdisc_curve_boot200.csv"
    if not boot_path.exists() or not agg_path.exists():
        raise FileNotFoundError(f"Missing {case}{tier}: boot={boot_path.exists()} agg={agg_path.exists()}")
    boot_df = pd.read_csv(boot_path)
    agg_df  = pd.read_csv(agg_path)
    piv = (boot_df.pivot(index="Lower Intercept", columns="run.number",
                         values="normalized.sum.likelihood").sort_index())
    x = _to_Ma(piv.index.values)
    boot = piv.values.T
    with np.errstate(invalid="ignore", divide="ignore"):
        mx = np.nanmax(boot, axis=1, keepdims=True);  mx[~np.isfinite(mx)] = 1.0;  mx[mx<=0]=1.0
        boot = np.divide(boot, mx, out=np.zeros_like(boot), where=(mx>0))
    x2 = _to_Ma(agg_df["Lower Intercept"].values)
    y  = agg_df["normalized.sum.likelihood"].astype(float).values
    ymax = np.nanmax(y) if np.isfinite(y).any() else 1.0;  y = (y/ymax) if ymax>0 else y
    if (x2.shape == x.shape) and np.allclose(x2, x):
        y_med = y
    else:
        y_med = np.interp(x, x2, y, left=np.nan, right=np.nan)
        if np.isnan(y_med).any():
            y_med_alt = np.nanmedian(boot, axis=0);  y_med[np.isnan(y_med)] = y_med_alt[np.isnan(y_med)]
    return x, boot, y_med

def _grid_step(x):
    d = np.diff(np.asarray(x,float))
    return float(np.nanmedian(d)) if d.size else 10.0

def _tol_half_window_truth_factory(true_ages, default_min=50.0, cap=120.0):
    t = np.array(sorted(true_ages), float)
    def half(tt):
        if t.size <= 1: return float(default_min)
        i = int(np.argmin(np.abs(t - tt)))
        left = (t[i]-t[i-1]) if i>0 else np.inf
        right= (t[i+1]-t[i]) if i<t.size-1 else np.inf
        return float(min(cap, max(default_min, 0.5*min(left,right))))
    return half

def _fwhm_window(x, y_med, x0, min_ma=ADAPTIVE_MIN, cap_ma=ADAPTIVE_CAP):
    if not _HAVE_SCIPY: return min_ma
    pk, _ = find_peaks(y_med, prominence=0.02, width=3)
    if pk.size==0: return min_ma
    j = pk[np.argmin(np.abs(x[pk]-x0))]
    widths, *_ = peak_widths(y_med, [j], rel_height=0.5)
    hw = 0.5*float(widths[0])*_grid_step(x)
    return float(min(cap_ma, max(min_ma, hw)))

# ── DD estimators ────────────────────────────────────────────────────────────
def reimink_event_stats(case, tier, true_ages, assign_mode="WINDOW", tol_for=None):
    x, boot, y_med = load_reimink(case, tier)
    valid = np.isfinite(boot).any(axis=1);  boot_v = boot[valid]
    if boot_v.size==0: return [], (x,y_med)
    with np.errstate(invalid="ignore"):
        idx = np.nanargmax(np.where(np.isfinite(boot_v), boot_v, -np.inf), axis=1)
    x_at_max = x[idx];  rows=[];  step=_grid_step(x)
    if assign_mode.upper()=="NEAREST":
        dists = np.column_stack([np.abs(x_at_max - a) for a in true_ages])
        owners = np.argmin(dists, axis=1)
        for ev, atrue in enumerate(true_ages,1):
            picks = x_at_max[owners==(ev-1)]
            rows.extend(_row_from_picks("Reimink", case, tier, ev, atrue, picks, step))
        return rows, (x,y_med)
    # WINDOW: keep maxima within tol
    for ev, atrue in enumerate(true_ages,1):
        half = float(tol_for(atrue)) if tol_for is not None else _fwhm_window(x,y_med,atrue)
        picks = x_at_max[np.abs(x_at_max-atrue)<=half]
        rows.extend(_row_from_picks("Reimink", case, tier, ev, atrue, picks, step))
    return rows, (x,y_med)

def _row_from_picks(method, case, tier, ev, atrue, picks, step):
    if picks.size==0:
        return [dict(method=method, case=case, tier=tier.upper(), event=ev,
                     true_age=atrue, age_med=np.nan, ci_lo=np.nan, ci_hi=np.nan, n_runs=0)]
    med = float(np.median(picks))
    if picks.size >= MIN_N_BOOT:
        lo,hi = np.percentile(picks, [2.5,97.5])
    elif picks.size >= 3:
        lo,hi = np.percentile(picks, [5,95])
    elif picks.size == 2:
        lo,hi = med-step, med+step
    else:
        lo=hi=med
    return [dict(method=method, case=case, tier=tier.upper(), event=ev,
                 true_age=atrue, age_med=med, ci_lo=float(lo), ci_hi=float(hi),
                 n_runs=int(picks.size))]

# PEAKS variant (requires SciPy); safe no-op if SciPy missing
def reimink_peaks_event_stats(case, tier, true_ages, tol_for=None):
    if not _HAVE_SCIPY: return []
    x, boot, y_med = load_reimink(case, tier)
    boot_safe = np.where(np.isfinite(boot), boot, -np.inf)
    idx = np.argmax(boot_safe, axis=1)
    x_max = x[idx]; x_max = x_max[np.isfinite(x_max)]
    pk, props = find_peaks(y_med, prominence=0.02, width=3)
    if pk.size==0: 
        return [dict(method="Reimink(PEAKS)", case=case, tier=tier.upper(), event=ev,
                     true_age=atrue, age_med=np.nan, ci_lo=np.nan, ci_hi=np.nan, n_runs=np.nan)
                for ev, atrue in enumerate(true_ages,1)]
    widths, *_ = peak_widths(y_med, pk, rel_height=0.5)
    dx = _grid_step(x)
    rows=[]
    for ev, atrue in enumerate(true_ages,1):
        half_truth = float(tol_for(atrue)) if tol_for is not None else _fwhm_window(x,y_med,atrue)
        # among all median peaks, pick the one whose center is within truth window and closest to truth
        candidates=[]
        for j, w in zip(pk, widths):
            center = float(x[j]); half = 0.5*float(w)*dx
            picks = x_max[np.abs(x_max-center) <= half]
            if picks.size >= 2 and abs(center-atrue) <= half_truth:
                med = float(np.median(picks))
                if picks.size >= 5: lo,hi = np.percentile(picks,[2.5,97.5])
                elif picks.size == 2: lo,hi = sorted(picks)
                else: lo=hi=med
                candidates.append((abs(center-atrue),
                                   dict(method="Reimink(PEAKS)", case=case, tier=tier.upper(), event=ev,
                                        true_age=atrue, age_med=med, ci_lo=float(lo), ci_hi=float(hi),
                                        n_runs=np.nan, support=picks.size/len(x_max))))
        if candidates:
            rows.append(min(candidates, key=lambda t:t[0])[1])
        else:
            rows.append(dict(method="Reimink(PEAKS)", case=case, tier=tier.upper(), event=ev,
                             true_age=atrue, age_med=np.nan, ci_lo=np.nan, ci_hi=np.nan, n_runs=np.nan))
    return rows

def cdc_catalogue_event_stats(case, tier, true_ages, cat_map, tol_for=None):
    tag = f"{case}{tier.upper()}"; entries = cat_map.get(tag, [])
    rows=[]
    for ev, atrue in enumerate(true_ages,1):
        if not entries:
            rows.append(dict(method="CDC", case=case, tier=tier.upper(), event=ev,
                             true_age=atrue, age_med=np.nan, ci_lo=np.nan, ci_hi=np.nan, n_runs=np.nan))
            continue
        if tol_for is not None:
            half = float(tol_for(atrue))
            cand = [e for e in entries if abs(e["age"]-atrue) <= half]
            if not cand:
                rows.append(dict(method="CDC", case=case, tier=tier.upper(), event=ev,
                                 true_age=atrue, age_med=np.nan, ci_lo=np.nan, ci_hi=np.nan, n_runs=np.nan))
                continue
            e = min(cand, key=lambda ee: abs(ee["age"]-atrue))
        else:
            e = min(entries, key=lambda ee: abs(ee["age"]-atrue))
        rows.append(dict(method="CDC", case=case, tier=tier.upper(), event=ev,
                         true_age=atrue, age_med=e["age"], ci_lo=e["lo"], ci_hi=e["hi"], n_runs=np.nan))
    return rows

def _cdc_reported_ages_for_dataset(case, tier, cat_map):
    return [e["age"] for e in cat_map.get(f"{case}{tier.upper()}", [])]

def _greedy_match(true_ages, est_ages, tol_for):
    t = np.array(sorted(true_ages), float); e = np.array(sorted(est_ages), float)
    used = np.zeros(e.size, bool); matches=[]
    for tt in t:
        tol=float(tol_for(tt)); idx = np.where(~used & (np.abs(e-tt)<=tol))[0]
        if idx.size:
            j = idx[np.argmin(np.abs(e[idx]-tt))]; matches.append((tt,e[j])); used[j]=True
    miss = [tt for tt in t if all(abs(tt-ee) > tol_for(tt) for ee in e)]
    extra = e[~used].tolist()
    return matches, miss, extra

def reimink_app_like(case, tier):
    x, boot, _ = load_reimink(case, tier)
    valid = np.isfinite(boot).any(axis=1); boot_v = boot[valid]
    if boot_v.size==0:
        return dict(method="Reimink(APP)", case=case, tier=tier.upper(),
                    event=1, true_age=np.nan, age_med=np.nan, ci_lo=np.nan, ci_hi=np.nan, n_runs=0)
    with np.errstate(invalid="ignore"):
        idx = np.nanargmax(np.where(np.isfinite(boot_v), boot_v, -np.inf), axis=1)
    ages = x[idx]; med = float(np.median(ages))
    if ages.size >= 5: lo,hi = np.percentile(ages,[2.5,97.5])
    elif ages.size == 2: step=_grid_step(x); lo,hi = med-step, med+step
    else: lo=hi=med
    return dict(method="Reimink(APP)", case=case, tier=tier.upper(),
                event=1, true_age=np.nan, age_med=med, ci_lo=float(lo), ci_hi=float(hi), n_runs=int(ages.size))

# ── main ────────────────────────────────────────────────────────────────────
def main():
    cat_map = load_catalogue_table(_pick_catalogue_file())
    print(f"[DD] REI_DIR = {REI_DIR}")
    # quick presence check for 1–4
    for case in ["1","2","3","4"]:
        for tier in TIERS:
            b = REI_DIR / f"{case}{tier}_bootstrap_curves_boot200.csv"
            a = REI_DIR / f"{case}{tier}_lowerdisc_curve_boot200.csv"
            if not (b.exists() and a.exists()):
                print(f"[DD] missing {case}{tier}: boot={b.exists()} agg={a.exists()}")

    all_rows=[]
    for case, truths in CASES_TRUE.items():
        if ONLY_CASES and case not in ONLY_CASES: continue
        for tier in TIERS:
            tol_for = _tol_half_window_truth_factory(truths, ADAPTIVE_MIN, ADAPTIVE_CAP)
            try:
                dd_rows, _ = reimink_event_stats(case, tier, truths, assign_mode=DD_ASSIGN, tol_for=tol_for)
                all_rows.extend(dd_rows)
            except FileNotFoundError as e:
                print("[skip] DD global-max:", e)
            if _HAVE_SCIPY:
                try:
                    ddp_rows = reimink_peaks_event_stats(case, tier, truths, tol_for=tol_for)
                    all_rows.extend(ddp_rows)
                except FileNotFoundError as e:
                    print("[skip] DD PEAKS:", e)
            c_rows = cdc_catalogue_event_stats(case, tier, truths, cat_map, tol_for=tol_for)
            all_rows.extend(c_rows)

    df = pd.DataFrame(all_rows)
    if df.empty:
        raise SystemExit("No rows collected. Check paths, SciPy, and ONLY_CASES.")
    df["bias"] = df["age_med"] - df["true_age"]
    df["abs_bias"] = df["bias"].abs()
    df["covers_truth"] = (df["ci_lo"] <= df["true_age"]) & (df["true_age"] <= df["ci_hi"])

    df_out = df.sort_values(["case","tier","event","method"])
    df_out.to_csv(OUT_DIR/"Dec_comparison_by_event.csv", index=False)
    print("✓ wrote", OUT_DIR/"comparison_by_event.csv")

    df_single = df_out[df_out["case"].isin(["1","2","3","4"])]
    single_sum = (df_single.groupby(["tier","method"], as_index=False)
        .agg(n=("age_med","count"),
             mae=("abs_bias","median"),
             max_abs=("abs_bias","max"),
             cover=("covers_truth","mean")))
    single_sum["cover"] = (single_sum["cover"]*100).round(1)
    single_sum.to_csv(OUT_DIR/"Dec_summary_single_cases1-4.csv", index=False)
    print("✓ wrote", OUT_DIR/"summary_single_cases1-4.csv")

    # --- NEW: dedicated summary for the two-event “mixed” cases 5–7
    df_mixed = df_out[df_out["case"].isin(["5","6","7"])]
    if not df_mixed.empty:
        mixed_sum = (df_mixed.groupby(["tier","method"], as_index=False)
            .agg(n=("age_med","count"),
                mae=("abs_bias","median"),
                max_abs=("abs_bias","max"),
                cover=("covers_truth","mean")))
        mixed_sum["cover"] = (mixed_sum["cover"]*100).round(1)
        mixed_sum.to_csv(OUT_DIR/"Dec_summary_mixed_cases5-7.csv", index=False)
        print("✓ wrote", OUT_DIR/"Dec_summary_mixed_cases5-7.csv")
    else:
        print("↷ no rows for cases 5–7 yet (check ONLY_CASES, catalogue tags, and Reimink files)")

    # --- optional: per-case, per-tier, per-method table for 5–7 (handy for SI)
    by_case_mixed = (df_out[df_out["case"].isin(["5","6","7"])]
        .groupby(["case","tier","method"], as_index=False)
        .agg(n=("age_med","count"),
            mae=("abs_bias","median"),
            max_abs=("abs_bias","max"),
            cover=("covers_truth","mean")))
    by_case_mixed["cover"] = (by_case_mixed["cover"]*100).round(1)
    by_case_mixed.to_csv(OUT_DIR/"Dec_summary_bycase_cases5-7.csv", index=False)
    print("✓ wrote", OUT_DIR/"summary_bycase_cases5-7.csv")

    summary = (df_out.groupby(["tier","method"], as_index=False)
        .agg(n=("age_med","count"),
             mae=("abs_bias","median"),
             max_abs=("abs_bias","max"),
             cover=("covers_truth","mean")))
    summary["cover"] = (summary["cover"]*100).round(1)
    summary.to_csv(OUT_DIR/"Dec_comparison_summary.csv", index=False)
    print("✓ wrote", OUT_DIR/"comparison_summary.csv")

    wide = (df_out.pivot_table(index=["case","tier","event","true_age"],
                                columns="method",
                                values=["age_med","ci_lo","ci_hi","abs_bias"]))
    with open(OUT_DIR/"Dec_comparison_table.tex","w") as fh:
        fh.write(wide.sort_index().to_latex(float_format=lambda x: f"{x:.0f}"))
    print("✓ wrote", OUT_DIR/"comparison_table.tex")

def dataset_level_event_recovery(cat_map):
    """
    Dataset-level event recovery for each tier and method.

    Uses:
      - CDC: all catalogue peak ages for the dataset
      - DD:  one APP-like age per dataset (median of bootstrap maxima)
      - DD-PEAKS: peak centers detected on the median likelihood curve

    Matching: greedy match within tol_for(true_age).
    """
    rows = []

    for tier in TIERS:
        tier = tier.lower()
        # accumulators per method
        acc = {
            "CDC": dict(true=0, pred=0, match=0, abs_err=[]),
            "DD": dict(true=0, pred=0, match=0, abs_err=[]),
            "DD--PEAKS": dict(true=0, pred=0, match=0, abs_err=[]),
        }

        for case, truths in CASES_TRUE.items():
            # truth-window half-width function for matching
            tol_for = _tol_half_window_truth_factory(truths, ADAPTIVE_MIN, ADAPTIVE_CAP)
            acc["CDC"]["true"] += len(truths)
            acc["DD"]["true"]  += len(truths)
            acc["DD--PEAKS"]["true"] += len(truths)

            tag = f"{case}{tier.upper()}"

            # ---- CDC predictions: all catalogue peaks for this dataset ----
            cdc_preds = [e["age"] for e in cat_map.get(tag, [])]
            acc["CDC"]["pred"] += len(cdc_preds)
            m, miss, extra = _greedy_match(truths, cdc_preds, tol_for)
            acc["CDC"]["match"] += len(m)
            acc["CDC"]["abs_err"].extend([abs(tt - ee) for tt, ee in m])

            # ---- DD predictions: one APP-like summary per dataset ----
            try:
                dd_app = reimink_app_like(case, tier)
                dd_preds = []
                if np.isfinite(dd_app.get("age_med", np.nan)):
                    dd_preds = [float(dd_app["age_med"])]
            except FileNotFoundError:
                dd_preds = []
            acc["DD"]["pred"] += len(dd_preds)
            m, miss, extra = _greedy_match(truths, dd_preds, tol_for)
            acc["DD"]["match"] += len(m)
            acc["DD"]["abs_err"].extend([abs(tt - ee) for tt, ee in m])

            # ---- DD--PEAKS predictions: peaks on the median curve ----
            ddp_preds = []
            if _HAVE_SCIPY:
                try:
                    x, boot, y_med = load_reimink(case, tier)
                    pk, props = find_peaks(y_med, prominence=0.02, width=3)
                    ddp_preds = [float(x[j]) for j in pk]
                except FileNotFoundError:
                    ddp_preds = []
            acc["DD--PEAKS"]["pred"] += len(ddp_preds)
            m, miss, extra = _greedy_match(truths, ddp_preds, tol_for)
            acc["DD--PEAKS"]["match"] += len(m)
            acc["DD--PEAKS"]["abs_err"].extend([abs(tt - ee) for tt, ee in m])

        # finalise per tier rows
        for method, a in acc.items():
            true = a["true"]
            pred = a["pred"]
            match = a["match"]
            recall = 100.0 * (match / true) if true else np.nan
            precision = 100.0 * (match / pred) if pred else np.nan
            f1 = (2 * recall * precision / (recall + precision)) if (recall > 0 and precision > 0) else 0.0
            mae_matched = float(np.median(a["abs_err"])) if a["abs_err"] else np.nan

            rows.append(dict(
                tier=tier.upper(),
                method=method,
                n_datasets=len(CASES_TRUE),
                recall=recall,
                precision=precision,
                f1=f1,
                mae_matched=mae_matched,
                n_true=true,
                n_pred=pred,
                n_matched=match,
            ))

    return pd.DataFrame(rows)

if __name__ == "__main__":
    # Run the existing event-level analyses
    main()

    # ------------------------------------------------------------------
    # NEW: dataset-level event recovery summary (recall/precision/F1)
    # ------------------------------------------------------------------
    cat_map = load_catalogue_table(_pick_catalogue_file())
    rec = dataset_level_event_recovery(cat_map)

    # Save CSV and LaTeX versions
    rec_csv = OUT_DIR / "Dec_event_recovery_test.csv"
    rec_tex = OUT_DIR / "Dec_event_recovery_test.tex"
    rec.to_csv(rec_csv, index=False)
    with open(rec_tex, "w") as fh:
        fh.write(
            rec.to_latex(
                index=False,
                float_format=lambda x: f"{x:.1f}",
                caption=(
                    "Dataset-level event recovery by tier across Cases~1--7. "
                    "Recall is the fraction of true events recovered; precision is the "
                    "fraction of reported events that match a true event; $F_1$ is their "
                    "harmonic mean. ``MAE (matched)'' is the median absolute error among matched events only."
                ),
                label="tab:event_recovery",
            )
        )
    print(f"✓ wrote {rec_csv} and {rec_tex}")
