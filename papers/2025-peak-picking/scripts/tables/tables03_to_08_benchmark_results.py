#!/usr/bin/env python3
"""tables03_to_08_benchmark_results.py

Reproduce manuscript benchmark Tables 3–8 from existing CDC and Reimink/DD outputs.

This script is designed to run "as-is" from within the paper repository, with all
inputs located under the paper's /data/derived/ folder.

Inputs (expected relative locations)
-----------------------------------
1) CDC ensemble catalogue CSV (one row per peak per dataset):
     <paper>/data/derived/ensemble_catalogue.csv

   Required columns (case-insensitive):
     sample, peak_no, age_ma, ci_low, ci_high, support

2) Reimink/DD outputs per dataset:
     <paper>/data/derived/reimink_discordance_dating/
         {case}{tier}_bootstrap_curves_boot200.csv
         {case}{tier}_lowerdisc_curve_boot200.csv

Outputs
-------
Writes CSV and LaTeX tabulars into:
  <paper>/outputs/<out-subdir>/   (default out-subdir: "tables")

Method definitions
------------------------------------------------------------------
Truth windows:
  Half-width for each true event is half the nearest-neighbour spacing in the list
  of true ages, clamped to [HALF_MIN, HALF_CAP] Ma.

CDC (event-wise):
  For each true event, select the best CDC candidate peak *within that event's truth window*.
  Ranking: smallest |age - true|, then highest support, then narrowest CI.

DD (baseline, event-wise):
  Compute bootstrap global maximum age for each bootstrap curve (one max per bootstrap).
  For each true event, take the subset of maxima inside that event’s truth window.
  Report:
    - age = median(subset)
    - CI = 95% equal-tailed if subset size >= 5,
           else small-sample fallbacks (see MIN_N_BOOT, and ci_with_small_sample_fallback()).

DD–PEAKS (event-wise):
  Peaks are detected on the *aggregate* curve from *_lowerdisc_curve_boot200.csv (after normalisation),
  using:
      find_peaks(y, prominence=0.02, width=3)
  For each detected peak, compute the FWHM half-width using peak_widths(rel_height=0.5),
  then assign bootstrap maxima whose ages fall within that FWHM window around the peak centre.
  For each true event, choose the closest eligible peak centre within the truth window, and report:
    - age = median(assigned maxima)
    - CI = 95% equal-tailed if assigned size >= 5, else fallbacks (matches old script)
    - support = assigned / total_boot (reported as percent in outputs)

Tables 3–6 use event-level rows; Table 7 uses dataset-level recovery metrics; Table 8 reports
median interval widths.

CLI flags (optional)
--------------------
  --catalogue PATH      override catalogue path
  --rei-dir PATH        override Reimink output directory
  --skip-missing        skip datasets with missing inputs instead of failing
  --out-subdir NAME     output subdir under <paper>/outputs/ (default: tables)

Dependencies
------------
SciPy is required for DD–PEAKS (find_peaks, peak_widths).

Author: Lucy Mathieson
Date: 29/12/2025
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

UTIL_DIR = Path(__file__).resolve().parents[1] / "_util"  # .../scripts/_util
if str(UTIL_DIR) not in sys.path:
    sys.path.insert(0, str(UTIL_DIR))

from paper_io import parse_paper_args
from benchmark_definitions import CASES_TRUE, TIER_TO_REI, TIERS

try:
    from scipy.signal import find_peaks, peak_widths  # type: ignore
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

# Truth-window clamp (Ma) used throughout the manuscript scoring
HALF_MIN = 50.0
HALF_CAP = 120.0

# Small-sample CI behaviour (matches 28 Nov script)
MIN_N_BOOT = 5


# ----------------------------- Small utilities -----------------------------
def _norm_sample(s: str) -> str:
    return str(s).strip().upper()


def _to_ma(x: np.ndarray) -> np.ndarray:
    """Convert years to Ma if values look like years."""
    x = np.asarray(x, float)
    if x.size == 0:
        return x
    return (x / 1e6) if np.nanmax(x) > 1e6 else x


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _pct(x: float) -> float:
    return float(np.clip(x, 0.0, 100.0))


def support_pct(raw, *, n_runs: Optional[int] = None) -> float:
    """Normalise support to percent, accepting fraction, percent, or counts."""
    if raw is None:
        return float("nan")
    if isinstance(raw, str):
        raw = raw.strip().replace("%", "")
    v = _safe_float(raw)
    if not np.isfinite(v):
        return float("nan")
    if v <= 1.0 + 1e-12:  # fraction
        return _pct(100.0 * v)
    if v <= 100.0 + 1e-9:  # already percent
        return _pct(v)
    if n_runs:
        return _pct(100.0 * v / float(n_runs))
    # unknown scale, assume percent-like
    return _pct(v)


def truth_half_window_factory(
    true_ages: Sequence[float],
    half_min: float = HALF_MIN,
    half_cap: float = HALF_CAP,
):
    """Half-window per true age from nearest-neighbour spacing, clamped."""
    t = np.array(sorted(true_ages), float)

    def half_for(tt: float) -> float:
        if t.size <= 1:
            return float(half_min)
        i = int(np.argmin(np.abs(t - tt)))
        left_gap = (t[i] - t[i - 1]) if i > 0 else np.inf
        right_gap = (t[i + 1] - t[i]) if i < t.size - 1 else np.inf
        half = 0.5 * float(min(left_gap, right_gap))
        half = max(half_min, min(half_cap, half))
        return float(half)

    return half_for


def grid_step(x: np.ndarray) -> float:
    d = np.diff(np.asarray(x, float))
    return float(np.nanmedian(d)) if d.size else 10.0


def ci_with_small_sample_fallback(
    picks: np.ndarray,
    step: float,
    *,
    min_n_95: int = MIN_N_BOOT,
) -> tuple[float, float, float, int]:
    """
    Match the CI fallback logic in the 28 Nov script.

      n >= min_n_95: 2.5/97.5 percentiles
      n >= 3:        5/95 percentiles
      n == 2:        med ± grid_step
      n == 1:        degenerate at median
    """
    picks = np.asarray(picks, float)
    picks = picks[np.isfinite(picks)]
    if picks.size == 0:
        return float("nan"), float("nan"), float("nan"), 0

    med = float(np.median(picks))
    n = int(picks.size)

    if n >= int(min_n_95):
        lo, hi = np.percentile(picks, [2.5, 97.5])
    elif n >= 3:
        lo, hi = np.percentile(picks, [5, 95])
    elif n == 2:
        lo, hi = med - float(step), med + float(step)
    else:
        lo = hi = med

    return float(med), float(lo), float(hi), n


# ----------------------------- Rounding (match manuscript) ------------------
def _round_half_away_from_zero(x: float) -> int:
    """
    Round to nearest integer with ties (.5) rounded away from zero.

    This avoids Python's default bankers rounding and matches typical manuscript tables.
    """
    x = float(x)
    if x >= 0:
        return int(np.floor(x + 0.5))
    return int(np.ceil(x - 0.5))


def format_int_or_blank(x: float) -> str:
    if not np.isfinite(x):
        return ""
    return str(_round_half_away_from_zero(float(x)))


def format_bias(x: float) -> str:
    """Signed integer bias; omit '+' for positives to match common table style."""
    if not np.isfinite(x):
        return ""
    v = _round_half_away_from_zero(float(x))
    return f"{v:d}"


# ----------------------------- CDC catalogue ------------------------------
def load_catalogue_table(path: Path) -> Dict[str, List[dict]]:
    """Return mapping sample -> list of {age, lo, hi, support}."""
    if not path.exists():
        raise FileNotFoundError(f"CDC catalogue not found: {path}")

    req = {"sample", "peak_no", "age_ma", "ci_low", "ci_high", "support"}

    df = pd.read_csv(path, comment="#", engine="python")
    df.columns = [str(c).strip().lower() for c in df.columns]

    missing = sorted(req - set(df.columns))
    if missing:
        raise ValueError(f"CDC catalogue missing columns: {missing}. Found: {list(df.columns)}")

    # Coerce numerics, tolerate commas and %.
    for col in ["age_ma", "ci_low", "ci_high", "support"]:
        df[col] = pd.to_numeric(
            df[col]
            .astype(str)
            .str.replace("%", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip(),
            errors="coerce",
        )

    out: Dict[str, List[dict]] = {}
    keep = df.dropna(subset=["sample", "age_ma", "ci_low", "ci_high"]).copy()
    for _, r in keep.iterrows():
        samp = _norm_sample(r["sample"])
        out.setdefault(samp, []).append(
            dict(
                age=float(r["age_ma"]),
                lo=float(r["ci_low"]),
                hi=float(r["ci_high"]),
                support=float(r["support"]) if pd.notna(r["support"]) else float("nan"),
            )
        )

    for k in out:
        out[k].sort(key=lambda d: d["age"])
    return out


def select_candidate_for_event(
    cands: Sequence[dict],
    true_age: float,
    half_win: float,
) -> Optional[dict]:
    """Pick the best CDC candidate inside the truth window."""
    in_win = []
    for c in cands:
        a = float(c["age"])
        if abs(a - true_age) <= half_win:
            in_win.append(c)
    if not in_win:
        return None

    # Rank: closest to truth, then higher support, then narrower CI.
    def key(c: dict):
        a = float(c["age"])
        sup = support_pct(c.get("support", float("nan")))
        w = float(c["hi"]) - float(c["lo"])
        if not np.isfinite(sup):
            sup = -1.0
        if not np.isfinite(w):
            w = float("inf")
        return (abs(a - true_age), -sup, w)

    return min(in_win, key=key)


def per_event_rows_cdc(
    sample: str,
    true_ages: Sequence[float],
    catalogue_map: Dict[str, List[dict]],
):
    """Event-level CDC rows (one row per true event)."""
    samp = _norm_sample(sample)
    cands = catalogue_map.get(samp, [])
    half_for = truth_half_window_factory(true_ages)

    rows = []
    for ev, atrue in enumerate(true_ages, 1):
        atrue = float(atrue)
        half = half_for(atrue)
        pick = select_candidate_for_event(cands, atrue, half)

        if pick is None:
            rows.append(
                dict(
                    sample=samp,
                    event=ev,
                    true_age=atrue,
                    age_med=float("nan"),
                    ci_lo=float("nan"),
                    ci_hi=float("nan"),
                    support=float("nan"),
                )
            )
        else:
            rows.append(
                dict(
                    sample=samp,
                    event=ev,
                    true_age=atrue,
                    age_med=float(pick["age"]),
                    ci_lo=float(pick["lo"]),
                    ci_hi=float(pick["hi"]),
                    support=support_pct(pick.get("support", float("nan"))),
                )
            )
    return rows


# ----------------------------- Reimink/DD ---------------------------------
def load_reimink(case: str, tier_letter: str, rei_dir: Path):
    """Load Reimink bootstrap curves and the aggregate (lowerdisc) curve."""
    tier = TIER_TO_REI[tier_letter]
    boot_path = rei_dir / f"{case}{tier}_bootstrap_curves_boot200.csv"
    agg_path = rei_dir / f"{case}{tier}_lowerdisc_curve_boot200.csv"
    if not boot_path.exists() or not agg_path.exists():
        raise FileNotFoundError(
            f"Missing Reimink files for {case}{tier_letter}: {boot_path.name}, {agg_path.name}"
        )

    boot_df = pd.read_csv(boot_path)
    agg_df = pd.read_csv(agg_path)

    piv = (
        boot_df.pivot(
            index="Lower Intercept",
            columns="run.number",
            values="normalized.sum.likelihood",
        ).sort_index()
    )
    x_boot = _to_ma(piv.index.values)
    boot = piv.values.T  # (n_boot, n_grid)

    # Normalise each bootstrap curve to peak=1 (as in old script).
    with np.errstate(invalid="ignore"):
        rowmax = np.nanmax(boot, axis=1, keepdims=True)
        rowmax[~np.isfinite(rowmax)] = 1.0
        rowmax[rowmax <= 0] = 1.0
        boot = np.divide(boot, rowmax, out=np.zeros_like(boot), where=(rowmax > 0))

    # Aggregate curve (lowerdisc), normalised to max=1.
    x_agg = _to_ma(agg_df["Lower Intercept"].values)
    y_agg = agg_df["normalized.sum.likelihood"].astype(float).values
    ymax = np.nanmax(y_agg) if np.isfinite(y_agg).any() else 1.0
    y_agg = (y_agg / ymax) if ymax > 0 else y_agg

    # Align aggregate curve onto bootstrap grid if needed.
    if (x_agg.shape == x_boot.shape) and np.allclose(x_agg, x_boot):
        y_med_agg = y_agg
    else:
        y_med_agg = np.interp(x_boot, x_agg, y_agg, left=np.nan, right=np.nan)
        if np.isnan(y_med_agg).any():
            y_alt = np.nanmedian(boot, axis=0)
            m = np.isnan(y_med_agg)
            y_med_agg[m] = y_alt[m]

    return x_boot, boot, y_med_agg


def bootstrap_global_max_ages(x_grid: np.ndarray, boot_2d: np.ndarray) -> np.ndarray:
    """Age at global maximum for each bootstrap curve (dropping all-NaN curves)."""
    if boot_2d.size == 0:
        return np.array([], float)
    valid = np.isfinite(boot_2d).any(axis=1)
    if not valid.any():
        return np.array([], float)
    boot_v = boot_2d[valid]
    with np.errstate(invalid="ignore"):
        j = np.nanargmax(np.where(np.isfinite(boot_v), boot_v, -np.inf), axis=1)
    return x_grid[j]


# ----------------------------- DD (event-wise) -----------------------------
def per_event_rows_dd(sample: str, true_ages: Sequence[float], rei_dir: Path):
    """
    Event-wise DD scoring (28 Nov logic):
      - compute bootstrap global maxima (one max per bootstrap; drop all-NaN curves)
      - for each true event, take maxima inside that event’s truth window
      - report median + CI using small-sample fallbacks
    """
    case = sample[:-1]
    tier = sample[-1]

    x, boot, _ = load_reimink(case, tier, rei_dir)
    x_max = bootstrap_global_max_ages(x, boot)

    half_for = truth_half_window_factory(true_ages)
    step = grid_step(x)

    rows = []
    for ev, atrue in enumerate(true_ages, 1):
        atrue = float(atrue)
        half = half_for(atrue)

        in_win = x_max[(x_max >= atrue - half) & (x_max <= atrue + half)]

        if in_win.size == 0:
            rows.append(
                dict(
                    sample=_norm_sample(sample),
                    event=ev,
                    true_age=atrue,
                    age_med=float("nan"),
                    ci_lo=float("nan"),
                    ci_hi=float("nan"),
                    support=float("nan"),
                )
            )
        else:
            age_med, ci_lo, ci_hi, _n = ci_with_small_sample_fallback(in_win, step)
            rows.append(
                dict(
                    sample=_norm_sample(sample),
                    event=ev,
                    true_age=atrue,
                    age_med=age_med,
                    ci_lo=float(ci_lo),
                    ci_hi=float(ci_hi),
                    support=float("nan"),  # not used for DD in Tables 3–6
                )
            )

    return rows


# ----------------------------- DD–PEAKS (28 Nov) ---------------------------
def dd_peak_candidates(
    case: str,
    tier: str,
    rei_dir: Path,
    *,
    prominence: float = 0.02,
    min_width_nodes: int = 3,
):
    """
    DD–PEAKS candidates matching the 28 Nov script.

    - Peaks found on the AGGREGATE curve from *_lowerdisc_curve_boot200.csv (y_med_agg).
    - find_peaks(..., prominence=0.02, width=3).
    - FWHM window half-width from peak_widths(rel_height=0.5) * dx.
    - Assign bootstrap maxima within ±half-width around each peak centre.

    Returns list of dicts with keys:
      center, age_med, lo, hi, support
    """
    if not _HAVE_SCIPY:
        raise RuntimeError(
            "SciPy is required for DD-PEAKS (scipy.signal.find_peaks, peak_widths). "
            "Install with: pip install scipy"
        )

    x, boot, y = load_reimink(case, tier, rei_dir)
    y = np.asarray(y, float)
    if not np.isfinite(y).any():
        return []

    # Bootstrap maxima extraction matching old peaks script:
    # (Note: does NOT drop all-NaN curves before argmax; matches the historical code.)
    boot_safe = np.where(np.isfinite(boot), boot, -np.inf)
    idx = np.argmax(boot_safe, axis=1)
    x_max = x[idx]
    x_max = x_max[np.isfinite(x_max)]
    if x_max.size == 0:
        return []

    pk, _ = find_peaks(y, prominence=float(prominence), width=int(min_width_nodes))
    if pk.size == 0:
        return []

    widths, *_ = peak_widths(y, pk, rel_height=0.5)
    dx = grid_step(x)

    cands = []
    for j, w in zip(pk.astype(int), widths):
        center = float(x[j])
        half = 0.5 * float(w) * dx

        picks = x_max[np.abs(x_max - center) <= half]
        if picks.size < 2:
            continue

        age_med = float(np.median(picks))

        # CI rules consistent with old script
        if picks.size >= MIN_N_BOOT:
            lo, hi = np.percentile(picks, [2.5, 97.5])
        elif picks.size == 2:
            lo, hi = float(np.min(picks)), float(np.max(picks))
        else:
            lo = hi = age_med

        cands.append(
            dict(
                center=center,
                age_med=age_med,
                lo=float(lo),
                hi=float(hi),
                support=support_pct(float(picks.size) / float(x_max.size)),  # percent
            )
        )

    return sorted(cands, key=lambda d: d["center"])


def per_event_rows_dd_peaks(sample: str, true_ages: Sequence[float], rei_dir: Path):
    """
    Event-wise DD–PEAKS attribution (28 Nov logic):
      - build DD–PEAKS candidates from aggregate curve
      - for each true event, select closest candidate peak centre within truth window
    """
    case = sample[:-1]
    tier = sample[-1]
    cands = dd_peak_candidates(case, tier, rei_dir)
    half_for = truth_half_window_factory(true_ages)

    rows = []
    for ev, atrue in enumerate(true_ages, 1):
        atrue = float(atrue)
        half = half_for(atrue)

        eligible = [c for c in cands if abs(float(c["center"]) - atrue) <= half]
        if not eligible:
            rows.append(
                dict(
                    sample=_norm_sample(sample),
                    event=ev,
                    true_age=atrue,
                    age_med=float("nan"),
                    ci_lo=float("nan"),
                    ci_hi=float("nan"),
                    support=float("nan"),
                )
            )
            continue

        pick = min(eligible, key=lambda c: abs(float(c["center"]) - atrue))
        rows.append(
            dict(
                sample=_norm_sample(sample),
                event=ev,
                true_age=atrue,
                age_med=float(pick["age_med"]),
                ci_lo=float(pick["lo"]),
                ci_hi=float(pick["hi"]),
                support=float(pick.get("support", float("nan"))),
            )
        )

    return rows


# ----------------------------- Scoring + tables ----------------------------
def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["bias"] = out["age_med"] - out["true_age"]
    out["abs_bias"] = out["bias"].abs()
    out["covers_truth"] = (out["ci_lo"] <= out["true_age"]) & (out["true_age"] <= out["ci_hi"])
    out["covers_truth"] = out["covers_truth"].fillna(False)
    out["width"] = out["ci_hi"] - out["ci_lo"]
    return out


def summarise(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """Summary metrics used in Tables 3–6."""
    g = df.groupby(group_cols, dropna=False)

    def _n_assigned(s: pd.Series) -> int:
        return int(s.notna().sum())

    def _nanmedian(a):
        a = np.asarray(a, float)
        return float(np.nanmedian(a)) if np.isfinite(a).any() else float("nan")

    def _nanmax(a):
        a = np.asarray(a, float)
        return float(np.nanmax(a)) if np.isfinite(a).any() else float("nan")

    out = g.agg(
        n_true=("true_age", "size"),
        n_assigned=("age_med", _n_assigned),
        median_bias=("bias", lambda s: _nanmedian(s.values)),
        mae=("abs_bias", lambda s: _nanmedian(s.values)),
        max_abs=("abs_bias", lambda s: _nanmax(s.values)),
        coverage=("covers_truth", "mean"),
    ).reset_index()

    out["coverage"] = 100.0 * out["coverage"].astype(float)
    return out


def write_csv_and_tex(
    df: pd.DataFrame,
    csv_path: Path,
    tex_path: Path,
    *,
    float_fmt: Optional[str] = None,
    column_format: Optional[str] = None,
):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    latex = df.to_latex(index=False, escape=False, float_format=float_fmt, column_format=column_format)
    tex_path.write_text(latex, encoding="utf-8")


def dataset_level_metrics(
    method: str,
    datasets: Sequence[Tuple[str, str, Sequence[float]]],
    catalogue_map: Dict[str, List[dict]],
    rei_dir: Path,
) -> dict:
    """
    Dataset-level recall/precision/F1 under truth-window matching (Table 7).

    Definitions (match old script intent):
      - CDC: all catalogue peak ages for the dataset
      - DD:  one "APP-like" age per dataset = median of bootstrap global maxima
      - DD-PEAKS: peak centres detected on the *aggregate* curve (find_peaks prominence=0.02, width=3)
    """
    total_true = 0
    total_pred = 0
    total_matched = 0
    abs_errs: List[float] = []

    for case, tier, truths in datasets:
        sample = f"{case}{tier}"
        true_ages = list(map(float, truths))
        total_true += len(true_ages)

        # ---- predictions per dataset ----
        if method == "CDC":
            preds: List[float] = [float(e["age"]) for e in catalogue_map.get(_norm_sample(sample), [])]

        elif method == "DD":
            x, boot, _ = load_reimink(case, tier, rei_dir)
            x_max = bootstrap_global_max_ages(x, boot)
            preds = [float(np.median(x_max))] if x_max.size else []

        elif method == "DD-PEAKS":
            x, _boot, y = load_reimink(case, tier, rei_dir)
            y = np.asarray(y, float)
            if np.isfinite(y).any():
                pk, _ = find_peaks(y, prominence=0.02, width=3)
                preds = [float(x[j]) for j in pk.astype(int)]
            else:
                preds = []
        else:
            raise ValueError(method)

        total_pred += len(preds)

        # ---- greedy matching within truth windows ----
        half_for = truth_half_window_factory(true_ages)
        unused = preds[:]  # list copy

        for t in sorted(true_ages):
            half = half_for(t)
            idxs = [i for i, p in enumerate(unused) if abs(p - t) <= half]
            if not idxs:
                continue
            best_i = min(idxs, key=lambda i: abs(unused[i] - t))
            p_best = unused.pop(best_i)
            total_matched += 1
            abs_errs.append(abs(p_best - t))

    recall = (total_matched / total_true) if total_true else float("nan")
    precision = (total_matched / total_pred) if total_pred else float("nan")
    if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = float("nan")

    mae_matched = float(np.median(abs_errs)) if abs_errs else float("nan")

    return dict(
        method=method,
        n_datasets=len(datasets),
        n_true=total_true,
        n_pred=total_pred,
        n_matched=total_matched,
        recall_pct=100.0 * recall if np.isfinite(recall) else float("nan"),
        precision_pct=100.0 * precision if np.isfinite(precision) else float("nan"),
        f1_pct=100.0 * f1 if np.isfinite(f1) else float("nan"),
        mae_matched=mae_matched,
    )


def main() -> None:
    base = parse_paper_args(__file__)

    ap = argparse.ArgumentParser()
    ap.add_argument("--catalogue", type=str, default=None)
    ap.add_argument("--rei-dir", type=str, default=None)
    ap.add_argument("--skip-missing", action="store_true")
    ap.add_argument("--out-subdir", type=str, default="tables")
    args = ap.parse_args()

    data_dir = Path(base.data_dir)
    out_dir = Path(base.out_dir) / args.out_subdir

    # Default paths: within the paper folder (no env vars)
    default_cat = data_dir / "derived" / "ensemble_catalogue.csv"
    default_rei = data_dir / "derived" / "reimink_discordance_dating"

    cat_path = Path(args.catalogue) if args.catalogue else default_cat
    rei_dir = Path(args.rei_dir) if args.rei_dir else default_rei

    if not cat_path.exists():
        raise SystemExit(
            "CDC catalogue CSV not found.\n"
            f"Expected: {default_cat}\n"
            "If your file is elsewhere, run with: --catalogue PATH"
        )
    if not rei_dir.exists():
        raise SystemExit(
            "Reimink output directory not found.\n"
            f"Expected: {default_rei}\n"
            "If your folder is elsewhere, run with: --rei-dir PATH"
        )

    if not _HAVE_SCIPY:
        raise SystemExit(
            "SciPy is required to reproduce DD-PEAKS tables.\n"
            "Install it (in the same environment) with: pip install scipy"
        )

    catalogue_map = load_catalogue_table(cat_path)

    # Dataset list (case, tier, truths)
    datasets: List[Tuple[str, str, Sequence[float]]] = []
    for case, truths in CASES_TRUE.items():
        for tier in TIERS:
            datasets.append((case, tier, truths))

    # Build per-event rows for each method
    all_rows: List[dict] = []
    missing: List[str] = []

    for case, tier, truths in datasets:
        sample = f"{case}{tier}"

        # CDC rows: catalogue may be empty (still valid)
        for r in per_event_rows_cdc(sample, truths, catalogue_map):
            r.update(case=case, tier=tier, method="CDC")
            all_rows.append(r)

        # DD and DD-PEAKS require Reimink files
        try:
            for r in per_event_rows_dd(sample, truths, rei_dir):
                r.update(case=case, tier=tier, method="DD")
                all_rows.append(r)
            for r in per_event_rows_dd_peaks(sample, truths, rei_dir):
                r.update(case=case, tier=tier, method="DD-PEAKS")
                all_rows.append(r)
        except FileNotFoundError as e:
            if args.skip_missing:
                missing.append(str(e))
                continue
            raise

    if not all_rows:
        raise SystemExit("No rows produced. Check inputs.")

    df = pd.DataFrame(all_rows)
    df = add_derived_columns(df)

    # Save the event-level table too (useful for debugging)
    write_csv_and_tex(
        df.sort_values(["case", "tier", "method", "event"]),
        out_dir / "by_event_all_methods.csv",
        out_dir / "by_event_all_methods.tex",
    )

    # ---------------- Table 3: aggregate across all events ----------------
    t3 = summarise(df, ["method"])
    t3 = t3[["method", "n_assigned", "mae", "coverage"]].copy()
    t3 = t3.sort_values(["method"])

    t3_fmt = t3.copy()
    t3_fmt["n_assigned"] = t3_fmt["n_assigned"].astype(int).astype(str)
    t3_fmt["mae"] = t3_fmt["mae"].map(format_int_or_blank)
    t3_fmt["coverage"] = t3_fmt["coverage"].map(format_int_or_blank)

    write_csv_and_tex(t3, out_dir / "table3_aggregate_per_event.csv", out_dir / "table3_aggregate_per_event.tex")
    write_csv_and_tex(
        t3_fmt,
        out_dir / "table3_aggregate_per_event_formatted.csv",
        out_dir / "table3_aggregate_per_event_formatted.tex",
    )

    # ---------------- Table 4: age bands ----------------
    df_band = df.copy()
    df_band["age_band"] = np.where(df_band["true_age"] < 1000.0, "<1 Ga", ">=1 Ga")
    t4 = summarise(df_band, ["age_band", "method"])
    t4 = t4[["age_band", "method", "n_true", "n_assigned", "median_bias", "mae", "coverage"]]
    t4 = t4.sort_values(["age_band", "method"])

    t4_fmt = t4.copy()
    t4_fmt["n_true"] = t4_fmt["n_true"].astype(int).astype(str)
    t4_fmt["n_assigned"] = t4_fmt["n_assigned"].astype(int).astype(str)
    t4_fmt["median_bias"] = t4_fmt["median_bias"].map(format_bias)
    t4_fmt["mae"] = t4_fmt["mae"].map(format_int_or_blank)
    t4_fmt["coverage"] = t4_fmt["coverage"].map(format_int_or_blank)

    write_csv_and_tex(t4, out_dir / "table4_age_bands.csv", out_dir / "table4_age_bands.tex")
    write_csv_and_tex(
        t4_fmt, out_dir / "table4_age_bands_formatted.csv", out_dir / "table4_age_bands_formatted.tex"
    )

    # ---------------- Table 5: single-stage (Cases 1–4) -------------------
    df5 = df[df["case"].isin(["1", "2", "3", "4"])].copy()

    t5 = summarise(df5, ["tier", "method"])
    t5 = t5[["tier", "method", "n_assigned", "mae", "max_abs", "coverage"]].sort_values(["tier", "method"])

    t5_fmt = t5.copy()
    t5_fmt["n_assigned"] = t5_fmt["n_assigned"].astype(int).astype(str)
    t5_fmt["mae"] = t5_fmt["mae"].map(format_int_or_blank)
    t5_fmt["max_abs"] = t5_fmt["max_abs"].map(format_int_or_blank)
    t5_fmt["coverage"] = t5_fmt["coverage"].map(format_int_or_blank)

    write_csv_and_tex(t5, out_dir / "table5_single_stage_by_tier.csv", out_dir / "table5_single_stage_by_tier.tex")
    write_csv_and_tex(
        t5_fmt,
        out_dir / "table5_single_stage_by_tier_formatted.csv",
        out_dir / "table5_single_stage_by_tier_formatted.tex",
    )

    # ---------------- Table 6: two-stage benchmarks (Cases 5–7) -----------
    df6 = df[df["case"].isin(["5", "6", "7"])].copy()

    t6 = summarise(df6, ["tier", "method"])
    t6 = t6[["tier", "method", "n_assigned", "mae", "max_abs", "coverage"]].sort_values(["tier", "method"])

    t6_all = summarise(df6, ["method"])
    t6_all.insert(0, "tier", "All (A--C)")
    t6_all = t6_all[["tier", "method", "n_assigned", "mae", "max_abs", "coverage"]]

    t6 = pd.concat([t6, t6_all], ignore_index=True)

    t6_fmt = t6.copy()
    t6_fmt["n_assigned"] = t6_fmt["n_assigned"].astype(int).astype(str)
    t6_fmt["mae"] = t6_fmt["mae"].map(format_int_or_blank)
    t6_fmt["max_abs"] = t6_fmt["max_abs"].map(format_int_or_blank)
    t6_fmt["coverage"] = t6_fmt["coverage"].map(format_int_or_blank)

    write_csv_and_tex(t6, out_dir / "table6_two_stage_by_tier.csv", out_dir / "table6_two_stage_by_tier.tex")
    write_csv_and_tex(
        t6_fmt,
        out_dir / "table6_two_stage_by_tier_formatted.csv",
        out_dir / "table6_two_stage_by_tier_formatted.tex",
    )

    # ---------------- Table 7: dataset-level recovery ---------------------
    rows7 = []
    for tier in TIERS:
        ds_tier = [(case, tier, truths) for case, truths in CASES_TRUE.items()]
        for m in ["CDC", "DD", "DD-PEAKS"]:
            r = dataset_level_metrics(m, ds_tier, catalogue_map, rei_dir)
            r["tier"] = str(tier).upper()
            rows7.append(r)

    t7 = pd.DataFrame(rows7)
    t7 = t7[
        [
            "tier",
            "method",
            "n_datasets",
            "n_true",
            "n_pred",
            "n_matched",
            "recall_pct",
            "precision_pct",
            "f1_pct",
            "mae_matched",
        ]
    ].sort_values(["tier", "method"])

    t7_fmt = t7.copy()
    for c in ["n_datasets", "n_true", "n_pred", "n_matched"]:
        t7_fmt[c] = t7_fmt[c].astype(int).astype(str)
    for c in ["recall_pct", "precision_pct", "f1_pct", "mae_matched"]:
        t7_fmt[c] = t7_fmt[c].map(format_int_or_blank)

    write_csv_and_tex(t7, out_dir / "table7_event_recovery_by_tier.csv", out_dir / "table7_event_recovery_by_tier.tex")
    write_csv_and_tex(
        t7_fmt,
        out_dir / "table7_event_recovery_by_tier_formatted.csv",
        out_dir / "table7_event_recovery_by_tier_formatted.tex",
    )

    # ---------------- Table 8: 95% interval widths (by tier) --------------
    dfw = df.copy()
    dfw = dfw[dfw["age_med"].notna()].copy()
    dfw = dfw[np.isfinite(dfw["width"])].copy()

    t8 = (
        dfw.groupby(["tier", "method"], dropna=False)
        .agg(median_width=("width", lambda s: float(np.nanmedian(s.values))))
        .reset_index()
    )

    t8["tier"] = t8["tier"].astype(str).str.upper()
    t8 = t8.sort_values(["tier", "method"])

    t8_fmt = t8.copy()
    t8_fmt["median_width"] = t8_fmt["median_width"].map(format_int_or_blank)

    write_csv_and_tex(t8, out_dir / "table8_interval_widths_by_tier.csv", out_dir / "table8_interval_widths_by_tier.tex")
    write_csv_and_tex(
        t8_fmt,
        out_dir / "table8_interval_widths_by_tier_formatted.csv",
        out_dir / "table8_interval_widths_by_tier_formatted.tex",
    )

    if missing:
        print("\n[skip-missing] The following datasets were missing Reimink inputs:")
        for s in missing:
            print("  -", s)

    print(f"\nWrote tables into: {out_dir}")
    print("Key outputs:")
    print("  table3_aggregate_per_event_formatted.tex")
    print("  table4_age_bands_formatted.tex")
    print("  table5_single_stage_by_tier_formatted.tex")
    print("  table6_two_stage_by_tier_formatted.tex")
    print("  table7_event_recovery_by_tier_formatted.tex")
    print("  table8_interval_widths_by_tier_formatted.tex")


if __name__ == "__main__":
    main()
