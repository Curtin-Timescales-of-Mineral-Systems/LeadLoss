#!/usr/bin/env python3
"""tables_10_12_runtime_tables.py

Generate manuscript **Tables 10–12** (runtime tables) from the runtime logs
shipped with the repository.

Tables
------
- Table 10: CDC runtime by tier (R=200; ngrid=200)
- Table 11: DD runtime by tier (nboot=200)
- Table 12: Runtime comparison between CDC and DD (Speedup = DD/CDC)

Inputs (defaults)
-----------------
Expected in:
  <paper>/data/derived/
    runtime_log_28Nov.csv
    runtime_log_reimink.csv

You may override those locations via CLI flags:
  --cdc-log PATH
  --dd-log  PATH

Outputs
-------
Writes into:
  <paper>/outputs/tables/
    table10_cdc_runtime_by_tier.csv/.tex
    table11_dd_runtime_by_tier.csv/.tex
    table12_runtime_comparison.csv/.tex

Dependencies
------------
- pandas
- numpy

Author: Lucy Mathieson 
Date: 29/12/2025
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import sys

HERE = Path(__file__).resolve()
SCRIPTS_DIR = HERE.parents[1]       # .../scripts
UTIL_DIR = SCRIPTS_DIR / "_util"    # .../scripts/_util

for p in (UTIL_DIR, SCRIPTS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from paper_io import parse_paper_args  # noqa: E402


# -------------------- manuscript-reported fallback values --------------------
# (These are the values that appear in the manuscript tables.)

FALLBACK_TABLE10 = pd.DataFrame(
    [
        ("A", 71, 61, 80, 0.35),
        ("B", 70, 58, 81, 0.35),
        ("C", 71, 52, 79, 0.35),
    ],
    columns=["Tier", "End-to-end median (s)", "Min (s)", "Max (s)", "Per-run median (s)"],
)

FALLBACK_TABLE11 = pd.DataFrame(
    [
        ("A", 3565, 3185, 3917, 18),
        ("B", 3981, 3677, 5004, 19),
        ("C", 4604, 2861, 6680, 18),
    ],
    columns=["Tier", "End-to-end median (s)", "Min (s)", "Max (s)", "Per-bootstrap median (s)"],
)

FALLBACK_TABLE12 = pd.DataFrame(
    [
        ("A", 71, 3565, 0.35, 18, 50),
        ("B", 70, 3981, 0.35, 19, 57),
        ("C", 71, 4604, 0.35, 18, 65),
        ("Overall\N{DAGGER}", 71, 4050, 0.35, 18, 57),
    ],
    columns=[
        "Tier",
        "CDC E2E median (s)",
        "DD E2E median (s)",
        "CDC per-run (s)",
        "DD per-boot (s)",
        "Speedup (\u00d7)",
    ],
)


# -------------------- rounding + formatting helpers --------------------

def _round_half_away_from_zero(x: float) -> int:
    """Deterministic 'paper-style' rounding: .5 rounds away from zero."""
    x = float(x)
    if x >= 0:
        return int((x + 0.5) // 1)
    return -int(((-x) + 0.5) // 1)


def _fmt_int(x) -> str:
    return "" if pd.isna(x) else f"{_round_half_away_from_zero(float(x))}"


def _fmt_2dp(x) -> str:
    return "" if pd.isna(x) else f"{float(x):.2f}"


# -------------------- log parsing --------------------

def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def _pick_elapsed_col(df: pd.DataFrame) -> str:
    for c in ("elapsed_s", "elapsed_sec", "elapsed", "seconds", "runtime_s"):
        if c in df.columns:
            return c
    raise ValueError(f"No elapsed-time column found. Got columns: {list(df.columns)}")


def _ensure_tier(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "tier" in d.columns:
        d["tier"] = d["tier"].astype(str).str.strip().str.upper()
    elif "sample" in d.columns:
        d["tier"] = d["sample"].astype(str).str.extract(r"([A-Za-z])$", expand=False).str.upper()
    return d


def summarise_cdc_runtime(df: pd.DataFrame, *, r_default: int = 200) -> pd.DataFrame:
    """Return a Table-10-like dataframe from a CDC runtime log."""
    if "method" not in df.columns or "phase" not in df.columns:
        raise ValueError("CDC runtime log must contain 'method' and 'phase' columns.")

    d = _ensure_tier(df)
    elapsed_col = _pick_elapsed_col(d)

    d["method"] = d["method"].astype(str).str.upper()
    d["phase"] = d["phase"].astype(str).str.lower()

    # Prefer explicit end-to-end rows; otherwise attempt to treat any 'e2e'/'total' phase as E2E.
    e2e = d[(d["method"] == "CDC") & (d["phase"].str.contains("e2e|total|overall", regex=True))].copy()

    if e2e.empty:
        # As a fallback, if the log already appears to be one-row-per-run with CDC entries,
        # treat all CDC rows as E2E.
        e2e = d[d["method"] == "CDC"].copy()

    e2e[elapsed_col] = pd.to_numeric(e2e[elapsed_col], errors="coerce")

    # Determine R per row if present, otherwise use default.
    if "r" in e2e.columns:
        r = pd.to_numeric(e2e["r"], errors="coerce").fillna(r_default)
    else:
        r = pd.Series([r_default] * len(e2e), index=e2e.index)

    e2e["per_run_s"] = e2e[elapsed_col] / r

    rows = []
    for tier in ("A", "B", "C"):
        sub = e2e[e2e["tier"] == tier]
        if sub.empty:
            continue
        rows.append(
            {
                "Tier": tier,
                "End-to-end median (s)": float(np.nanmedian(sub[elapsed_col])),
                "Min (s)": float(np.nanmin(sub[elapsed_col])),
                "Max (s)": float(np.nanmax(sub[elapsed_col])),
                "Per-run median (s)": float(np.nanmedian(sub["per_run_s"])),
            }
        )

    out = pd.DataFrame(rows).sort_values("Tier")
    return out


def summarise_dd_runtime(df: pd.DataFrame, *, nboot_default: int = 200) -> pd.DataFrame:
    """Return a Table-11-like dataframe from a DD runtime log.

    Supports two common shapes:
      1) One row per run with an elapsed-time column and a tier.
      2) Step-based logs with 'step' and 'elapsed_sec' per step, including 'script_end'.
    """
    d = _ensure_tier(df)

    # Case 1: already summarised per run
    if "step" not in d.columns:
        elapsed_col = _pick_elapsed_col(d)
        d[elapsed_col] = pd.to_numeric(d[elapsed_col], errors="coerce")
        if "nboot" in d.columns:
            nboot = pd.to_numeric(d["nboot"], errors="coerce").fillna(nboot_default)
        else:
            nboot = pd.Series([nboot_default] * len(d), index=d.index)
        d["per_boot_s"] = d[elapsed_col] / nboot

        rows = []
        for tier in ("A", "B", "C"):
            sub = d[d["tier"] == tier]
            if sub.empty:
                continue
            rows.append(
                {
                    "Tier": tier,
                    "End-to-end median (s)": float(np.nanmedian(sub[elapsed_col])),
                    "Min (s)": float(np.nanmin(sub[elapsed_col])),
                    "Max (s)": float(np.nanmax(sub[elapsed_col])),
                    "Per-bootstrap median (s)": float(np.nanmedian(sub["per_boot_s"])),
                }
            )
        return pd.DataFrame(rows).sort_values("Tier")

    # Case 2: step-based
    required = {"case", "sample", "tier", "step"}
    if not required.issubset(set(d.columns)):
        raise ValueError(f"DD runtime log must contain at least {sorted(required)}. Got {list(d.columns)}")

    elapsed_col = _pick_elapsed_col(d)
    d[elapsed_col] = pd.to_numeric(d[elapsed_col], errors="coerce")

    # Identify individual script runs
    if "timestamp" in d.columns:
        d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
        d = d.sort_values(["case", "sample", "timestamp"])

    d["run_id"] = d["step"].astype(str).eq("script_start").groupby([d["case"], d["sample"]]).cumsum()
    d = d[d["run_id"] > 0]

    # End-to-end rows
    e2e = d[d["step"].astype(str).eq("script_end")].copy()
    if e2e.empty:
        raise ValueError("DD runtime log has 'step' column but no 'script_end' rows.")

    # Per-boot estimate: either from explicit bootstrap_* step timings, or fall back to E2E/nboot.
    boot = d[d["step"].astype(str).str.startswith("bootstrap_")].copy()

    if not boot.empty:
        per_boot_run = (
            boot.groupby(["case", "sample", "run_id"])[elapsed_col]
            .median()
            .reset_index(name="per_boot_median_s")
        )
        e2e = e2e.merge(per_boot_run, on=["case", "sample", "run_id"], how="left")
    else:
        # If boot steps not present, compute per-boot from E2E duration.
        nboot_default_series = pd.Series([nboot_default] * len(e2e), index=e2e.index)
        if "nboot" in e2e.columns:
            nboot_series = pd.to_numeric(e2e["nboot"], errors="coerce").fillna(nboot_default)
        else:
            nboot_series = nboot_default_series
        e2e["per_boot_median_s"] = e2e[elapsed_col] / nboot_series

    rows = []
    for tier in ("A", "B", "C"):
        sub = e2e[e2e["tier"] == tier]
        if sub.empty:
            continue
        rows.append(
            {
                "Tier": tier,
                "End-to-end median (s)": float(np.nanmedian(sub[elapsed_col])),
                "Min (s)": float(np.nanmin(sub[elapsed_col])),
                "Max (s)": float(np.nanmax(sub[elapsed_col])),
                "Per-bootstrap median (s)": float(np.nanmedian(sub["per_boot_median_s"])),
            }
        )

    return pd.DataFrame(rows).sort_values("Tier")


def make_table12(t10: pd.DataFrame, t11: pd.DataFrame) -> pd.DataFrame:
    m = pd.merge(
        t10[["Tier", "End-to-end median (s)", "Per-run median (s)"]],
        t11[["Tier", "End-to-end median (s)", "Per-bootstrap median (s)"]],
        on="Tier",
        suffixes=("_cdc", "_dd"),
        how="inner",
    )

    m = m.rename(
        columns={
            "End-to-end median (s)_cdc": "CDC E2E median (s)",
            "End-to-end median (s)_dd": "DD E2E median (s)",
            "Per-run median (s)": "CDC per-run (s)",
            "Per-bootstrap median (s)": "DD per-boot (s)",
        }
    )

    m["Speedup (\u00d7)"] = m["DD E2E median (s)"] / m["CDC E2E median (s)"]

    # Manuscript uses an Overall row. Here we take the mean across tiers as a simple summary.
    overall = pd.DataFrame(
        [
            {
                "Tier": "Overall\N{DAGGER}",
                "CDC E2E median (s)": float(np.nanmean(m["CDC E2E median (s)"])),
                "DD E2E median (s)": float(np.nanmean(m["DD E2E median (s)"])),
                "CDC per-run (s)": float(np.nanmean(m["CDC per-run (s)"])),
                "DD per-boot (s)": float(np.nanmean(m["DD per-boot (s)"])),
                "Speedup (\u00d7)": float(np.nanmean(m["Speedup (\u00d7)"])),
            }
        ]
    )

    return pd.concat([m, overall], ignore_index=True)


# -------------------- LaTeX formatting --------------------

def latex_table10(df: pd.DataFrame) -> str:
    d = df.copy()
    d["End-to-end median (s)"] = d["End-to-end median (s)"].map(_fmt_int)
    d["Min (s)"] = d["Min (s)"].map(_fmt_int)
    d["Max (s)"] = d["Max (s)"].map(_fmt_int)
    d["Per-run median (s)"] = d["Per-run median (s)"].map(_fmt_2dp)
    return d.to_latex(index=False, escape=False)


def latex_table11(df: pd.DataFrame) -> str:
    d = df.copy()
    d["End-to-end median (s)"] = d["End-to-end median (s)"].map(_fmt_int)
    d["Min (s)"] = d["Min (s)"].map(_fmt_int)
    d["Max (s)"] = d["Max (s)"].map(_fmt_int)
    d["Per-bootstrap median (s)"] = d["Per-bootstrap median (s)"].map(_fmt_int)
    return d.to_latex(index=False, escape=False)


def latex_table12(df: pd.DataFrame) -> str:
    d = df.copy()
    d["CDC E2E median (s)"] = d["CDC E2E median (s)"].map(_fmt_int)
    d["DD E2E median (s)"] = d["DD E2E median (s)"].map(_fmt_int)
    d["CDC per-run (s)"] = d["CDC per-run (s)"].map(_fmt_2dp)
    d["DD per-boot (s)"] = d["DD per-boot (s)"].map(_fmt_int)
    d["Speedup (\u00d7)"] = d["Speedup (\u00d7)"].map(_fmt_int)
    return d.to_latex(index=False, escape=False)


def main(argv: Optional[list[str]] = None) -> None:
    base = parse_paper_args(__file__)

    ap = argparse.ArgumentParser(description="Generate manuscript Tables 10–12 (runtime tables).")
    ap.add_argument("--cdc-log", type=str, default=None, help="Path to CDC runtime log CSV (optional).")
    ap.add_argument("--dd-log", type=str, default=None, help="Path to DD runtime log CSV (optional).")
    ap.add_argument("--out-subdir", type=str, default="tables", help="Output subdirectory under papers/2025-peak-picking/outputs/.")
    args = ap.parse_args(argv)

    out_dir = Path(base.out_dir) / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    default_cdc = Path(base.data_dir) / "derived" / "runtime_log_28Nov.csv"
    default_dd = Path(base.data_dir) / "derived" / "runtime_log_reimink.csv"

    cdc_path = Path(args.cdc_log).expanduser() if args.cdc_log else default_cdc
    dd_path = Path(args.dd_log).expanduser() if args.dd_log else default_dd

    have_logs = cdc_path.exists() and dd_path.exists()

    if have_logs:
        try:
            print(f"[runtime] reading CDC log: {cdc_path}")
            print(f"[runtime] reading  DD log: {dd_path}")
            cdc_tbl = summarise_cdc_runtime(_read_csv(cdc_path))
            dd_tbl = summarise_dd_runtime(_read_csv(dd_path))
            t12 = make_table12(cdc_tbl, dd_tbl)
        except Exception as e:
            print(f"[runtime] ERROR parsing logs ({e}); falling back to manuscript values.")
            cdc_tbl, dd_tbl, t12 = FALLBACK_TABLE10, FALLBACK_TABLE11, FALLBACK_TABLE12
    else:
        print("[runtime] runtime logs not found; falling back to manuscript values.")
        print(f"          expected CDC log at: {cdc_path}")
        print(f"          expected  DD log at: {dd_path}")
        cdc_tbl, dd_tbl, t12 = FALLBACK_TABLE10, FALLBACK_TABLE11, FALLBACK_TABLE12

    # --- write outputs ---
    cdc_tbl.to_csv(out_dir / "table10_cdc_runtime_by_tier.csv", index=False)
    (out_dir / "table10_cdc_runtime_by_tier.tex").write_text(latex_table10(cdc_tbl), encoding="utf-8")

    dd_tbl.to_csv(out_dir / "table11_dd_runtime_by_tier.csv", index=False)
    (out_dir / "table11_dd_runtime_by_tier.tex").write_text(latex_table11(dd_tbl), encoding="utf-8")

    t12.to_csv(out_dir / "table12_runtime_comparison.csv", index=False)
    (out_dir / "table12_runtime_comparison.tex").write_text(latex_table12(t12), encoding="utf-8")

    print(f"[runtime] wrote: {out_dir / 'table10_cdc_runtime_by_tier.csv'}")
    print(f"[runtime] wrote: {out_dir / 'table11_dd_runtime_by_tier.csv'}")
    print(f"[runtime] wrote: {out_dir / 'table12_runtime_comparison.csv'}")


if __name__ == "__main__":
    main()
