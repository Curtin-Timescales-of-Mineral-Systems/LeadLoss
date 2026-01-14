#!/usr/bin/env python3
"""tables_09_concordant_fraction_sweep.py

Generate manuscript **Table 9**:
  "Concordant fraction sweep results for the single-loss geometry (25 Ma lower intercept)."

Inputs
------
Default: built-in manuscript-reported values (Table 9, v1.2).

Optional override:
  --source-csv PATH
    A CSV with the following columns (case-insensitive):
      c_pct,
      legacy_age_ma, legacy_ci_lo_ma, legacy_ci_hi_ma,
      ens_age_ma,    ens_ci_lo_ma,    ens_ci_hi_ma

Outputs
-------
Writes into:
  <paper>/outputs/tables/
    table09_concordant_fraction_sweep_raw.csv   (machine-readable numeric)
    table09_concordant_fraction_sweep.csv       (publication-formatted; human-friendly)
    table09_concordant_fraction_sweep.tex       (LaTeX tabular)

Dependencies
------------
  - pandas

Author: Lucy Mathieson
Date: 29/12/2025
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

import sys

HERE = Path(__file__).resolve()
SCRIPTS_DIR = HERE.parents[1]       # .../scripts
UTIL_DIR = SCRIPTS_DIR / "_util"    # .../scripts/_util

for p in (UTIL_DIR, SCRIPTS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from paper_io import parse_paper_args  # noqa: E402


def _round_half_away_from_zero(x: float) -> int:
    """Deterministic 'paper-style' rounding: .5 rounds away from zero."""
    x = float(x)
    if x >= 0:
        return int((x + 0.5) // 1)
    return -int(((-x) + 0.5) // 1)


def _default_table9_v12() -> pd.DataFrame:
    """Manuscript Table 9 values (v1.2)."""
    return pd.DataFrame(
        [
            # C (%), legacy age, legacy CI lo, legacy CI hi, ensemble age, ensemble CI lo, ensemble CI hi
            (5,  26, 21, 46, 33, 27, 58),
            (10, 26, 21, 36, 25, 13, 38),
            (20, 26, 21, 41, 24, 13, 38),
            (30, 26, 21, 36, 24, 13, 38),
            (40, 26, 21, 26, 24, 13, 38),
            (60, 26, 21, 26, 24, 13, 38),
        ],
        columns=[
            "c_pct",
            "legacy_age_ma",
            "legacy_ci_lo_ma",
            "legacy_ci_hi_ma",
            "ens_age_ma",
            "ens_ci_lo_ma",
            "ens_ci_hi_ma",
        ],
    )


def _load_from_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {str(c).strip().lower(): c for c in df.columns}

    def need(name: str) -> str:
        if name not in cols:
            raise SystemExit(
                f"[Table 9] Missing required column '{name}' in {path.name}. "
                f"Got: {list(df.columns)}"
            )
        return cols[name]

    out = pd.DataFrame(
        {
            "c_pct": df[need("c_pct")],
            "legacy_age_ma": df[need("legacy_age_ma")],
            "legacy_ci_lo_ma": df[need("legacy_ci_lo_ma")],
            "legacy_ci_hi_ma": df[need("legacy_ci_hi_ma")],
            "ens_age_ma": df[need("ens_age_ma")],
            "ens_ci_lo_ma": df[need("ens_ci_lo_ma")],
            "ens_ci_hi_ma": df[need("ens_ci_hi_ma")],
        }
    )

    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna(subset=["c_pct"]).sort_values("c_pct")
    return out


def _to_public_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Format columns to match the manuscript Table 9 layout."""
    d = df_raw.copy()

    def as_int(x) -> str:
        return "" if pd.isna(x) else str(_round_half_away_from_zero(float(x)))

    def ci(lo, hi) -> str:
        if pd.isna(lo) or pd.isna(hi):
            return ""
        # LaTeX en-dash is rendered with `--` (we also use it in the CSV for readability).
        return f"{_round_half_away_from_zero(float(lo))}--{_round_half_away_from_zero(float(hi))}"

    return pd.DataFrame(
        {
            "C (%)": d["c_pct"].map(lambda x: "" if pd.isna(x) else f"{_round_half_away_from_zero(float(x))}%"),
            "Legacy age (Ma)": d["legacy_age_ma"].map(as_int),
            "Legacy 95% CI (Ma)": [ci(lo, hi) for lo, hi in zip(d["legacy_ci_lo_ma"], d["legacy_ci_hi_ma"])],
            "Ensemble age (Ma)": d["ens_age_ma"].map(as_int),
            "Ensemble 95% CI (Ma)": [ci(lo, hi) for lo, hi in zip(d["ens_ci_lo_ma"], d["ens_ci_hi_ma"])],
        }
    )


def main(argv: Optional[list[str]] = None) -> None:
    base = parse_paper_args(__file__)
    out_dir = Path(base.out_dir) / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    ap = argparse.ArgumentParser(description="Generate Table 9 (concordant fraction sweep).")
    ap.add_argument("--source-csv", type=str, default=None, help="Optional CSV with raw sweep results.")
    args = ap.parse_args(argv)

    if args.source_csv:
        src = Path(args.source_csv).expanduser().resolve()
        if not src.exists():
            raise SystemExit(f"[Table 9] --source-csv not found: {src}")
        print(f"[Table 9] reading source CSV: {src.name}")
        df_raw = _load_from_csv(src)
    else:
        print("[Table 9] using manuscript-reported values (built-in, v1.2).")
        df_raw = _default_table9_v12()

    df_public = _to_public_df(df_raw)

    # Machine-readable version (numeric columns)
    df_raw.to_csv(out_dir / "table09_concordant_fraction_sweep_raw.csv", index=False)
    # Publication-formatted version
    df_public.to_csv(out_dir / "table09_concordant_fraction_sweep.csv", index=False)

    # LaTeX tabular (standalone tabular, no table environment).
    # IMPORTANT: escape=True so '%' in headers and values becomes '\%' for LaTeX.
    (out_dir / "table09_concordant_fraction_sweep.tex").write_text(
        df_public.to_latex(index=False, escape=True),
        encoding="utf-8",
    )

    print(f"[Table 9] wrote: {out_dir / 'table09_concordant_fraction_sweep.csv'}")
    print(f"[Table 9] wrote: {out_dir / 'table09_concordant_fraction_sweep.tex'}")


if __name__ == "__main__":
    main()
