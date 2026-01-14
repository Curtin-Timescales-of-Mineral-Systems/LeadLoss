#!/usr/bin/env python3
"""tables_01_02_benchmark_definitions.py

Generate manuscript Table 1 and Table 2 (benchmark scenario definitions) as CSV and LaTeX.

Source of truth: scripts/tables/benchmark_definitions.py

Outputs (default):
  <paper>/outputs/tables/
    table1_single_stage_definitions.csv/.tex
    table2_multi_stage_definitions.csv/.tex

Author: Lucy Mathieson
Date: 29/12/2025
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Make scripts/_util importable when running this file directly
UTIL_DIR = Path(__file__).resolve().parents[1] / "_util"  # .../scripts/_util
if str(UTIL_DIR) not in sys.path:
    sys.path.insert(0, str(UTIL_DIR))

from paper_io import parse_paper_args  # noqa: E402
from benchmark_definitions import (  # noqa: E402
    CASES_1TO4_CHORDS,
    CASES_1TO4_WEIGHTS,
    CASES_5TO7_SUBPOPS,
    CASES_5TO7_SUBPOP_WEIGHTS,
    TIERS,
)


def _round_half_away_from_zero(x: float) -> int:
    x = float(x)
    if x >= 0:
        return int((x + 0.5) // 1)
    return -int(((-x) + 0.5) // 1)


def _arrow_path(t_up: int, events) -> str:
    """Render Pb-loss trajectory as 't_up -> t_low -> ...'."""
    if isinstance(events, (list, tuple)):
        parts = [str(int(t_up))] + [str(int(e)) for e in events]
    else:
        parts = [str(int(t_up)), str(int(events))]
    return " -> ".join(parts)


def main() -> None:
    base = parse_paper_args(__file__)
    out_dir = Path(base.out_dir) / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    tier_label = f"{TIERS[0]}--{TIERS[-1]}"  # LaTeX-friendly en dash

    # ---------------- Table 1 ----------------
    rows1 = []
    for case in sorted(CASES_1TO4_CHORDS.keys()):
        chords = CASES_1TO4_CHORDS[case]
        w = CASES_1TO4_WEIGHTS.get(case, [1.0 / len(chords)] * len(chords))
        w = np.asarray(w, float)
        if w.size != len(chords):
            raise SystemExit(f"[Table 1] weights length mismatch for case {case}: {w.size} vs {len(chords)}")
        # Normalise defensively (should already sum to 1)
        wsum = float(np.sum(w))
        if wsum > 0:
            w = w / wsum

        for i, (c, wi) in enumerate(zip(chords, w), 1):
            rows1.append(
                dict(
                    case=str(case),
                    tier=tier_label,
                    trajectory=f"({i})",
                    pb_loss=_arrow_path(c["t_up"], [c["t_low"]]),
                    weight_pct=100.0 * float(wi),
                )
            )

    t1 = pd.DataFrame(rows1)
    t1["weight_pct"] = t1["weight_pct"].map(lambda v: _round_half_away_from_zero(v)).astype(int)
    t1.to_csv(out_dir / "table1_single_stage_definitions.csv", index=False)
    (out_dir / "table1_single_stage_definitions.tex").write_text(
        t1.to_latex(index=False, escape=True),
        encoding="utf-8",
    )

    # ---------------- Table 2 ----------------
    rows2 = []
    for case in sorted(CASES_5TO7_SUBPOPS.keys()):
        subpops = CASES_5TO7_SUBPOPS[case]
        weights = np.asarray(CASES_5TO7_SUBPOP_WEIGHTS[case], float)
        if weights.size != len(subpops):
            raise SystemExit(f"[Table 2] weights length mismatch for case {case}: {weights.size} vs {len(subpops)}")
        wsum = float(np.sum(weights))
        if wsum > 0:
            weights = weights / wsum

        for i, (traj, wi) in enumerate(zip(subpops, weights), 1):
            # Each traj is a list of chord dicts; represent as t_up -> t_low -> t_low2
            t_up = int(traj[0]["t_up"])
            lows = [int(ch["t_low"]) for ch in traj]
            rows2.append(
                dict(
                    case=str(case),
                    tier=tier_label,
                    trajectory=f"({i})",
                    pb_loss=_arrow_path(t_up, lows),
                    weight_pct=100.0 * float(wi),
                )
            )

    t2 = pd.DataFrame(rows2)
    t2["weight_pct"] = t2["weight_pct"].map(lambda v: _round_half_away_from_zero(v)).astype(int)
    t2.to_csv(out_dir / "table2_multi_stage_definitions.csv", index=False)
    (out_dir / "table2_multi_stage_definitions.tex").write_text(
        t2.to_latex(index=False, escape=True),
        encoding="utf-8",
    )

    print(f"Wrote: {out_dir / 'table1_single_stage_definitions.tex'}")
    print(f"Wrote: {out_dir / 'table2_multi_stage_definitions.tex'}")


if __name__ == "__main__":
    main()
