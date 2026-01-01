"""benchmark_definitions.py

Single source of truth for the synthetic benchmark definitions used in:
  - Table 1 (Cases 1 to 4)
  - Table 2 (Cases 5 to 7)
  - Synthetic data generators (synth_tw_native*.py)
  - Benchmark scoring scripts (tables_01_02_benchmark_results.py)

All ages are in Ma.
"""

from __future__ import annotations

from typing import Dict, List, Sequence


TIERS: List[str] = ["A", "B", "C"]
TIER_TO_REI = {"A": "a", "B": "b", "C": "c"}

# True event ages for scoring (per case)
CASES_TRUE: Dict[str, List[float]] = {
    "1": [700.0],
    "2": [300.0, 1800.0],
    "3": [400.0],
    "4": [500.0, 1800.0],
    "5": [500.0, 1500.0],
    "6": [500.0, 1500.0],
    "7": [500.0, 1500.0],
}

# ---------------- Table 1: Cases 1 to 4 ----------------
# Each case is a list of single-step chords (t_up -> t_low)
CASES_1TO4_CHORDS: Dict[int, List[dict]] = {
    1: [dict(t_up=3000, t_low=700)],
    2: [dict(t_up=3200, t_low=1800), dict(t_up=3200, t_low=300)],
    3: [dict(t_up=3200, t_low=400), dict(t_up=3000, t_low=400), dict(t_up=2800, t_low=400)],
    4: [dict(t_up=3200, t_low=500), dict(t_up=3000, t_low=1800), dict(t_up=2800, t_low=1800)],
}

# Optional chord weights (fractions). If missing, equal weights are assumed.
CASES_1TO4_WEIGHTS: Dict[int, List[float]] = {
    1: [1.0],
    2: [0.5, 0.5],
    3: [0.33, 0.33, 0.34],
    4: [0.50, 0.25, 0.25],
}

# ---------------- Table 2: Cases 5 to 7 ----------------
# Chords available for each case
CASES_5TO7_CHORDS: Dict[int, List[dict]] = {
    # Case 5 uses one crystallisation age (3000 Ma)
    5: [dict(t_up=3000, t_low=1500), dict(t_up=3000, t_low=500)],
    # Case 6 uses one crystallisation age (3000 Ma)
    6: [dict(t_up=3000, t_low=1500), dict(t_up=3000, t_low=500)],
    # Case 7 superimposes two crystallisation ages (3200 and 3000 Ma)
    7: [dict(t_up=3200, t_low=1500), dict(t_up=3000, t_low=1500), dict(t_up=3200, t_low=500)],
}

# Subpopulations: each entry is a "trajectory", represented as a list of 1 or 2 chords.
# Two-chord trajectories represent two sequential Pb-loss steps (t_mid then t_low).
CASES_5TO7_SUBPOPS: Dict[int, List[List[dict]]] = {
    5: [
        [CASES_5TO7_CHORDS[5][0]],                         # 3000 -> 1500
        [CASES_5TO7_CHORDS[5][0], CASES_5TO7_CHORDS[5][1]] # 3000 -> 1500 -> 500
    ],
    6: [
        [CASES_5TO7_CHORDS[6][0]],                         # 3000 -> 1500
        [CASES_5TO7_CHORDS[6][1]],                         # 3000 -> 500
        [CASES_5TO7_CHORDS[6][0], CASES_5TO7_CHORDS[6][1]] # 3000 -> 1500 -> 500
    ],
    7: [
        [CASES_5TO7_CHORDS[7][0]],                         # 3200 -> 1500
        [CASES_5TO7_CHORDS[7][1]],                         # 3000 -> 1500
        [CASES_5TO7_CHORDS[7][0], CASES_5TO7_CHORDS[7][2]] # 3200 -> 1500 -> 500
    ],
}

# Subpopulation weights (fractions). Must sum to 1 per case.
CASES_5TO7_SUBPOP_WEIGHTS: Dict[int, List[float]] = {
    5: [0.60, 0.40],
    6: [0.40, 0.30, 0.30],
    7: [0.40, 0.30, 0.30],
}

