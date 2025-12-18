#!/usr/bin/env python3
"""
Synthetic U–Pb generator that draws DISCORDANT arrays natively in
Tera–Wasserburg (TW) space, then converts to Wetherill for filtering
and for the second app.

- Concordant grains: unchanged (drawn by age on Wetherill concordia).
- Discordant grains: straight lines in TW with Gaussian along/perp scatter.
- Exports: Wetherill-style raw, Reimink-style, and TW CSV.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.random import default_rng
from sklearn.linear_model import RANSACRegressor, LinearRegression

# ============== 1. GLOBAL CONSTANTS & HELPERS =================================
ERR_REL_RANGE = (0.005, 0.015)     # 0.5–1.5 %   (1σ on each ratio)
ERR_X_MULT    = 1.0
ERR_Y_MULT    = 1.5                # widen σy so ellipses are not needles
RHO_CONST     = 0.85               # used in Reimink export (if needed)

ERR_X_MULT_CONC = 1.0              # concordant grains – keep realistic
ERR_Y_MULT_CONC = 1.0

# ------------------------------------------------------------------
SEED = 42
rng  = default_rng(SEED)

# Decay constants (1/Ma)
L235 = 9.8485e-4
L238 = 1.55125e-4

def wetherill_xy(t_ma: float) -> Tuple[float, float]:
    if t_ma <= 0:
        return (0.0, 0.0)
    return (math.expm1(L235 * t_ma), math.expm1(L238 * t_ma))

def age_207_235(x: float) -> float:
    return math.log1p(x)/L235 if x>0 else 0.0

def age_206_238(y: float) -> float:
    return math.log1p(y)/L238 if y>0 else 0.0

def fractional_discordance(x: float, y: float) -> float:
    tA = age_207_235(x)
    tB = age_206_238(y)
    if tA<=0 or tB<=0:
        return 0.0
    return 1.0 - min(tA, tB)/max(tA, tB)

def is_reversed(x: float, y: float) -> bool:
    return (age_206_238(y) > age_207_235(x))

def conc_y_from_x(x: float) -> float:
    t = age_207_235(x)
    return math.expm1(L238 * t) if t>0 else 0.0

# ============== 2. DEFINITIONS: Tiers & Cases =================================

@dataclass
class Tier:
    name: str
    sig_perp: float
    sig_along: float
    noise_frac: Tuple[float, float]

@dataclass
class Chord:
    t_up: float
    t_low: float
    f_min: float = 0.0
    f_max: float = 0.95
    sig_perp_override: Optional[float] = None
    sig_along_override: Optional[float] = None

@dataclass
class CaseDef:
    name: str
    chords: List[Chord]
    chord_weights: Optional[List[float]] = None

TIERS: Dict[str, Tier] = {
    'A': Tier('A', sig_perp=0.000, sig_along=0.000, noise_frac=(0.001, 0.001)),
    'B': Tier('B', sig_perp=0.005, sig_along=0.005, noise_frac=(0.005, 0.01)),
    'C': Tier('C', sig_perp=0.010, sig_along=0.010, noise_frac=(0.005, 0.02)),
}

CASES: Dict[int, CaseDef] = {
    1: CaseDef(
        name='1',
        chords=[Chord(t_up=3000, t_low=700, f_min=0.0, f_max=0.95)]
    ),
    2: CaseDef(
        name='2',
        chords=[
            Chord(t_up=3100, t_low=1800, f_min=0.0, f_max=0.95),
            Chord(t_up=3100, t_low=300,   f_min=0.0, f_max=0.95),
        ],
        chord_weights=[0.5, 0.5]
    ),
    3: CaseDef(
        name='3', #3
        chords=[
            Chord(t_up=1100, t_low=25, f_min=0.0, f_max=0.95), #3200 400
            Chord(t_up=1400, t_low=25, f_min=0.0, f_max=0.95), #3000 400
            Chord(t_up=1700, t_low=25, f_min=0.0, f_max=0.95), #2800 400
        ]
    ),
    4: CaseDef(
        name='4',
        chords=[
            Chord(t_up=3200, t_low=500,  f_min=0.0, f_max=0.95),
            Chord(t_up=3100, t_low=1800, f_min=0.0, f_max=0.95),
            Chord(t_up=3000, t_low=1800, f_min=0.0, f_max=0.95),
        ],
        chord_weights=[0.5, 0.25, 0.25]
    ),
}

# ============== 3. GENERATION PARAMETERS & FILTERS ============================

N_POINTS = 160
CONC_FRAC = 0.25

SPREAD_SINGLE = 5.0
SPREAD_MULTI  = 15.0

MIN_DISCORDANCE = 0.01
CLEARANCE_FRAC  = 0.03

R_TW = 137.818                 # 238U/235U
TW_CAP = 0.26                  # cap in p = 207Pb/206Pb

def passes_filter(x: float, y: float, xerr: float, yerr: float) -> bool:
    if x <= 0 or y <= 0:
        return False
    if is_reversed(x, y):
        return False

    disc0 = fractional_discordance(x, y)
    if disc0 < MIN_DISCORDANCE:
        return False

    # worst-case with measurement error (push toward concordia)
    xw = max(0, x - xerr)
    yw = y + yerr
    disc1 = fractional_discordance(xw, yw)
    if disc1 < MIN_DISCORDANCE:
        return False

    # Clearance below concordia
    yw_conc = conc_y_from_x(xw)
    if yw >= yw_conc * (1.0 - CLEARANCE_FRAC):
        return False

    # TW Pb/Pb cap
    twPb = (x / y) / R_TW
    if twPb > TW_CAP:
        return False

    return True

# ============== 4. TW HELPERS (transform, line, concordia) ====================

def wetherill_to_TW_xy(x: float, y: float) -> Tuple[float, float]:
    """(x,y) -> (u,p) with u = 238U/206Pb, p = 207Pb/206Pb."""
    if y <= 0:
        return (np.nan, np.nan)
    u = 1.0 / y
    p = (x / y) / R_TW
    return (u, p)

def TW_to_wetherill(u: float, p: float) -> Tuple[float, float]:
    """(u,p) -> (x,y) with p = (x/y)/R_TW and u = 1/y."""
    if u <= 0:
        return (np.nan, np.nan)
    y = 1.0 / u
    x = (p * R_TW) * y
    return (x, y)

def jacobian_xy_to_TW(x: float, y: float) -> np.ndarray:
    """Jacobian of (u,p) wrt (x,y) for covariance propagation."""
    # u = 1/y ; p = (R_TW*x)/y
    return np.array([
        [0.0,       -1.0/(y*y)],
        [R_TW/y,    -R_TW*x/(y*y)]
    ])

def TW_concordia_p(u: float) -> float:
    """Radiogenic 207Pb/206Pb on TW concordia at abscissa u."""
    if u <= 0:
        return np.nan
    y = 1.0/u
    t = age_206_238(y)
    x = math.expm1(L235*t)
    return (x / y) / R_TW

def chord_to_TW_line(ch: Chord) -> Tuple[float, float, float, float]:
    """
    Return (A, B, u_min, u_max) for the TW line p = A*u + B corresponding
    to the Wetherill chord between t_up and t_low, clipped to [f_min, f_max].
    """
    xU, yU = wetherill_xy(ch.t_up)
    xL, yL = wetherill_xy(ch.t_low)
    dx, dy = (xL - xU), (yL - yU)

    if abs(dy) < 1e-15:
        # Fallback: use the two TW points directly
        uU, pU = 1.0 / yU, (xU / yU) / R_TW
        uL, pL = 1.0 / yL, (xL / yL) / R_TW
        A = (pL - pU) / (uL - uU)
        B = pU - A * uU
    else:
        # Correct 1/R_TW form:
        # x(f) = xU + f*dx, y(f) = yU + f*dy, u = 1/y, p = (x/y)/R_TW
        #  => p(u) = [(xU - (dx/dy)*yU)/R_TW] * u + [(dx/dy)/R_TW]
        A = (xU - (dx / dy) * yU) / R_TW
        B = (dx / dy) / R_TW

    def u_of_f(f: float) -> float:
        y = yU + f * dy
        return (1.0 / y) if y > 0 else np.nan

    u0 = u_of_f(ch.f_min)
    u1 = u_of_f(ch.f_max)
    u_min, u_max = (min(u0, u1), max(u0, u1))
    return A, B, u_min, u_max

def f_at_discordance(ch: Chord, d_target: float, n: int = 1024) -> float:
    """
    Smallest f in [f_min, f_max] (from the upper intercept) on the Wetherill chord
    where fractional_discordance >= d_target. Works even though discordance is 0 at both
    ends and peaks in the interior.
    """
    xU, yU = wetherill_xy(ch.t_up)
    xL, yL = wetherill_xy(ch.t_low)
    dx, dy = (xL - xU), (yL - yU)

    def disc_at(f: float) -> float:
        x = xU + f * dx
        y = yU + f * dy
        return fractional_discordance(x, y)

    # 1) coarse grid to find the first crossing
    fs = np.linspace(ch.f_min, ch.f_max, n)
    discs = np.array([disc_at(f) for f in fs])

    if discs.max() < d_target:
        # chord never reaches the threshold => allow the whole interval
        return ch.f_min

    idxs = np.where(discs >= d_target)[0]
    k = int(idxs[0])  # first index where condition is met

    # 2) refine by bisection between fs[k-1] (below) and fs[k] (above)
    f_lo = fs[k - 1] if k > 0 else fs[k] * 0.5
    f_hi = fs[k]
    for _ in range(50):
        mid = 0.5 * (f_lo + f_hi)
        if disc_at(mid) >= d_target:
            f_hi = mid
        else:
            f_lo = mid
    return f_hi

# ============== 5. SYNTHESIS FUNCTIONS =======================================

# TW-native controls
D_MARGIN = 0.005   # little safety above MIN_DISCORDANCE (e.g., +0.5%)

def make_one_discordant_TW(chord: Chord, tier: Tier) -> Tuple[float,float,float,float,float,float]:
    """
    Draw one discordant point by sampling directly on the TW line for the chord,
    adding along/perp Gaussian scatter, then converting back to Wetherill and
    applying the existing passes_filter().
    """
    # Trim near-concordia end by the discordance threshold
    D_MARGIN = 0.005  # keep a bit beyond your MIN_DISCORDANCE
    f_threshold = f_at_discordance(chord, MIN_DISCORDANCE + D_MARGIN)
    f_lo = max(chord.f_min, f_threshold)
    f_hi = chord.f_max
    if not (f_lo < f_hi):
        return (np.nan,)*6

    # u-range from the trimmed f-range
    xU, yU = wetherill_xy(chord.t_up)
    xL, yL = wetherill_xy(chord.t_low)
    dx, dy = (xL - xU), (yL - yU)
    def u_of_f(f):
        y = yU + f * dy
        return (1.0 / y) if y > 0 else np.nan
    u_min = min(u_of_f(f_lo), u_of_f(f_hi))
    u_max = max(u_of_f(f_lo), u_of_f(f_hi))
    if not (np.isfinite(u_min) and np.isfinite(u_max) and u_min < u_max):
        return (np.nan,)*6

    A, B, _, _ = chord_to_TW_line(chord)

    # Scatter scales in orthonormal (u,p) coordinates
    sig_perp  = chord.sig_perp_override  if chord.sig_perp_override  is not None else tier.sig_perp
    sig_along = chord.sig_along_override if chord.sig_along_override is not None else tier.sig_along

    # Unit vectors along/perp the TW line
    norm = math.hypot(1.0, A)
    e_par  = (1.0 / norm, A / norm)
    e_perp = (-A / norm, 1.0 / norm)

    for _attempt in range(1000):
        # (1) sample a location on the straight line in TW
        u  = rng.uniform(u_min, u_max)
        p  = A * u + B

        # (2) add geometric scatter (anisotropic)
        u_geo = u + rng.normal(0.0, sig_along) * e_par[0] + rng.normal(0.0, sig_perp) * e_perp[0]
        p_geo = p + rng.normal(0.0, sig_along) * e_par[1] + rng.normal(0.0, sig_perp) * e_perp[1]
        if not (u_geo > 0):
            continue

        # (3) convert back to Wetherill
        x_geo, y_geo = TW_to_wetherill(u_geo, p_geo)
        if not (x_geo > 0 and y_geo > 0):
            continue

        # (4) measurement errors drawn in Wetherill (as before)
        pct   = rng.uniform(*ERR_REL_RANGE)
        x_err = abs(x_geo) * pct * ERR_X_MULT
        y_err = abs(y_geo) * pct * ERR_Y_MULT

        # (5) reuse filter and p-cap
        if ((x_geo / y_geo) / R_TW) > TW_CAP:
            continue
        if passes_filter(x_geo, y_geo, x_err, y_err):
            return (x_geo, y_geo, x_err, y_err, chord.t_up, chord.t_low)

    return (np.nan,)*6

def build_concordant_grains(case_def: CaseDef, tier: Tier, n_conc: int, floor_at_t_up: bool = True) -> pd.DataFrame:
    rows = []
    chords = case_def.chords
    if len(chords)==1:
        t_up_list = [chords[0].t_up]
        spread = SPREAD_SINGLE
    else:
        t_up_list = [ch.t_up for ch in chords]
        spread = SPREAD_MULTI

    # distribute among chords
    if case_def.chord_weights and len(case_def.chord_weights)==len(t_up_list):
        w = np.array(case_def.chord_weights, float)
        w /= w.sum()
        ccounts = np.floor(w*n_conc).astype(int)
        remainder = n_conc - ccounts.sum()
        for i in range(remainder):
            ccounts[i%len(ccounts)] += 1
    else:
        base, rem = divmod(n_conc, len(t_up_list))
        ccounts = [base]*len(t_up_list)
        for i in range(rem):
            ccounts[i]+=1

    for i, tup_val in enumerate(t_up_list):
        for _ in range(ccounts[i]):
            tc = rng.normal(tup_val, spread)
            if floor_at_t_up:
                tc = max(tc, tup_val)   # <-- prevent concordants younger than the upper intercept
            if tc < 0:
                tc = 0
            xC, yC = wetherill_xy(tc)

            # measurement error
            pct   = rng.uniform(*ERR_REL_RANGE)
            x_err = abs(xC) * pct * ERR_X_MULT_CONC
            y_err = abs(yC) * pct * ERR_Y_MULT_CONC

            rows.append([xC, yC, x_err, y_err, tup_val, np.nan, True])

    cols = ['x','y','x_err','y_err','t_up_true','t_low_true','is_concordant']
    return pd.DataFrame(rows, columns=cols)

def build_discordant_grains(case_def: CaseDef, tier: Tier, n_disc: int) -> pd.DataFrame:
    rows = []
    chords = case_def.chords
    n_chord = len(chords)
    if case_def.chord_weights and len(case_def.chord_weights)==n_chord:
        w = np.array(case_def.chord_weights, float)
        w /= w.sum()
        ccounts = np.floor(w*n_disc).astype(int)
        remainder = n_disc - ccounts.sum()
        for i in range(remainder):
            ccounts[i%len(ccounts)] += 1
    else:
        base, rem = divmod(n_disc, n_chord)
        ccounts = [base]*n_chord
        for i in range(rem):
            ccounts[i]+=1

    # Make sure we actually reach the target count per chord
    for i, chord in enumerate(chords):
        need = ccounts[i]
        made = 0
        safety = 0
        while made < need and safety < need * 50:
            xg, yg, xerr, yerr, tup, tlow = make_one_discordant_TW(chord, tier)
            if not (np.isnan(xg) or np.isnan(yg)):
                rows.append([xg, yg, xerr, yerr, tup, tlow, False])
                made += 1
            safety += 1

    cols = ['x','y','x_err','y_err','t_up_true','t_low_true','is_concordant']
    return pd.DataFrame(rows, columns=cols)

def simulate_case(case_def: CaseDef, tier: Tier) -> pd.DataFrame:
    """
    Creates one synthetic dataset (panel) for a given (case, tier).
    """
    n_conc = int(N_POINTS * CONC_FRAC)
    n_disc = N_POINTS - n_conc

    df_conc = build_concordant_grains(case_def, tier, n_conc)
    df_disc = build_discordant_grains(case_def, tier, n_disc)
    df = pd.concat([df_conc, df_disc], ignore_index=True)
    df['Case'] = case_def.name
    df['Tier'] = tier.name
    return df

def simulate_all_cases() -> List[pd.DataFrame]:
    panels = []
    for i_case in [1,2,3,4]:
        cdef = CASES[i_case]
        for tiername in ['A','B','C']:
            tier = TIERS[tiername]
            dfp = simulate_case(cdef, tier)
            panels.append(dfp)
    return panels

# ============== 6. PLOTTING (Wetherill; optional TW plot helper) ==============

def plot_grid(panels: List[pd.DataFrame]):
    fig, axes = plt.subplots(4, 3, figsize=(10,12), sharex=True, sharey=True)

    tvals = np.linspace(0, 4500, 300)
    cx, cy = [], []
    for tt in tvals:
        xx, yy = wetherill_xy(tt)
        cx.append(xx)
        cy.append(yy)

    for df_panel in panels:
        cnum = df_panel['Case'].iat[0]
        tname= df_panel['Tier'].iat[0]
        row = int(cnum)-1
        col = ['A','B','C'].index(tname)
        ax = axes[row,col]
        ax.plot(cx, cy, color='0.7', lw=1.0)
        cdef = CASES[int(cnum)]
        for ch in cdef.chords:
            xU, yU = wetherill_xy(ch.t_up)
            xL, yL = wetherill_xy(ch.t_low)
            ax.plot([xU, xL],[yU, yL], color='0.8', ls='--', lw=1.0)

        ax.scatter(df_panel['x'], df_panel['y'], s=10,
                   facecolors='none', edgecolors='tab:blue')

        ax.set_xlim(0, 25)
        ax.set_ylim(0, 1.0)
        ax.set_title(f"Case {cnum}, Tier {tname}", fontsize=9)

        if row==3:
            ax.set_xlabel(r"$^{207}\mathrm{Pb}/^{235}\mathrm{U}$", fontsize=9)
        if col==0:
            ax.set_ylabel(r"$^{206}\mathrm{Pb}/^{238}\mathrm{U}$", fontsize=9)

    fig.suptitle("Synthetic U–Pb (Wetherill): 4 Cases × 3 Tiers (TW-native discordant)", fontsize=12)
    fig.tight_layout()
    return fig

def plot_tw_grid(panels: List[pd.DataFrame]):
    """
    Optional: p (207/206) vs u (238U/206Pb) view to confirm straightness.
    """
    fig, axes = plt.subplots(4, 3, figsize=(10,12), sharex=True, sharey=True)

    # TW concordia sampled by t
    tvals = np.linspace(1.0, 4500, 500)
    uC, pC = [], []
    for tt in tvals:
        x, y = wetherill_xy(tt)
        u = 1.0/max(y, 1e-12)
        p = (x/max(y, 1e-12))/R_TW
        uC.append(u)
        pC.append(p)

    for df_panel in panels:
        cnum = df_panel['Case'].iat[0]
        tname= df_panel['Tier'].iat[0]
        row = int(cnum)-1
        col = ['A','B','C'].index(tname)
        ax = axes[row,col]
        ax.plot(uC, pC, color='0.7', lw=1.0)

        # TW chord lines
        cdef = CASES[int(cnum)]
        for ch in cdef.chords:
            A, B, u_min, u_max = chord_to_TW_line(ch)
            uu = np.linspace(u_min, u_max, 50)
            ax.plot(uu, A*uu + B, color='0.8', ls='--', lw=1.0)

        # data (convert panel to TW)
        uvals = 1.0 / df_panel['y'].values
        pvals = (df_panel['x'].values / df_panel['y'].values) / R_TW
        ax.scatter(uvals, pvals, s=10, facecolors='none', edgecolors='tab:blue')

        ax.set_xlim(0, 6)          # tweak if you need a broader u-range
        ax.set_ylim(0, 0.30)
        ax.set_title(f"Case {cnum}, Tier {tname}", fontsize=9)
        if row==3:
            ax.set_xlabel(r"$^{238}\mathrm{U}/^{206}\mathrm{Pb}$ (u)", fontsize=9)
        if col==0:
            ax.set_ylabel(r"$^{207}\mathrm{Pb}/^{206}\mathrm{Pb}$ (p)", fontsize=9)

    fig.suptitle("Synthetic U–Pb (Tera–Wasserburg): 4 Cases × 3 Tiers", fontsize=12)
    fig.tight_layout()
    return fig

# ============== 7. EXPORTS & METRICS ==========================================

def to_teraW(df: pd.DataFrame) -> pd.DataFrame:
    """Return one CDC-ready Tera–Wasserburg CSV block."""
    U_val = 1.0 / df['y']
    U_err = df['y_err'] / (df['y'] ** 2)
    p_val = (df['x'] / df['y']) / R_TW
    p_err = np.sqrt(
        (df['x_err'] / (df['y'] * R_TW)) ** 2 +
        ((df['x'] * df['y_err']) / (df['y'] ** 2 * R_TW)) ** 2
    )
    out = pd.DataFrame({
        'Sample': df['Sample'],
        'uPbValue': U_val,
        'uPbError': U_err,
        'pbPbValue': p_val,
        'pbPbError': p_err
    })
    return out

def to_reimink(df: pd.DataFrame) -> pd.DataFrame:
    """Return one Reimink-style CSV block."""
    age76 = np.log1p(df['x']) / L235
    out = pd.DataFrame({
        'Sample': df['Sample'],
        'Pb7_U35': df['x'],
        'Pb7_U35.2SE.abs': 2.0 * df['x_err'],
        'Pb6_U38': df['y'],
        'Pb6_U38.2SE.abs': 2.0 * df['y_err'],
        'rho': RHO_CONST,
        'age76': age76
    })
    return out

EPS, MIN_INLIERS, MAX_LINES = 0.008, 8, 5

def linearity_L(df_panel, eps=EPS, min_inliers=MIN_INLIERS, max_lines=MAX_LINES):
    disc = df_panel[df_panel['t_low_true'].notna()]
    if len(disc) < min_inliers:
        return 0.0

    X = disc[['x']].values
    y = disc['y'].values
    remaining = np.arange(len(disc))
    inliers = np.zeros(len(disc), bool)

    for _ in range(max_lines):
        if len(remaining) < min_inliers:
            break

        mdl = LinearRegression().fit(X[remaining], y[remaining])
        slope = mdl.coef_[0]

        eps_eff = eps * math.sqrt(1 + slope**2)
        ran = RANSACRegressor(LinearRegression(),
                              min_samples=min_inliers,
                              residual_threshold=eps_eff,
                              random_state=42)
        ran.fit(X[remaining], y[remaining])
        mask = ran.inlier_mask_
        if mask.sum() < min_inliers:
            break
        inliers[remaining[mask]] = True
        remaining = remaining[~mask]

    return inliers.sum() / len(disc)

# ---------- Concordant-fraction sweep for Case 3A with fixed discordants ----------
import zlib
from pathlib import Path

def _stable_seed(tag: str, base: int = SEED) -> int:
    return (base ^ zlib.adler32(tag.encode("utf-8"))) & 0xFFFFFFFF

def build_fixed_discordant_backbone(case_id: int, tier_name: str, n_disc: int, seed: int = SEED) -> pd.DataFrame:
    """Generate the discordant array once; reuse across all C-levels."""
    global rng
    rng = default_rng(seed)
    cdef = CASES[case_id]
    tier = TIERS[tier_name]
    disc = build_discordant_grains(cdef, tier, n_disc)
    # mark explicitly (True ↔ known discordants)
    disc['is_concordant'] = False
    return disc.reset_index(drop=True)

def make_dataset_from_backbone(backbone_disc: pd.DataFrame,
                               case_id: int, tier_name: str, C_pct: int,
                               seed: int = SEED, floor_conc_at_t_up: bool = True) -> pd.DataFrame:
    """Reuse identical discordants; add only the number of concordants needed to hit C%."""
    n_disc = len(backbone_disc)
    if n_disc == 0:
        raise ValueError("Backbone has zero discordant grains.")
    if C_pct >= 100:
        raise ValueError("C must be < 100.")
    n_conc = int(round(n_disc * C_pct / (100.0 - C_pct)))

    # draw concordants with a stable per-case/tier/C seed (discordants unchanged)
    global rng
    rng = default_rng(_stable_seed(f"Case{case_id}{tier_name}_C{C_pct:02d}", base=seed))

    cdef = CASES[case_id]
    tier = TIERS[tier_name]
    conc = build_concordant_grains(cdef, tier, n_conc, floor_at_t_up=floor_conc_at_t_up)
    conc['is_concordant'] = True

    df = pd.concat([conc, backbone_disc.copy()], ignore_index=True)
    df['Case'] = str(case_id)
    df['Tier'] = tier_name
    df['Sample'] = f"Case{case_id}{tier_name}_C{C_pct:02d}"
    return df

def sweep_case3A(out_root="case3A_sweep",
                 C_list=(5,10,20,30,40,50,60),
                 n_points_total=160, base_C_for_backbone=25,
                 seed: int = SEED,
                 floor_conc_at_t_up: bool = True):
    """
    Build discordants once (Case 3, Tier A). Then for each C in C_list,
    add concordants to achieve that fraction. Writes a *combined TW CSV*.
    """
    out_root = Path(out_root)
    (out_root/"cdc_in").mkdir(parents=True, exist_ok=True)
    (out_root/"weth").mkdir(parents=True, exist_ok=True)
    (out_root/"dd_in").mkdir(parents=True, exist_ok=True)
    (out_root/"plots").mkdir(parents=True, exist_ok=True)

    # Decide how many discordants to fix in the backbone
    n_disc = n_points_total - int(round(n_points_total * base_C_for_backbone/100.0))
    if n_disc <= 0:
        raise ValueError("Backbone needs at least 1 discordant.")

    # 1) Build backbone once
    backbone = build_fixed_discordant_backbone(case_id=3, tier_name="A", n_disc=n_disc, seed=seed)

    # 2) Generate datasets for each C (discordants identical)
    tw_all, weth_all, reim_all = [], [], []
    for C in C_list:
        df = make_dataset_from_backbone(backbone, case_id=3, tier_name="A", C_pct=C,
                                        seed=seed, floor_conc_at_t_up=floor_conc_at_t_up)

        # Optional per-sample files
        sample = df['Sample'].iat[0]
        to_teraW(df).to_csv(out_root/"cdc_in"/f"{sample}_TW.csv", index=False)
        to_reimink(df).to_csv(out_root/"dd_in"/f"{sample}_Reim.csv", index=False)
        df.to_csv(out_root/"weth"/f"{sample}_Weth.csv", index=False)

        # Append to combined stacks
        tw_all.append(to_teraW(df).assign(Sample=sample))
        weth_all.append(df.assign(Sample=sample))
        reim_all.append(to_reimink(df).assign(Sample=sample))

        # Quick panel plot if you like (comment out to skip)
        try:
            fig, ax = plt.subplots(figsize=(7.5,4.6))
            tvals = np.linspace(0, 4500, 400)
            cx, cy = [wetherill_xy(t)[0] for t in tvals], [wetherill_xy(t)[1] for t in tvals]
            ax.plot(cx, cy, color='0.7', lw=1.5)
            mask = df['is_concordant'].astype(bool)
            ax.errorbar(df.loc[mask,'x'], df.loc[mask,'y'],
                        xerr=2*df.loc[mask,'x_err'], yerr=2*df.loc[mask,'y_err'],
                        fmt='+', ms=5, mec='tab:green', ecolor='tab:green', alpha=0.9, label='Concordant')
            ax.errorbar(df.loc[~mask,'x'], df.loc[~mask,'y'],
                        xerr=2*df.loc[~mask,'x_err'], yerr=2*df.loc[~mask,'y_err'],
                        fmt='+', ms=5, mec='tab:orange', ecolor='tab:orange', alpha=0.9, label='Discordant')
            ax.set_xlim(0, 25); ax.set_ylim(0, 1.0)
            ax.set_title(sample); ax.legend(loc='lower right', frameon=False)
            fig.tight_layout(); fig.savefig(out_root/"plots"/f"{sample}.png", dpi=200); plt.close(fig)
        except Exception:
            pass

    # 3) Write combined CSVs (TW requested; Weth/Reimink optional)
    pd.concat(tw_all, ignore_index=True).to_csv(out_root/"tw_all_samples_combined.csv", index=False)
    pd.concat(weth_all, ignore_index=True).to_csv(out_root/"weth_all_samples_combined.csv", index=False)
    pd.concat(reim_all, ignore_index=True).to_csv(out_root/"reim_all_samples_combined.csv", index=False)
    print(f"✓ wrote combined TW: {out_root/'tw_all_samples_combined.csv'}")

from pathlib import Path

from pathlib import Path

def sweep_all_cases(out_root="all_cases_Csweep",
                    cases=(1, 2, 3, 4),
                    tiers=("A", "B", "C"),
                    C_list=(5, 10, 20, 30, 40, 50, 60),
                    n_points_total=160,
                    base_C_for_backbone=25,
                    seed: int = SEED,
                    floor_conc_at_t_up: bool = True):
    """
    Run the concordant-fraction sweep for multiple cases and tiers.

    For each case (1–4), writes:
      - Per-sample CSVs (TW, Weth, Reimink) in all_cases_Csweep/CaseX/...
      - ONE combined TW/Weth/Reim file per case:
            CaseX_TW_combined.csv, CaseX_Weth_combined.csv, CaseX_Reim_combined.csv

    AND ALSO:
      - ONE global combined file with *all* cases/tiers/C%s:
            all_cases_TW_combined.csv
            all_cases_Weth_combined.csv
            all_cases_Reim_combined.csv
    """
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # How many discordants to fix in the backbone for each Case×Tier
    n_disc_base = n_points_total - int(round(n_points_total * base_C_for_backbone/100.0))
    if n_disc_base <= 0:
        raise ValueError("Backbone needs at least 1 discordant.")

    # Global accumulators across *all* cases
    tw_all_cases   = []
    weth_all_cases = []
    reim_all_cases = []

    for case_id in cases:
        # --- per-case directory structure
        case_root = out_root / f"Case{case_id}"
        (case_root / "cdc_in").mkdir(parents=True, exist_ok=True)
        (case_root / "weth").mkdir(parents=True, exist_ok=True)
        (case_root / "dd_in").mkdir(parents=True, exist_ok=True)
        (case_root / "plots").mkdir(parents=True, exist_ok=True)

        tw_blocks   = []   # for per-case combined TW
        weth_blocks = []   # for per-case combined Weth
        reim_blocks = []   # for per-case combined Reim

        for tier_name in tiers:
            # 1) Build discordant backbone for THIS Case×Tier
            backbone_seed = _stable_seed(f"Case{case_id}{tier_name}_backbone", base=seed)
            backbone = build_fixed_discordant_backbone(case_id=case_id,
                                                       tier_name=tier_name,
                                                       n_disc=n_disc_base,
                                                       seed=backbone_seed)

            # 2) For each concordant fraction C, add concordants and write per-sample files
            for C in C_list:
                df = make_dataset_from_backbone(backbone_disc=backbone,
                                                case_id=case_id,
                                                tier_name=tier_name,
                                                C_pct=C,
                                                seed=seed,
                                                floor_conc_at_t_up=floor_conc_at_t_up)
                df['C_pct'] = C
                sample = df['Sample'].iat[0]

                # Prepare the various export styles
                tw_df   = to_teraW(df)
                reim_df = to_reimink(df)

                # Per-sample files under the case directory
                tw_df.to_csv(case_root / "cdc_in" / f"{sample}_TW.csv", index=False)
                reim_df.to_csv(case_root / "dd_in" / f"{sample}_Reim.csv", index=False)
                df.to_csv(case_root / "weth" / f"{sample}_Weth.csv", index=False)

                # Accumulate for per-case combined files
                tw_case = tw_df.assign(
                    Sample=sample,
                    Case=case_id,
                    Tier=tier_name,
                    C_pct=C
                )
                weth_case = df.copy().assign(
                    Case=case_id,
                    Tier=tier_name,
                    C_pct=C
                )
                reim_case = reim_df.assign(
                    Sample=sample,
                    Case=case_id,
                    Tier=tier_name,
                    C_pct=C
                )

                tw_blocks.append(tw_case)
                weth_blocks.append(weth_case)
                reim_blocks.append(reim_case)

                # Also accumulate to the *global* all-cases stacks
                tw_all_cases.append(tw_case)
                weth_all_cases.append(weth_case)
                reim_all_cases.append(reim_case)

                # Optional Wetherill plot per sample (same as before)
                try:
                    fig, ax = plt.subplots(figsize=(7.5, 4.6))
                    tvals = np.linspace(0, 4500, 400)
                    cx = [wetherill_xy(t)[0] for t in tvals]
                    cy = [wetherill_xy(t)[1] for t in tvals]
                    ax.plot(cx, cy, color='0.7', lw=1.5)

                    mask = df['is_concordant'].astype(bool)
                    ax.errorbar(df.loc[mask, 'x'], df.loc[mask, 'y'],
                                xerr=2*df.loc[mask, 'x_err'], yerr=2*df.loc[mask, 'y_err'],
                                fmt='+', ms=5, mec='tab:green', ecolor='tab:green',
                                alpha=0.9, label='Concordant')
                    ax.errorbar(df.loc[~mask, 'x'], df.loc[~mask, 'y'],
                                xerr=2*df.loc[~mask, 'x_err'], yerr=2*df.loc[~mask, 'y_err'],
                                fmt='+', ms=5, mec='tab:orange', ecolor='tab:orange',
                                alpha=0.9, label='Discordant')
                    ax.set_xlim(0, 25)
                    ax.set_ylim(0, 1.0)
                    ax.set_title(sample)
                    ax.legend(loc='lower right', frameon=False)
                    fig.tight_layout()
                    fig.savefig(case_root / "plots" / f"{sample}.png", dpi=200)
                    plt.close(fig)
                except Exception:
                    pass

        # --- One combined file per case (unchanged behaviour)
        if tw_blocks:
            pd.concat(tw_blocks, ignore_index=True).to_csv(
                case_root / f"Case{case_id}_TW_combined.csv", index=False
            )

        if weth_blocks:
            pd.concat(weth_blocks, ignore_index=True).to_csv(
                case_root / f"Case{case_id}_Weth_combined.csv", index=False
            )

        if reim_blocks:
            pd.concat(reim_blocks, ignore_index=True).to_csv(
                case_root / f"Case{case_id}_Reim_combined.csv", index=False
            )

        print(f"✓ wrote combined TW for Case {case_id}: {case_root / f'Case{case_id}_TW_combined.csv'}")

    # --- Global all-cases files ---
    if tw_all_cases:
        pd.concat(tw_all_cases, ignore_index=True).to_csv(
            out_root / "all_cases_TW_combined.csv", index=False
        )
    if weth_all_cases:
        pd.concat(weth_all_cases, ignore_index=True).to_csv(
            out_root / "all_cases_Weth_combined.csv", index=False
        )
    if reim_all_cases:
        pd.concat(reim_all_cases, ignore_index=True).to_csv(
            out_root / "all_cases_Reim_combined.csv", index=False
        )

# ============== 8. MAIN ========================================================

if __name__ == "__main__":
    sweep_all_cases(
        out_root="all_cases_Csweep",
        cases=[1, 2, 3, 4],
        tiers=["A", "B", "C"],
        C_list=[5, 10, 20, 30, 40, 50, 60],
        n_points_total=160,          # same total scale you used before
        base_C_for_backbone=25,      # backbone has ~75% discordants
        seed=42,
        floor_conc_at_t_up=True      # keep concordants at/older than each chord’s t_up
    )
