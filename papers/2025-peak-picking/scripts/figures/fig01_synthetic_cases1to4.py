#!/usr/bin/env python3
"""
Synthetic U–Pb generator that draws DISCORDANT arrays natively in
Tera–Wasserburg (TW) space, then converts to Wetherill for filtering
and for the second app.
- Exports: Wetherill-style raw, Reimink-style, and TW CSV.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.random import default_rng
from sklearn.linear_model import RANSACRegressor, LinearRegression
from matplotlib.patches import Ellipse
from matplotlib import colors as mcolors

# ============== 1. GLOBAL CONSTANTS & HELPERS =================================
ERR_REL_RANGE = (0.005, 0.015)     # 0.5–1.5 %   (1σ on each ratio)
ERR_X_MULT    = 1.0
ERR_Y_MULT    = 1.5                # widen σy so ellipses are not needles
RHO_CONST     = 0.85               # used in Reimink export (if needed)

ERR_X_MULT_CONC = 1.0              # concordant grains – keep realistic
ERR_Y_MULT_CONC = 1.0

# ------------------------------------------------------------------
PAPER_DIR = Path(__file__).resolve().parents[2]   # .../papers/2025-peak-picking
FIG_DIR   = PAPER_DIR / "outputs" / "figures"
DATA_DIR  = PAPER_DIR / "outputs" / "derived" / "synthetic"

FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
rng  = default_rng(SEED)

# Decay constants (1/Ma)
L235 = 9.8485e-4
L238 = 1.55125e-4

from matplotlib.patches import Ellipse  # add this

import matplotlib as mpl
mpl.rcParams.update({
    "font.family":      "serif",
    "font.size":        10,
    "mathtext.fontset": "stix",
    "axes.labelsize":   10,
    "axes.titlesize":   11,
    "axes.linewidth":   0.8,
    "xtick.direction":  "in",
    "ytick.direction":  "in",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "legend.fontsize":  8,
    "lines.linewidth":  1.2,
    "savefig.dpi":      300,
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
})

# Colours & opacities (IsoplotR-ish style)
COL_CONC = "mediumseagreen"       # bright green for concordant
COL_DISC = "thistle"           # pale lilac for discordant
ALPHA_CONC_FACE = 0.80     # slightly transparent fill
ALPHA_DISC_FACE = 0.80    # lighter fill for discordant
EDGE_LW = 0.15             # thin black outline
CENTER_DOT_MS = 1       # centre dot size

# keep this
CHI2_95 = 5.991  # 95% ellipse scale for 2 dof

def ellipse_patch(x, y, sx, sy, *, is_conc: bool, rho: float = RHO_CONST) -> Ellipse:
    cov = np.array([[sx*sx, rho*sx*sy],
                    [rho*sx*sy, sy*sy]])
    lam, vec = np.linalg.eigh(cov)
    lam, vec = lam[::-1], vec[:, ::-1]

    # 95% confidence ellipse for 2 dof
    w, h = 2.0 * np.sqrt(lam * CHI2_95)

    angle = np.degrees(np.arctan2(vec[1, 0], vec[0, 0]))
    base = COL_CONC if is_conc else COL_DISC
    fa   = ALPHA_CONC_FACE if is_conc else ALPHA_DISC_FACE
    r, g, b = mcolors.to_rgb(base)

    return Ellipse(
        (x, y), w, h, angle=angle,
        facecolor=(r, g, b, fa),
        edgecolor="black",
        lw=EDGE_LW,
        zorder=4,
    )

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
            Chord(t_up=3200, t_low=1800, f_min=0.0, f_max=0.95),
            Chord(t_up=3200, t_low=300,   f_min=0.0, f_max=0.95),
        ],
        chord_weights=[0.5, 0.5]
    ),
    3: CaseDef(
        name='3', #3
        chords=[
            Chord(t_up=3200, t_low=400, f_min=0.0, f_max=0.95), #3200 400
            Chord(t_up=3000, t_low=400, f_min=0.0, f_max=0.95), #3000 400
            Chord(t_up=2800, t_low=400, f_min=0.0, f_max=0.95), #2800 400
        ]
    ),
    4: CaseDef(
        name='4',
        chords=[
            Chord(t_up=3200, t_low=500,  f_min=0.0, f_max=0.95),
            Chord(t_up=3000, t_low=1800, f_min=0.0, f_max=0.95), #3100
            Chord(t_up=2800, t_low=1800, f_min=0.0, f_max=0.95), #3000
        ],
        chord_weights=[0.5, 0.25, 0.25]
    ),
}

# ============== 3. GENERATION PARAMETERS & FILTERS ============================

N_POINTS = 160
CONC_FRAC = 0.25

SPREAD_SINGLE = 5.0
SPREAD_MULTI  = 15.0

MIN_DISCORDANCE = 0.05 #0.01
CLEARANCE_FRAC  = 0.03

R_TW = 137.818                 # 238U/235U
TW_CAP = 0.25                  # cap in p = 207Pb/206Pb

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

def build_concordant_grains(case_def: CaseDef, tier: Tier, n_conc: int) -> pd.DataFrame:
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

def plot_grid(panels: List[pd.DataFrame]) -> plt.Figure:
    """4×3 Wetherill grid ordered by case (rows 1–4) then tier (cols A–C)."""
    fig, axes = plt.subplots(
        4, 3,
        figsize=(7.5, 9),
        sharex=True,
        sharey=True,
        gridspec_kw={"wspace": 0.03, "hspace": 0.03},
    )

    # Concordia curve (same for all subplots)
    tvals = np.linspace(0, 4500, 400)
    cx, cy = zip(*(wetherill_xy(tt) for tt in tvals))

    for df_panel in panels:
        case = int(df_panel['Case'].iat[0])
        tier = df_panel['Tier'].iat[0]
        r = case - 1
        c = ['A','B','C'].index(tier)
        ax = axes[r, c]

        # concordia and guide chords
        ax.plot(cx, cy, color="slategray", lw=1.0, zorder=0)
        for ch in CASES[case].chords:
            xU, yU = wetherill_xy(ch.t_up)
            xL, yL = wetherill_xy(ch.t_low)
            ax.plot([xU, xL], [yU, yL], ls="--", lw=0.8, color="lightslategray", zorder=1)

        for x, y, sx, sy, is_c in df_panel[['x','y','x_err','y_err','is_concordant']].itertuples(index=False):
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            patch = ellipse_patch(x, y, sx, sy, is_conc=bool(is_c), rho=RHO_CONST)
            ax.add_patch(patch)
            # centre dot
            ax.plot(
                x, y,
                marker='o',
                ms=CENTER_DOT_MS,
                mfc='black',
                mec='none',
                zorder=5,
            )

        # cosmetics
        ax.set_xlim(0, 25)
        ax.set_ylim(0, 0.8)
        ax.tick_params(direction='in', labelsize=8)
        if r == 0:
            ax.set_title(f"Tier {tier}", fontweight='bold', fontsize=10)
        if c == 2:
            ax.text(1.02, 0.5, f"Case {case}", transform=ax.transAxes,
                    rotation=-90, ha='left', va='center', fontsize=9, fontweight='bold')

        # cosmetics
        ax.set_xlim(0, 25)
        ax.set_ylim(0, 0.8)

        # only bottom row has x numbers, only left column has y numbers
        show_xlabels = (r == 3)
        show_ylabels = (c == 0)
        ax.tick_params(
            direction='in',
            labelsize=7,
            labelbottom=show_xlabels,
            labelleft=show_ylabels,
        )

        cl = df_panel["CL_code"].iat[0]
        ax.text(
            0.96, 0.04, cl,
            transform=ax.transAxes,
            fontsize=6,
            ha="right", va="bottom",
            color="0.3",
            bbox=dict(
                facecolor="white",
                edgecolor="0.5",
                boxstyle="round,pad=0.15",
                alpha=0.7,
            ),
        )
    # legend inside first panel (top-left), two rows (vertical)
    handles = [
        plt.Line2D([], [], ls='', marker='s', markersize=6,
                   markerfacecolor=COL_CONC, markeredgecolor='black',
                   label="Concordant"),
        plt.Line2D([], [], ls='', marker='s', markersize=6,
                   markerfacecolor=COL_DISC, markeredgecolor='black',
                   label="Discordant"),
    ]

    axes[0, 0].legend(
        handles=handles,
        loc="upper left",    # inside the first subplot
        frameon=False,
        ncol=1,              # 1 column -> 2 rows (one above the other)
        borderaxespad=0.3,
        handlelength=1.0,
        handletextpad=0.4,
    )

    # global x/y labels
    fig.text(0.5, 0.07, r"$^{207}\mathrm{Pb}/^{235}\mathrm{U}$",
             ha="center", va="center", fontsize=9)
    fig.text(0.07, 0.5, r"$^{206}\mathrm{Pb}/^{238}\mathrm{U}$",
             ha="center", va="center", rotation="vertical", fontsize=9)

    # choose interior tick labels only (no 0 and no 25 / 0.8)
    # apply to the shared axes (any one panel is fine)
    axes[3, 0].set_xticks([0, 5, 10, 15, 20])
    axes[3, 0].set_xticklabels(["0", "5", "10", "15", "20"])

    axes[3, 0].set_yticks([0, 0.1, 0.3, 0.5, 0.7])
    axes[3, 0].set_yticklabels(["0", "0.1", "0.3", "0.5", "0.7"])

    fig.tight_layout(rect=[0.06, 0.11, 1.0, 0.94])
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
            ax.plot(uu, A*uu + B, color='lightslategray', ls='--', lw=1.0)

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

def to_teraW(df: pd.DataFrame, rho: float = RHO_CONST) -> pd.DataFrame:
    x  = df['x'].to_numpy()
    y  = df['y'].to_numpy()
    sx = df['x_err'].to_numpy()
    sy = df['y_err'].to_numpy()

    # central values
    u_val = 1.0 / y
    p_val = (x / y) / R_TW

    # uncertainties (propagated from Wetherill; app still gets scalar 1σ’s)
    u_err = sy / (y**2)
    var_x_over_y = (sx**2)/(y**2) + (x**2*sy**2)/(y**4) - (2.0*x/(y**3))*rho*sx*sy
    var_x_over_y = np.clip(var_x_over_y, 0.0, np.inf)  # numerical safety
    p_err = np.sqrt(var_x_over_y) / R_TW

    return pd.DataFrame({
        'Sample':    df['Sample'],
        'uPbValue':  u_val,
        'uPbError':  u_err,
        'pbPbValue': p_val,
        'pbPbError': p_err
    })


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

# ============== 8. MAIN ========================================================

if __name__ == "__main__":
    from pathlib import Path

    # 1) simulate all case×tier panels
    panels = []
    for i_case in [1, 2, 3, 4]:
        cdef = CASES[i_case]
        for tiername in ["A", "B", "C"]:
            tier = TIERS[tiername]
            dfp = simulate_case(cdef, tier)

            # metrics & labeling (same as your original)
            Lval = linearity_L(dfp)
            dfp["L"] = Lval
            dfp["True_I"] = ";".join(str(ch.t_low) for ch in cdef.chords)

            n_total = len(dfp)
            n_conc  = dfp["is_concordant"].sum()
            C_pct = int(round(100 * n_conc / n_total))
            L_pct = int(round(100 * Lval))
            dfp["CL_code"] = f"C{C_pct:02d}L{L_pct:02d}"

            panels.append(dfp)

    master = pd.concat(panels, ignore_index=True)
    master["Sample"] = master["Case"].astype(str) + master["Tier"]

    # 2) plots
    fig_w = plot_grid(panels)
    fig_w.savefig(FIG_DIR / "cases1-4_Wetherill_grid.svg")
    fig_w.savefig(FIG_DIR / "cases1-4_Wetherill_grid.tiff")
    fig_w.savefig(FIG_DIR / "cases1-4_Wetherill_grid.png")
    plt.show()

    # 3) write three separate files
    path_tw   = DATA_DIR / "cases1to4_synth_TW_test.csv"
    path_reim = DATA_DIR / "cases1to4_synth_reim_test.csv"
    path_weth = DATA_DIR / "cases1to4_synth_weth_test.csv"

    to_teraW(master).to_csv(path_tw,   index=False)
    to_reimink(master).to_csv(path_reim, index=False)
    master.to_csv(path_weth, index=False)

    print(f"Wrote {len(master)} rows to:")
    print("  ", path_tw)
    print("  ", path_reim)
    print("  ", path_weth)

