#!/usr/bin/env python3
"""
Synthetic U–Pb generator for Cases 5–7 that draws DISCORDANT arrays natively
in Tera–Wasserburg (TW) space (u = 238U/206Pb, p = 207Pb/206Pb). For two-stage
histories, we build the partial-loss intermediate in Wetherill and then sample
the *final segment* in TW so the arrays splay but remain TW-native.

Exports: Wetherill-style, Reimink-style, and TW CSV (same column names you use).
"""

import argparse
from pathlib import Path

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.random import default_rng
from sklearn.linear_model import RANSACRegressor, LinearRegression

# ====================== 1. GLOBAL CONSTANTS & HELPERS =========================
import matplotlib as mpl
from matplotlib.patches import Ellipse
import matplotlib.colors as mcolors

# manuscript house-style
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

# Concordant / discordant colours
COL_CONC = "mediumseagreen"       # bright green for concordant
COL_DISC = "thistle"           # pale lilac for discordant
ALPHA_CONC_FACE = 0.80     # slightly transparent fill
ALPHA_DISC_FACE = 0.80     # lighter fill for discordant
EDGE_LW = 0.15             # thin black outline
CENTER_DOT_MS = 1        # centre dot size

# 95% (chi-square 2 dof) scale
# 95% chi-square (2 dof)
CHI2_95 = 5.991

ERR_REL_RANGE   = (0.005, 0.015)  # 0.5–1.5 %  (1σ on each ratio)
ERR_X_MULT      = 1.0
ERR_Y_MULT      = 1.5              # widen σy so ellipses aren’t needles
RHO_CONST       = 0.85

ERR_X_MULT_CONC = 1.0              # concordant grains – realistic
ERR_Y_MULT_CONC = 1.0

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
    return math.log1p(x)/L235 if x > 0 else 0.0

def age_206_238(y: float) -> float:
    return math.log1p(y)/L238 if y > 0 else 0.0

def fractional_discordance(x: float, y: float) -> float:
    tA, tB = age_207_235(x), age_206_238(y)
    if tA <= 0 or tB <= 0:
        return 0.0
    return 1.0 - min(tA, tB)/max(tA, tB)

def is_reversed(x: float, y: float) -> bool:
    return (age_206_238(y) > age_207_235(x))

def conc_y_from_x(x: float) -> float:
    t = age_207_235(x)
    return math.expm1(L238 * t) if t > 0 else 0.0

# ====================== 2. TIERS & CASE DEFINITIONS ===========================

@dataclass
class Tier:
    name: str
    sig_perp: float         # geometric σ⊥ in TW
    sig_along: float        # geometric σ∥ in TW
    noise_frac: Tuple[float, float]  # not used here; kept for parity

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
    chord_weights: Optional[List[float]] = None  # for concordant distribution

TIERS: Dict[str, Tier] = {
    'A': Tier('A', sig_perp=0.000, sig_along=0.000, noise_frac=(0.001, 0.001)),
    'B': Tier('B', sig_perp=0.005, sig_along=0.005, noise_frac=(0.005, 0.010)),
    'C': Tier('C', sig_perp=0.010, sig_along=0.010, noise_frac=(0.005, 0.020)),
}

# These chords are used for concordant-draw anchoring/guide lines.
CASES: Dict[int, CaseDef] = {
    5: CaseDef(
        name='5',
        chords=[Chord(t_up=3000, t_low=1500)]    # one UI, older LI referenced
    ),
    6: CaseDef(
        name='6',
        chords=[
            Chord(t_up=3000, t_low=1500),
            Chord(t_up=3000, t_low=500)
        ],
        chord_weights=[0.5, 0.5]
    ),
    7: CaseDef(
        name='7',
        chords=[
            Chord(t_up=3200, t_low=1500),
            Chord(t_up=3000, t_low=1500)
        ],
        chord_weights=[0.5, 0.5]
    ),
}

# Subpopulation recipes for two‑stage geometry (weights sum to 1)
SUBPOPS = {
    5: dict(
        tups   = [3000, 3000],
        events = [[1500], [1500, 500]],     # single older, then two‑stage (older→younger)
        weights= [0.60, 0.40],
    ),
    6: dict(
        tups   = [3000, 3000, 3000],
        events = [[1500], [500], [1500, 500]],
        weights= [0.40, 0.30, 0.30],
    ),
    7: dict(
        tups   = [3200, 3000, 3200],
        events = [[1500], [1500], [1500, 500]],
        weights= [0.40, 0.30, 0.30],
    ),
}

# Partial‑loss fractions for staged histories (uniform ranges)
FRACTION_RANGES = {
    1: (0.70, 0.95),   # fraction toward the first event
    2: (0.05, 0.95),   # fraction of the *second* step (from the intermediate)
}

# ====================== 3. GENERATION PARAMS & FILTERS ========================

N_POINTS        = 160
CONC_FRAC       = 0.25
SPREAD_SINGLE   = 5.0
SPREAD_MULTI    = 15.0

MIN_DISCORDANCE = 0.01
CLEARANCE_FRAC  = 0.03

R_TW   = 137.818                # 238U/235U
TW_CAP = 0.26                   # cap in p = 207Pb/206Pb

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
    if fractional_discordance(xw, yw) < MIN_DISCORDANCE:
        return False

    # Clearance below concordia
    yw_conc = conc_y_from_x(xw)
    if y >= yw_conc * (1.0 - CLEARANCE_FRAC):
        return False

    # TW Pb/Pb cap
    if ((x / y) / R_TW) > TW_CAP:
        return False

    return True

# ====================== 4. TW HELPERS =========================================

def weth_to_tw(x: float, y: float) -> Tuple[float, float]:
    """(x,y) -> (u,p) with u = 238U/206Pb, p = 207Pb/206Pb."""
    if y <= 0:
        return (np.nan, np.nan)
    u = 1.0 / y
    p = (x / y) / R_TW
    return (u, p)

def tw_to_weth(u: float, p: float) -> Tuple[float, float]:
    """(u,p) -> (x,y) with p = (x/y)/R_TW and u = 1/y."""
    if u <= 0:
        return (np.nan, np.nan)
    y = 1.0 / u
    x = (p * R_TW) * y
    return (x, y)

def tw_line_from_points(x0: float, y0: float, x1: float, y1: float) -> Tuple[float, float, float, float]:
    """
    Return (A, B, u_min, u_max) for the TW line p = A*u + B through two Wetherill points.
    """
    u0, p0 = weth_to_tw(x0, y0)
    u1, p1 = weth_to_tw(x1, y1)
    if not (np.isfinite(u0) and np.isfinite(p0) and np.isfinite(u1) and np.isfinite(p1)):
        return (np.nan, np.nan, np.nan, np.nan)
    if abs(u1 - u0) < 1e-15:
        return (np.nan, np.nan, np.nan, np.nan)
    A = (p1 - p0) / (u1 - u0)
    B = p0 - A * u0
    u_min, u_max = (min(u0, u1), max(u0, u1))
    return (A, B, u_min, u_max)

def sample_on_tw_segment(x0: float, y0: float, x1: float, y1: float,
                         tier: Tier,
                         sig_perp_override: Optional[float] = None,
                         sig_along_override: Optional[float] = None) -> Tuple[float, float, float, float]:
    """
    TW-native sampling: pick (u,p) along the TW line through (x0,y0)-(x1,y1),
    add along/perp Gaussian scatter in TW, convert back to Wetherill, add meas. errors,
    and return a single discordant point (x,y,xerr,yerr). May return NaNs on failure.
    """
    A, B, u_min, u_max = tw_line_from_points(x0, y0, x1, y1)
    if not (np.isfinite(A) and np.isfinite(B) and np.isfinite(u_min) and np.isfinite(u_max) and (u_max > u_min)):
        return (np.nan, np.nan, np.nan, np.nan)

    # unit vectors along/perp in TW
    norm = math.hypot(1.0, A)
    e_par  = (1.0 / norm, A / norm)
    e_perp = (-A / norm, 1.0 / norm)

    sig_perp  = sig_perp_override  if sig_perp_override  is not None else tier.sig_perp
    sig_along = sig_along_override if sig_along_override is not None else tier.sig_along

    for _ in range(500):
        u  = rng.uniform(u_min, u_max)
        p  = A * u + B

        u_geo = u + rng.normal(0.0, sig_along) * e_par[0] + rng.normal(0.0, sig_perp) * e_perp[0]
        p_geo = p + rng.normal(0.0, sig_along) * e_par[1] + rng.normal(0.0, sig_perp) * e_perp[1]
        if not (u_geo > 0):
            continue

        x_geo, y_geo = tw_to_weth(u_geo, p_geo)
        if not (x_geo > 0 and y_geo > 0):
            continue

        pct   = rng.uniform(*ERR_REL_RANGE)
        x_err = abs(x_geo) * pct * ERR_X_MULT
        y_err = abs(y_geo) * pct * ERR_Y_MULT

        if passes_filter(x_geo, y_geo, x_err, y_err):
            return (x_geo, y_geo, x_err, y_err)

    return (np.nan, np.nan, np.nan, np.nan)

# ====================== 5. MULTI‑STAGE HELPERS (W->TW) ========================

def partial_step(x0: float, y0: float, t_evt: float, frac: float) -> Tuple[float, float]:
    """
    Wetherill linear interpolation toward the event concordia point by 'frac'.
    (This is the usual approximation for staged Pb-loss trajectories.)
    """
    xe, ye = wetherill_xy(t_evt)
    return (x0 + frac * (xe - x0), y0 + frac * (ye - y0))

def draw_one_discordant_case5to7(t_up: float, events: List[float], tier: Tier) -> Tuple[float, float, float, float]:
    """
    For a given subpopulation defined by starting upper-intercept t_up and a sequence
    of one or two events, return one discordant point (x,y,xerr,yerr) generated
    TW-natively on the *final* segment.
    """
    # starting point on concordia (UI)
    x0, y0 = wetherill_xy(t_up)

    if len(events) == 1:
        # One-stage: sample on TW line (UI -> event)
        x1, y1 = wetherill_xy(events[0])
        return sample_on_tw_segment(x0, y0, x1, y1, tier)

    elif len(events) == 2:
        # Two-stage: first move part-way to older event, then toward younger;
        # finally sample along the *last* segment in TW (gives the splay).
        f1_lo, f1_hi = FRACTION_RANGES[1]
        f2_lo, f2_hi = FRACTION_RANGES[2]
        for _ in range(200):
            f1 = rng.uniform(f1_lo, f1_hi)  # toward first (older) event
            x1, y1 = partial_step(x0, y0, events[0], f1)

            f2 = rng.uniform(f2_lo, f2_hi)  # then toward the younger
            x2, y2 = partial_step(x1, y1, events[1], f2)

            xg, yg, xe, ye = sample_on_tw_segment(x1, y1, x2, y2, tier)
            if np.isfinite(xg) and np.isfinite(yg):
                return (xg, yg, xe, ye)

        return (np.nan, np.nan, np.nan, np.nan)

    else:
        # Not used here, but safe fallback
        return (np.nan, np.nan, np.nan, np.nan)

# ====================== 6. BUILDERS ===========================================

def build_concordant_grains(case_def: CaseDef, tier: Tier, n_conc: int) -> pd.DataFrame:
    rows = []
    chords = case_def.chords
    if len(chords) == 1:
        t_up_list = [chords[0].t_up]
        spread = SPREAD_SINGLE
    else:
        t_up_list = [ch.t_up for ch in chords]
        spread = SPREAD_MULTI

    # distribute among chords
    if case_def.chord_weights and len(case_def.chord_weights) == len(t_up_list):
        w = np.array(case_def.chord_weights, float); w /= w.sum()
        ccounts = np.floor(w * n_conc).astype(int)
        for i in range(n_conc - ccounts.sum()):
            ccounts[i % len(ccounts)] += 1
    else:
        base, rem = divmod(n_conc, len(t_up_list))
        ccounts = [base] * len(t_up_list)
        for i in range(rem):
            ccounts[i] += 1

    for i, tup_val in enumerate(t_up_list):
        for _ in range(ccounts[i]):
            tc = rng.normal(tup_val, spread)
            tc = max(tc, 0.0)
            xC, yC = wetherill_xy(tc)

            pct   = rng.uniform(*ERR_REL_RANGE)
            x_err = abs(xC) * pct * ERR_X_MULT_CONC
            y_err = abs(yC) * pct * ERR_Y_MULT_CONC

            rows.append([xC, yC, x_err, y_err, tup_val, np.nan, True])

    cols = ['x','y','x_err','y_err','t_up_true','t_low_true','is_concordant']
    return pd.DataFrame(rows, columns=cols)

def build_discordant_grains_5to7(case_id: int, case_def: CaseDef, tier: Tier, n_disc: int) -> pd.DataFrame:
    """
    TW-native discordant generator for Cases 5–7 using the SUBPOPS recipes above.
    """
    spec = SUBPOPS[case_id]
    tups, events_list, weights = spec['tups'], spec['events'], spec['weights']

    w = np.array(weights, float); w /= w.sum()
    counts = np.floor(w * n_disc).astype(int)
    for i in range(n_disc - counts.sum()):
        counts[i % len(counts)] += 1

    rows: List[List[float]] = []
    for i_sub, n_sub in enumerate(counts):
        t_up   = float(tups[i_sub])
        events = list(events_list[i_sub])

        made = 0; safety = 0
        while made < n_sub and safety < n_sub * 200:
            xg, yg, xe, ye = draw_one_discordant_case5to7(t_up, events, tier)
            safety += 1
            if not (np.isfinite(xg) and np.isfinite(yg)):
                continue
            # record t_low_true as the last event age for convenience
            rows.append([xg, yg, xe, ye, t_up, float(events[-1]), False])
            made += 1

    cols = ['x','y','x_err','y_err','t_up_true','t_low_true','is_concordant']
    return pd.DataFrame(rows, columns=cols)

def simulate_case(case_id: int, tier: Tier) -> pd.DataFrame:
    """
    Creates one synthetic dataset (panel) for a given (case, tier).
    """
    case_def = CASES[case_id]
    n_conc = int(N_POINTS * CONC_FRAC)
    n_disc = N_POINTS - n_conc

    df_conc = build_concordant_grains(case_def, tier, n_conc)
    df_disc = build_discordant_grains_5to7(case_id, case_def, tier, n_disc)

    df = pd.concat([df_conc, df_disc], ignore_index=True)
    df['Case'] = case_def.name
    df['Tier'] = tier.name
    return df

def simulate_all_cases() -> List[pd.DataFrame]:
    panels = []
    for i_case in [5, 6, 7]:
        for tiername in ['A','B','C']:
            tier = TIERS[tiername]
            panels.append(simulate_case(i_case, tier))
    return panels

# ====================== 7. PLOTTING (Wetherill) ===============================
def ellipse_patch(x, y, sx, sy, *, is_conc: bool, rho: float = RHO_CONST) -> Ellipse:
    """Return a 95% covariance ellipse patch for (x,y) with 1σ errors sx, sy and correlation rho."""
    cov = np.array([[sx*sx,         rho*sx*sy],
                    [rho*sx*sy,     sy*sy]])
    lam, vec = np.linalg.eigh(cov)
    lam, vec = lam[::-1], vec[:, ::-1]

    # 95% ellipse for 2 dof
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

def plot_grid(panels: List[pd.DataFrame]):
    case_ids = [5, 6, 7]
    tiers    = ['A', 'B', 'C']
    nrows, ncols = len(case_ids), len(tiers)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(7.5, 7.5),
        sharex=True,
        sharey=True,
        gridspec_kw={"wspace": 0.03, "hspace": 0.03},
    )

    # Concordia curve
    tvals = np.linspace(0, 4500, 300)
    cx, cy = zip(*[wetherill_xy(tt) for tt in tvals])

    # Look-up by (case, tier)
    panel_lu = {(int(df['Case'].iat[0]), df['Tier'].iat[0]): df for df in panels}

    for r, cid in enumerate(case_ids):
        for c, tname in enumerate(tiers):
            ax = axes[r, c]
            df_panel = panel_lu[(cid, tname)]

            # Concordia
            ax.plot(cx, cy, color='slategray', lw=0.8, zorder=0)

            # Guide chords
            cdef = CASES[cid]
            for ch in cdef.chords:
                xU, yU = wetherill_xy(ch.t_up)
                xL, yL = wetherill_xy(ch.t_low)
                ax.plot([xU, xL], [yU, yL], color='lightslategray', ls='--', lw=0.6, zorder=1)

            # Ellipses + small centre dot
            for x, y, sx, sy, is_c in df_panel[['x','y','x_err','y_err','is_concordant']].itertuples(index=False):
                if not (np.isfinite(x) and np.isfinite(y)):
                    continue
                patch = ellipse_patch(x, y, sx, sy, is_conc=bool(is_c), rho=RHO_CONST)
                ax.add_patch(patch)
                # small coloured centre dot
                col = COL_CONC if is_c else COL_DISC
                ax.plot(x, y, marker='o', ms=CENTER_DOT_MS,
                        mfc='k', mec='black', mew=0.3, zorder=5)

            # Axes limits
            ax.set_xlim(0, 25)
            ax.set_ylim(0, 0.8)

            # Only bottom row has x labels, only left column has y labels
            show_xlabels = (r == nrows - 1)
            show_ylabels = (c == 0)
            ax.tick_params(
                direction='in',
                labelsize=7,
                pad=2,
                labelbottom=show_xlabels,
                labelleft=show_ylabels,
            )

            # Column titles & row labels
            if r == 0:
                ax.set_title(f"Tier {tname}", fontsize=10, fontweight='bold')
            if c == ncols - 1:
                ax.text(
                    1.02, 0.5, f"Case {cid}",
                    transform=ax.transAxes,
                    rotation=-90,
                    va='center', ha='left',
                    fontsize=9, fontweight='bold',
                )
            # cl = df_panel["CL_code"].iat[0]
            # ax.text(
            #     0.96, 0.04, cl,
            #     transform=ax.transAxes,
            #     fontsize=6,
            #     ha="right", va="bottom",
            #     color="0.3",
            #     bbox=dict(
            #         facecolor="white",
            #         edgecolor="0.5",
            #         boxstyle="round,pad=0.15",
            #         alpha=0.7,
            #     ),
            # )

    # Use interior ticks so 0 and max don't clash between panels
    axes[-1, 0].set_xticks([5, 10, 15, 20])
    axes[-1, 0].set_xticklabels(["5", "10", "15", "20"])

    axes[-1, 0].set_yticks([0.1, 0.3, 0.5, 0.7])
    axes[-1, 0].set_yticklabels(["0.1", "0.3", "0.5", "0.7"])

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

    # global axis labels
    fig.text(
        0.5, 0.07,                           # y=0.03 -> a bit farther from the panels
        r"$^{207}\mathrm{Pb}/^{235}\mathrm{U}$",
        ha="center", va="center", fontsize=9,
    )
    fig.text(
        0.07, 0.5,                          # x=0.045 -> closer to the panels
        r"$^{206}\mathrm{Pb}/^{238}\mathrm{U}$",
        ha="center", va="center",
        rotation="vertical", fontsize=9,
    )

    fig.tight_layout(rect=[0.08, 0.08, 0.98, 0.96])
    return fig

# ====================== 8. EXPORTS & METRICS ==================================

def to_teraW(df: pd.DataFrame) -> pd.DataFrame:
    R = R_TW
    x   = df['x'].to_numpy(float)
    y   = df['y'].to_numpy(float)
    sx  = df['x_err'].to_numpy(float)
    sy  = df['y_err'].to_numpy(float)
    rho = np.full_like(x, RHO_CONST, dtype=float)
    if 'rho' in df:
        try: rho = df['rho'].to_numpy(dtype=float)
        except Exception: pass

    u_val = 1.0 / y
    u_err = sy / (y**2)

    # var of (x/y)/R with covariance
    var_xy = (sx**2)/(y**2) + (x**2 * sy**2)/(y**4) - (2.0 * rho * x * sx * sy)/(y**3)
    var_xy = np.maximum(var_xy, 0.0)  # numerical guard

    p_val = (x / y) / R
    p_err = np.sqrt(var_xy) / R

    return pd.DataFrame({
        'Sample':   df['Sample'],
        'uPbValue': u_val,   'uPbError': u_err,
        'pbPbValue': p_val,  'pbPbError': p_err
    })


def to_reimink(df: pd.DataFrame) -> pd.DataFrame:
    age76 = np.log1p(df['x']) / L235
    return pd.DataFrame({
        'Sample': df['Sample'],
        'Pb7_U35': df['x'],
        'Pb7_U35.2SE.abs': 2.0 * df['x_err'],
        'Pb6_U38': df['y'],
        'Pb6_U38.2SE.abs': 2.0 * df['y_err'],
        'rho': RHO_CONST,
        'age76': age76
    })

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

# ====================== 9. MAIN ===============================================

def _default_paper_dir() -> Path:
    p = Path(__file__).resolve()
    cand = p.parents[2]
    if (cand / "data").is_dir() and (cand / "scripts").is_dir():
        return cand
    for parent in p.parents:
        if (parent / "data").is_dir() and (parent / "scripts").is_dir():
            return parent
    return cand


def _parse_args() -> argparse.Namespace:
    paper_dir = _default_paper_dir()
    ap = argparse.ArgumentParser(description="Figure 2: Synthetic Cases 5–7 (Wetherill grid) and optional CSV exports.")

    ap.add_argument("--paper-dir", type=Path, default=paper_dir, help="Paper/repo root directory.")
    ap.add_argument("--fig-dir", type=Path, default=(paper_dir / "Figures"), help="Output directory for figure files.")
    ap.add_argument(
        "--data-out-dir",
        type=Path,
        default=(paper_dir / "outputs" / "synthetic"),
        help="Output directory for generated CSVs (default: <paper>/outputs/synthetic).",
    )
    ap.add_argument(
        "--inputs-dir",
        type=Path,
        default=(paper_dir / "data" / "inputs" / "Cases 1-7 Pb loss Inputs"),
        help="Repository input-data directory (only used with --write-inputs).",
    )

    ap.add_argument("--write-inputs", action="store_true", help="Write canonical input CSVs under --inputs-dir.")
    ap.add_argument("--overwrite", action="store_true", help="Allow overwriting existing CSVs.")
    ap.add_argument("--save-fig", action="store_true", help="Write figure files to --fig-dir.")
    ap.add_argument("--formats", type=str, default="svg,png,pdf", help="Comma-separated output formats for the figure.")
    ap.add_argument("--show", action="store_true", help="Display the figure interactively.")

    return ap.parse_args()


def _safe_write_csv(df: pd.DataFrame, path: Path, *, overwrite: bool) -> None:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file (use --overwrite): {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    args = _parse_args()

    paper_dir = args.paper_dir.expanduser().resolve()
    fig_dir = args.fig_dir.expanduser().resolve()
    data_out_dir = args.data_out_dir.expanduser().resolve()
    inputs_dir = args.inputs_dir.expanduser().resolve()
    formats = [s.strip().lower() for s in str(args.formats).split(",") if s.strip()]

    panels = []
    for i_case in [5, 6, 7]:
        for tiername in ["A", "B", "C"]:
            tier = TIERS[tiername]
            dfp = simulate_case(i_case, tier)

            # add L and CL_code for each panel
            Lval = linearity_L(dfp)
            dfp["L"] = Lval

            n_total = len(dfp)
            n_conc = dfp["is_concordant"].sum()
            C_pct = int(round(100 * n_conc / n_total))
            L_pct = int(round(100 * Lval))
            dfp["CL_code"] = f"C{C_pct:02d}L{L_pct:02d}"

            panels.append(dfp)

    master = pd.concat(panels, ignore_index=True)
    master["Sample"] = master["Case"].astype(str) + master["Tier"]

    # Figure
    fig = plot_grid(panels)
    if args.save_fig:
        fig_dir.mkdir(parents=True, exist_ok=True)
        out_stub = "fig02_synthetic_cases5to7_wetherill_grid"
        out_paths = []
        for ext in formats:
            out_path = fig_dir / f"{out_stub}.{ext}"
            fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
            out_paths.append(out_path)
        print("[Fig02] wrote:")
        for p in out_paths:
            print("  ", p)

    if args.show:
        plt.show()
    plt.close(fig)

    # CSV exports
    if args.write_inputs:
        out_root = inputs_dir
    else:
        out_root = data_out_dir
    out_root.mkdir(parents=True, exist_ok=True)

    path_tw = out_root / "synthetic_teraW_5to7_all.csv"
    path_reim = out_root / "synthetic_reimink_5to7_all.csv"

    _safe_write_csv(to_teraW(master), path_tw, overwrite=args.overwrite)
    _safe_write_csv(to_reimink(master), path_reim, overwrite=args.overwrite)

    print(f"[Fig02] wrote {len(master)} rows to:")
    print("  ", path_tw)
    print("  ", path_reim)


if __name__ == "__main__":
    main()
