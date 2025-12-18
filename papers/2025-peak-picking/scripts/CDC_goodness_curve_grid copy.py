#!/usr/bin/env python3
"""
CDC goodness-surface grids with ensemble-catalogue overlay.

Each panel shows:
  • all runs' goodness curves S(t) (grey)
  • the median goodness curve (black)
  • peaks on the median (orange dots + age labels)  [illustrative]
  • true ages (dashed verticals)
  • ENSEMBLE CATALOGUE ages (green triangles) with 95% CI whiskers and support

Author : <you>
Updated: 2025-08-07
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from scipy.signal import find_peaks
from matplotlib.patches import Polygon
from pathlib import Path
import os
from pathlib import Path

# 1) Point to your “New outputs” by default, but allow env overrides.
KS_DIR = Path(os.environ.get(
    "KS_DIR",
    "/Users/lucymathieson/Desktop/Peak-Picking-Manuscript-Python"
)).expanduser()

CURVE_SURFACE    = os.environ.get("CDC_UI_SURFACE", "PEN").strip().lower()
TRIANGLE_SURFACE = os.environ.get("CDC_CATALOGUE_SURFACE", CURVE_SURFACE).strip().lower()

# Penalised → ensemble_catalogue.csv ; Raw → ensemble_catalogue_np.csv
BASE_CAT = Path(os.environ.get(
    "CAT_BASE",
    "/Users/lucymathieson/Desktop/Peak-Picking-Manuscript-Python/ensemble_catalogue_RESULTS_repick"
)).expanduser()

CATALOGUE_CSV = (BASE_CAT.with_suffix(".csv") if TRIANGLE_SURFACE == "pen"
                 else BASE_CAT.with_name(BASE_CAT.name + "_np").with_suffix(".csv"))

OUTDIR = Path("/Users/lucymathieson/Desktop/Peak-Picking-Manuscript-Python/grids repick")
OUTDIR.mkdir(parents=True, exist_ok=True)

print(f"[paths] KS_DIR={KS_DIR} (exists={KS_DIR.exists()})")
print(f"[paths] CATALOGUE_CSV={CATALOGUE_CSV} (exists={CATALOGUE_CSV.exists()})")

def _discover_tags(ks_dir: Path):
    # Collect tags like "4C" from filenames "4C_123.npz"
    tags = sorted({p.stem.split("_")[0].upper() for p in ks_dir.rglob("*.npz")})
    print(f"[paths] Found {len(tags)} tag(s) with NPZ data: {', '.join(tags) if tags else '(none)'}")
    return tags

FOUND_TAGS = _discover_tags(KS_DIR)


# ── truth & layout ───────────────────────────────────────────────────────────
TRUE = {
    "1": [700],
    "2": [300, 1800],
    "3": [400],
    "4": [500, 1800],
    "5": [500, 1500],
    "6": [500, 1500],
    "7": [500, 1500],
}
TIERS  = ["A", "B", "C"]       # columns
CASES  = ["1", "2", "3", "4"]  # rows

# ── style ────────────────────────────────────────────────────────────────────
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

ORANGE = "#d95f02"   # median-curve peaks (illustrative)
GREEN  = "#1b9e77"   # ensemble-catalogue overlay

from matplotlib.transforms import ScaledTranslation


def _snap_to_local_max(x, y, x0, w_nodes=3):
    j0  = int(np.argmin(np.abs(x - x0)))
    jlo = max(0, j0 - w_nodes); jhi = min(len(x)-1, j0 + w_nodes)
    jpk = jlo + int(np.argmax(y[jlo:jhi+1]))
    return x[jpk], y[jpk]


def _tri_down_on_line(ax, x, y, *, step, width_steps=1.6, height_data=0.035,
                      facecolor="#1b9e77", edgecolor="k",
                      linewidth=0.6, zorder=6):
    """
    Draw a small *downward* triangle as a polygon with its APEX exactly at (x, y).
    - width set in multiples of grid step (x units, Ma)
    - height set in axis data units (y units)
    """
    w = width_steps * float(step)
    h = float(height_data)
    tri = Polygon([(x, y), (x - 0.5*w, y + h), (x + 0.5*w, y + h)],
                  closed=True, facecolor=facecolor, edgecolor=edgecolor,
                  linewidth=linewidth, zorder=zorder, clip_on=False)
    ax.add_patch(tri)


def _grid_step(x):
    dif = np.diff(np.asarray(x, float))
    return float(np.median(dif)) if dif.size else 10.0

def condense_catalogue(entries, *, step=10.0, merge_within=None,
                       prefer="support", top_k_label=2,
                       ci_overlap_frac=0.50):
    """
    Merge peaks that are either within `merge_within` Ma OR whose
    95% CIs overlap by at least `ci_overlap_frac` of the smaller CI.
    Keep one representative per merged group.
    """
    if not entries:
        return []

    if merge_within is None:
        merge_within = 3.0 * float(step)  # ~3 grid steps by default

    ents = sorted(entries, key=lambda e: float(e["age"]))

    def sup_val(e):
        s = support_pct(e.get("support", np.nan))
        return -np.inf if np.isnan(s) else float(s)

    def ci_width(e):
        return float(e["hi"]) - float(e["lo"])

    def ci_overlap(e1, e2):
        lo = max(float(e1["lo"]), float(e2["lo"]))
        hi = min(float(e1["hi"]), float(e2["hi"]))
        if hi <= lo:
            return False
        span1 = max(ci_width(e1), 1e-9)
        span2 = max(ci_width(e2), 1e-9)
        frac  = (hi - lo) / min(span1, span2)
        return frac >= ci_overlap_frac

    # group by age proximity OR CI overlap
    groups, cur = [], [ents[0]]
    for e in ents[1:]:
        if (abs(float(e["age"]) - float(cur[-1]["age"])) <= merge_within) or ci_overlap(e, cur[-1]):
            cur.append(e)
        else:
            groups.append(cur)
            cur = [e]
    groups.append(cur)

    reps = []
    for g in groups:
        if prefer == "support":
            rep = max(g, key=lambda e: (sup_val(e), -ci_width(e)))
        else:
            rep = min(g, key=lambda e: (ci_width(e), -sup_val(e)))
        rep = dict(rep)
        reps.append(rep)

    # choose which reps get text labels (triangles + CI show for ALL)
    if top_k_label is None or top_k_label < 0 or top_k_label >= len(reps):
        idx_labeled = set(range(len(reps)))
    else:
        order = sorted(range(len(reps)),
                       key=lambda i: (sup_val(reps[i]),
                                      -ci_width(reps[i])),
                       reverse=True)
        idx_labeled = set(order[:int(top_k_label)])

    for i, r in enumerate(reps):
        r["_label"] = (i in idx_labeled)

    return reps

def load_npz_both(sample_tag: str, ks_dir: Path):
    """
    Returns
    -------
    x_ma : (n_grid,) float
    S_raw_runs : (n_runs, n_grid) float
    S_pen_runs : (n_runs, n_grid) float
    """
    # 1) Prefer the aggregate per-run file if present (e.g., 1A_runs_S.npz)
    agg = list(ks_dir.rglob(f"{sample_tag}_runs_S.npz"))
    if agg:
        dat = np.load(agg[0])
        x = (dat["age_Ma"] if "age_Ma" in dat.files else dat["age_ma"]).astype(float)
        S_raw = dat["S_runs_raw"].astype(float) if "S_runs_raw" in dat.files else dat["S_runs_pen"].astype(float)
        S_pen = dat["S_runs_pen"].astype(float) if "S_runs_pen" in dat.files else S_raw.copy()
        return x, S_raw, S_pen

    # 2) Otherwise fall back to per-run bundles (ignore medians-only files)
    files = [p for p in ks_dir.rglob(f"{sample_tag}_*.npz")
             if not (p.name.endswith("_ensemble_surfaces.npz") or p.name.endswith("_runs_S.npz"))]
    if not files:
        raise FileNotFoundError(f"No MC surfaces for {sample_tag}")

    def _first(dat, names):
        for nm in names:
            if nm in dat.files: return dat[nm].astype(float)
        return None

    def _as_2d(a):
        a = np.asarray(a, float)
        return a[None, :] if a.ndim == 1 else a

    x_ma = None
    S_raw_list, S_pen_list = [], []
    used_any = False

    for f in sorted(files):
        dat = np.load(f)
        x = _first(dat, ["age_Ma", "age_ma"])
        if x is None:
            print(f"[skip] {f.name}: no age grid key");  continue
        if x_ma is None: x_ma = x
        elif x.shape != x_ma.shape or not np.allclose(x, x_ma):
            raise ValueError(f"Inconsistent grids for {sample_tag}: {f.name}")

        # Prefer S*; fall back to D*
        S_raw = _first(dat, ["S_runs_raw", "S_raw_runs", "S_raw", "goodness_raw"])
        S_pen = _first(dat, ["S_runs_pen", "S_pen_runs", "S_pen", "S_runs", "S", "goodness_pen", "goodness"])
        if S_raw is None or S_pen is None:
            D_raw = _first(dat, ["D_raw"])
            D_pen = _first(dat, ["D_pen", "D_star", "D"])
            if S_raw is None and D_raw is not None: S_raw = 1.0 - D_raw
            if S_pen is None and D_pen is not None: S_pen = 1.0 - D_pen
            if S_raw is None and S_pen is not None: S_raw = S_pen
            if S_pen is None and S_raw is not None: S_pen = S_raw

        if S_raw is None and S_pen is None:
            print(f"[skip] {f.name}: missing D*/S* keys; has {list(dat.files)}")
            continue

        S_raw = _as_2d(S_raw); S_pen = _as_2d(S_pen)
        if S_raw.shape[-1] != x_ma.size or S_pen.shape[-1] != x_ma.size:
            raise ValueError(f"{f.name}: curve length != grid length")

        if S_raw.shape[0] != S_pen.shape[0]:
            if S_raw.shape[0] == 1: S_raw = np.repeat(S_raw, S_pen.shape[0], axis=0)
            elif S_pen.shape[0] == 1: S_pen = np.repeat(S_pen, S_raw.shape[0], axis=0)

        S_raw_list.append(S_raw); S_pen_list.append(S_pen); used_any = True

    if not used_any:
        raise FileNotFoundError(f"No usable NPZ surfaces for {sample_tag}")
    return x_ma, np.vstack(S_raw_list), np.vstack(S_pen_list)


def load_npz(sample_tag: str):
    """
    Returns:
        x_ma   : 1-D age grid in Ma
        boot   : 2-D array [n_runs, n_grid] of S(t) = 1 – D (or 1 – D*)
    """
    files = sorted(KS_DIR.glob(f"{sample_tag}_*.npz"))
    if not files:
        raise FileNotFoundError(f"No MC surfaces for {sample_tag}")
    x_ma, arrs = None, []
    for f in files:
        dat = np.load(f)
        # defensive keys
        if "age_Ma" in dat.files:
            x = dat["age_Ma"].astype(float)
        elif "age_ma" in dat.files:
            x = dat["age_ma"].astype(float)
        else:
            raise KeyError(f"{f.name}: no 'age_Ma'/'age_ma' in npz")

        if x_ma is None:
            x_ma = x
        else:
            if (x.shape != x_ma.shape) or not np.allclose(x, x_ma):
                raise ValueError(f"Inconsistent age grids among {sample_tag} npz files.")

        if "D" in dat.files:
            D = dat["D"].astype(float)
        elif "D_star" in dat.files:
            D = dat["D_star"].astype(float)
        else:
            raise KeyError(f"{f.name}: no 'D' or 'D_star' in npz")

        s = 1.0 - D
        arrs.append(s)
    return x_ma, np.vstack(arrs)

from matplotlib.lines import Line2D  # already imported above

def _legend_handles(include_median_peaks=True):
    handles = [
        Line2D([0],[0], color="0.75", lw=1.2, label="all runs $S(t)$"),
        Line2D([0],[0], color="k", lw=1.5, label="median $S(t)$"),
        Line2D([0],[0], color="0.6", lw=0.8, ls="--", label="true age"),
        Line2D([0],[0], marker="v", markersize=6, markerfacecolor=GREEN, markeredgecolor="k",
               lw=0, label="ensemble age"),
        Line2D([0],[0], color=GREEN, lw=1.6, label="95% CI"),
    ]
    if include_median_peaks:
        handles.insert(2, Line2D([0],[0], marker="o", markersize=5,
                                 markerfacecolor=ORANGE, markeredgecolor=ORANGE,
                                 lw=0, label="peaks (median)"))
    return handles


def support_pct(raw, n_runs=None):
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return np.nan
    v = float(raw)
    if v <= 1.0 + 1e-12:   # fraction
        p = 100.0 * v
    elif v <= 100.0 + 1e-9:  # percent
        p = v
    elif n_runs:
        p = 100.0 * v / float(n_runs)
    else:
        p = v
    return float(np.clip(p, 0.0, 100.0))

def _norm_tag(s: str) -> str:
    return str(s).strip().upper()

def load_catalogue_table(path: Path):
    """
    CSV must contain columns (case-insensitive):
      sample, peak_no, age_ma, ci_low, ci_high, support
    Returns dict: { '1A': [ {age, lo, hi, support}, ... ], ... }
    Robust to thousands separators (1,746), percent signs, and stray spaces.
    """
    import io, re
    if not path.exists():
        print(f"[overlay] catalogue CSV not found: {path}")
        return {}

    reqcols = {"sample","peak_no","age_ma","ci_low","ci_high","support"}

    def _read_csv_like(obj):
        # Use python engine (handles ragged rows); keep only needed columns.
        common = dict(
            comment="#",
            skip_blank_lines=True,
            encoding="utf-8-sig",
            engine="python",
            sep=",",
            skipinitialspace=True,
            usecols=lambda c: str(c).strip().lower() in reqcols
        )
        try:  # pandas ≥1.3
            return pd.read_csv(obj, on_bad_lines="warn", **common)
        except TypeError:  # pandas <1.3
            return pd.read_csv(obj, error_bad_lines=False, warn_bad_lines=True, **common)

    try:
        df = _read_csv_like(path)
    except pd.errors.ParserError:
        # Remove thousands separators inside numbers (e.g., 1,746 -> 1746) and retry.
        text = path.read_text(encoding="utf-8-sig")
        text = re.sub(r'(?<=\d),(?=\d{3}(?:\D|$))', '', text)
        df = _read_csv_like(io.StringIO(text))

    # Normalise headers and coerce numerics
    df.columns = [str(c).strip().lower() for c in df.columns]
    for col in ["age_ma", "ci_low", "ci_high", "support"]:
        df[col] = pd.to_numeric(
            df[col].astype(str)
                  .str.replace('%', '', regex=False)
                  .str.replace(',', '', regex=False)
                  .str.strip(),
            errors="coerce"
        )

    # Build map
    out = {}
    keep = df.dropna(subset=["sample", "age_ma", "ci_low", "ci_high"])
    for _, r in keep.iterrows():
        samp = _norm_tag(r["sample"])
        out.setdefault(samp, []).append(dict(
            age=float(r["age_ma"]),
            lo=float(r["ci_low"]),
            hi=float(r["ci_high"]),
            support=(float(r["support"]) if pd.notna(r["support"]) else np.nan),
        ))
    for k in out:
        out[k].sort(key=lambda d: d["age"])

    print(f"[overlay] catalogue entries for samples: {', '.join(sorted(out.keys()))}")
    return out

def draw_catalogue_above(ax, entries, n_runs, step, y_base=1.035, dy=0.055):
    for i, e in enumerate(entries):
        y = y_base + i * dy
        # CI bar (always)
        ax.plot([e["lo"], e["hi"]], [y, y], color=GREEN, lw=1.6,
                zorder=5, clip_on=False)
        # triangle whose tip sits on the CI bar
        _tri_down_on_line(ax, e["age"], y, step=step, facecolor=GREEN,
                          edgecolor="k", linewidth=0.5, zorder=6)
        # optional text
        if e.get("_label", True):
            pct = support_pct(e.get("support"), n_runs=n_runs)
            label = f"{e['age']:.0f}" if not np.isfinite(pct) \
                    else f"{e['age']:.0f} ({int(round(pct))}%)"
            ax.text(e["age"], y + 0.012, label, ha="center", va="bottom",
                    fontsize=7, color="k", zorder=7, clip_on=False)

def draw_catalogue_on_curve(ax, entries, x, y_curve, *, show_support=True,
                            n_runs=None, capsize=3):
    """
    Triangles sit on the provided 1-D curve y_curve(x).
    Asymmetric CI whiskers in x; optional support label.
    """
    for e in entries:
        y = float(np.interp(e["age"], x, y_curve))
        # left/right x-uncertainties
        xerr = np.array([[max(0.0, e["age"] - e["lo"])],
                         [max(0.0, e["hi"]  - e["age"])]], dtype=float)
        ax.errorbar(
            e["age"], y,
            xerr=xerr, yerr=None,
            fmt="^", ms=6, mfc=GREEN, mec="k", mew=0.6,
            ecolor=GREEN, elinewidth=1.3, capsize=capsize, zorder=6
        )
        if e.get("_label", True) and show_support:
            pct = support_pct(e.get("support"), n_runs=n_runs)
            label = f"{e['age']:.0f}" if not np.isfinite(pct) else f"{e['age']:.0f} ({int(round(pct))}%)"
            ax.text(e["age"], y + 0.03, label, ha="center", va="bottom",
                    fontsize=7, color="k", zorder=7, clip_on=False)
def save_figure(fig, stub: str):
    paths = []
    for ext in ("png", "pdf", "svg"):
        p = OUTDIR / f"{stub}.{ext}"
        fig.savefig(p, bbox_inches="tight", pad_inches=0.02)
        paths.append(str(p))
    print("[saved]\n  " + "\n  ".join(paths))

def make_grid(cases, title, outfile_stub, catalogue_map,
              overlay_mode="curve", legend_compact=True,
              show_median_peaks=False, label_median_peaks=False):

    assert overlay_mode in {"above", "curve"}
    n_row, n_col = len(cases), len(TIERS)
    fig, axes = plt.subplots(n_row, n_col, figsize=(10, 3.3*n_row),
                             sharex=True, sharey=True)

    any_overlay = False
    max_entries = 0

    for r, case in enumerate(cases):
        for c, tier in enumerate(TIERS):
            ax  = axes[r, c]
            tag = f"{case}{tier}".upper()  # e.g., '1A'

            try:
                x, S_raw, S_pen = load_npz_both(tag, KS_DIR)
            except FileNotFoundError:
                ax.set_xlim(0, 2000); ax.set_ylim(0, 1.05)
                ax.set_title(f"Tier {tier} (no NPZ for {tag})", color="crimson", fontsize=9)
                for age in TRUE.get(case, []):
                    ax.axvline(age, ls="--", lw=0.8, color="0.6", zorder=0)
                continue  # ← only when files truly missing

            # ---- draw curves (this must be OUTSIDE the except) ----
            boot_curve = S_pen if CURVE_SURFACE    == "pen" else S_raw
            tri_curve  = S_pen if TRIANGLE_SURFACE == "pen" else S_raw
            y_med      = np.nanmedian(boot_curve, axis=0)
            y_tri_med  = np.nanmedian(tri_curve,  axis=0)

            # --- median-curve peak markers (optional) ---
            if show_median_peaks:
                pk, _ = find_peaks(y_med, prominence=0.02, width=3)
                if pk.size:
                    ax.plot(x[pk], y_med[pk], "o", ms=4,
                            mfc=ORANGE, mec=ORANGE, lw=0, zorder=3)
                    for j in pk:
                        ax.text(x[j], y_med[j] + 0.02, f"{x[j]:.0f}",
                                ha="center", va="bottom", fontsize=7, color=ORANGE)

            for y in boot_curve:
                ax.plot(x, y, color="0.75", lw=0.4, alpha=0.6, zorder=1)
            ax.plot(x, y_med, color="k", lw=1.5, zorder=2)

            for age in TRUE.get(case, []):
                ax.axvline(age, ls="--", lw=0.8, color="0.6", zorder=0)

            # overlay from penalised catalogue
            entries_raw = catalogue_map.get(_norm_tag(tag), [])
            n_runs = boot_curve.shape[0]
            if entries_raw:
                any_overlay = True
                step = _grid_step(x)
                entries = condense_catalogue(entries_raw, step=step,
                                            merge_within=3.0*step,
                                            prefer="support", top_k_label=2,
                                            ci_overlap_frac=0.50)
                if overlay_mode == "above":
                    max_entries = max(max_entries, len(entries))
                    draw_catalogue_above(ax, entries, n_runs, step)
                else:
                    draw_catalogue_on_curve(ax, entries, x, y_tri_med,
                                            show_support=True, n_runs=n_runs, capsize=3)

            ax.set_xlim(0, 2000)
            if r == n_row-1: ax.set_xlabel("Pb-loss age (Ma)")
            if c == 0:       ax.set_ylabel("Normalised goodness, $S$")
            if r == 0:       ax.set_title(f"Tier {tier}", fontweight="bold")
            if c == n_col-1:
                ax.text(1.04, 0.5, f"Case {case}", transform=ax.transAxes,
                        rotation=-90, va="center", ha="left",
                        fontsize=9, fontweight="bold")


    # y-limits / legend / save
    if any_overlay:
        if overlay_mode == "above":
            y_base, dy = 1.035, 0.055
            needed_top = max(1.12, y_base + (max_entries - 1) * dy + 0.06)
        else:
            needed_top = 1.12
        for ax in np.ravel(axes):
            if ax.has_data(): ax.set_ylim(0, needed_top)
    else:
        for ax in np.ravel(axes):
            if ax.has_data(): ax.set_ylim(0, 1.05)

    if legend_compact:
        fig.legend(handles=_legend_handles(include_median_peaks=show_median_peaks),
                   loc="upper center", ncol=6, frameon=False,
                   bbox_to_anchor=(0.5, 1.00), borderaxespad=0.6)

    fig.suptitle(title, y=0.985, fontweight="bold")
    fig.tight_layout(h_pad=0.9, w_pad=0.7, rect=[0, 0, 1, 0.97])
    save_figure(fig, outfile_stub)
    plt.show()

# ── run: Cases 1–4 then 5–7 ─────────────────────────────────────────────────
catalogue_map = load_catalogue_table(CATALOGUE_CSV)

# Cases 1–4
make_grid(CASES,
          "CDC goodness surfaces with ensemble catalogue — Cases 1–4 × Tiers A–C",
          "Fig_goodness_grid_cases1-4_with_catalogue",
          catalogue_map, overlay_mode="above",
          legend_compact=True, show_median_peaks=False)

# Cases 5–7  (was show_median_peaks=True)
make_grid(["5","6","7"],
          "CDC goodness surfaces with ensemble catalogue — Cases 5–7 × Tiers A–C",
          "Fig_goodness_grid_cases5-7_with_catalogue",
          catalogue_map, overlay_mode="curve",
          legend_compact=True, show_median_peaks=False)

