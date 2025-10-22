from __future__ import annotations
from typing import Sequence, Tuple, Optional, Iterable, List, Union
import numpy as np
from matplotlib.lines import Line2D

class GoodnessAxis:
    """
    Ensemble goodness panel (median across runs).
    x: Age (Ma), y: S = 1 - D or 1 - D* (0..1).
    Draws the curve, peak markers at their y on the curve, and shaded CI windows.
    """
    def __init__(self, ax, *, title: str = "Goodness"):
        self.ax = ax
        self.ax.set_title(title)
        self.ax.set_xlabel("Age (Ma)")
        self.ax.set_ylabel("Goodness S")
        # self.ax.set_xlim(left=0)
        (self._line,)  = self.ax.plot([], [], lw=1.8)
        (self._peaks,) = self.ax.plot([], [], ls="", marker="o", ms=6, mfc="none", mec="red")

        self._ages_ma_last: Optional[np.ndarray] = None
        self._y_last: Optional[np.ndarray] = None
        self._catalogue_rows: List[dict] = []

        self._win_patches: List = []
        self._win_edges: List[Line2D] = []
        self._hi_line: Optional[Line2D] = None

    # ------------------------------- public API -------------------------------

    def set_curve(self, ages_ma, y):
        ages_ma = np.asarray(ages_ma, float)
        y       = np.asarray(y, float)

        self._ages_ma_last = ages_ma
        self._y_last       = y

        m = np.isfinite(ages_ma) & np.isfinite(y)
        if m.any():
            self._line.set_data(ages_ma[m], y[m])
        else:
            self._line.set_data([], [])

        if self._catalogue_rows and m.any():
            xs = np.array([float(r.get("age_ma", np.nan)) for r in self._catalogue_rows], float)
            xs = np.clip(xs, ages_ma[m][0], ages_ma[m][-1])
            j  = np.clip(np.searchsorted(ages_ma[m], xs), 1, ages_ma[m].size - 1)
            jl = j - 1
            pick = np.where(np.abs(ages_ma[m][j] - xs) < np.abs(ages_ma[m][jl] - xs), j, jl)
            ys  = y[m][pick]
            self._peaks.set_data(xs, ys)

        self.ax.relim(); self.ax.autoscale_view()
        self.ax.set_xlim(left=0)
        self.ax.figure.canvas.draw_idle()


    def update_curve(self, ages_ma, y, peaks=None, windows=None, label=None):
        ages_ma = np.asarray(ages_ma, float)
        y       = np.asarray(y, float)

        self._ages_ma_last = ages_ma
        self._y_last       = y

        m = np.isfinite(ages_ma) & np.isfinite(y)
        if m.any():
            ax_x = ages_ma[m]; ax_y = y[m]
            self._line.set_data(ax_x, ax_y)
        else:
            self._line.set_data([], [])
            ax_x = None

        # optional peak markers (use nearest node on the finite subset)
        if peaks is not None and len(peaks) and m.any():
            xs = np.asarray(peaks, float)
            xs = np.clip(xs, ax_x[0], ax_x[-1])
            j  = np.clip(np.searchsorted(ax_x, xs), 1, ax_x.size - 1)
            jl = j - 1
            pick = np.where(np.abs(ax_x[j] - xs) < np.abs(ax_x[jl] - xs), j, jl)
            ys = ax_y[pick]
            self._peaks.set_data(xs, ys)
        elif self._catalogue_rows and m.any():
            xs = np.array([float(r.get("age_ma", np.nan)) for r in self._catalogue_rows], float)
            xs = np.clip(xs, ax_x[0], ax_x[-1])
            j  = np.clip(np.searchsorted(ax_x, xs), 1, ax_x.size - 1)
            jl = j - 1
            pick = np.where(np.abs(ax_x[j] - xs) < np.abs(ax_x[jl] - xs), j, jl)
            ys = ax_y[pick]
            self._peaks.set_data(xs, ys)

        # CI windows
        if windows:
            self.set_windows(windows, primary_idx=0 if len(windows) > 0 else None)

        self.ax.relim(); self.ax.autoscale_view()
        self.ax.set_xlim(left=0)
        self.ax.figure.canvas.draw_idle()



    def set_peak_catalogue(self, rows: Optional[List[Union[dict, tuple]]]) -> None:
        parsed: List[dict] = []
        if rows:
            for r in rows:
                if isinstance(r, dict):
                    parsed.append(dict(
                        age_ma=float(r["age_ma"]),
                        ci_low=float(r["ci_low"]),
                        ci_high=float(r["ci_high"]),
                        support=float(r.get("support", np.nan)),
                    ))
                else:
                    age, lo, hi, *rest = r
                    sup = rest[0] if rest else np.nan
                    parsed.append(dict(age_ma=float(age), ci_low=float(lo), ci_high=float(hi), support=float(sup)))
        self._catalogue_rows = parsed

        wins = [(r["ci_low"], r["ci_high"]) for r in parsed
                if np.isfinite(r["ci_low"]) and np.isfinite(r["ci_high"]) and r["ci_high"] > r["ci_low"]]
        self.set_windows(wins, primary_idx=(0 if wins else None))

        # place markers at catalogue medians on top of the curve
        if self._ages_ma_last is not None and self._y_last is not None and parsed:
            xs = np.array([r["age_ma"] for r in parsed], float)
            # clamp into grid and use nearest node (handles the rightmost peak)
            xs = np.clip(xs, self._ages_ma_last[0], self._ages_ma_last[-1])
            j  = np.clip(np.searchsorted(self._ages_ma_last, xs), 1, self._ages_ma_last.size - 1)
            jl = j - 1
            pick = np.where(np.abs(self._ages_ma_last[j] - xs) < np.abs(self._ages_ma_last[jl] - xs), j, jl)
            ys = self._y_last[pick]
            self._peaks.set_data(xs, ys)


        self.ax.figure.canvas.draw_idle()

    def highlight_peak(self, idx: Optional[int]) -> None:
        self._remove_hi_line()
        if idx is None or idx < 0 or idx >= len(self._catalogue_rows):
            self.ax.figure.canvas.draw_idle(); return

        wins = [(r["ci_low"], r["ci_high"]) for r in self._catalogue_rows
                if np.isfinite(r["ci_low"]) and np.isfinite(r["ci_high"]) and r["ci_high"] > r["ci_low"]]
        self.set_windows(wins, primary_idx=(idx if idx < len(wins) else None))

        x = float(self._catalogue_rows[idx]["age_ma"])
        if np.isfinite(x):
            self._hi_line = self.ax.axvline(x, lw=1.6, ls="--", color=self._line.get_color(), alpha=0.85)
        self.ax.figure.canvas.draw_idle()

    def clear(self) -> None:
        self._line.set_data([], []); self._peaks.set_data([], [])
        self._ages_ma_last = None; self._y_last = None; self._catalogue_rows = []
        self._clear_windows(); self._remove_hi_line()
        self.ax.relim(); self.ax.autoscale_view(); self.ax.figure.canvas.draw_idle()

    # ------------------------------- internals --------------------------------

    def set_windows(
        self,
        windows_ma: Sequence[Tuple[float, float]],
        *,
        primary_idx: Optional[int] = None,
        alpha_main: float = 0.28,
        alpha_other: float = 0.12,
        draw_primary_edge: bool = True,
    ) -> None:
        self._clear_windows()
        if not windows_ma: return
        for i, (lo, hi) in enumerate(windows_ma):
            lo = float(lo); hi = float(hi)
            if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo: continue
            is_primary = (primary_idx is not None and i == int(primary_idx))
            a = alpha_main if is_primary else alpha_other
            patch = self.ax.axvspan(lo, hi, alpha=a, zorder=0, lw=0)
            self._win_patches.append(patch)
            if is_primary and draw_primary_edge:
                (edge_line,) = self.ax.plot([lo, hi], [0, 0], lw=1.0)
                edge_line.set_transform(self.ax.get_xaxis_transform())
                self._win_edges.append(edge_line)

    def _clear_windows(self):
        for p in self._win_patches:
            try: p.remove()
            except Exception: pass
        for e in self._win_edges:
            try: e.remove()
            except Exception: pass
        self._win_patches = []; self._win_edges = []

    def _remove_hi_line(self):
        if self._hi_line is not None:
            try: self._hi_line.remove()
            except Exception: pass
            self._hi_line = None

    # Optional legacy helper
    def update_from_runs(self, runs, ages_grid_years: Optional[Iterable[float]] = None, **_):
        runs = runs or []
        winners_ma = [float(getattr(r, "optimal_pb_loss_age", np.nan)) / 1e6 for r in runs]
        winners_ma = np.array([w for w in winners_ma if np.isfinite(w)], float)
        if winners_ma.size == 0:
            self.clear(); return

        if ages_grid_years is not None:
            ages_ma = np.asarray(ages_grid_years, float) / 1e6
        else:
            try:
                grid_y = np.array(sorted(runs[0].statistics_by_pb_loss_age.keys()), float)
                ages_ma = grid_y / 1e6
            except Exception:
                lo, hi = np.nanmin(winners_ma), np.nanmax(winners_ma)
                ages_ma = np.linspace(lo, hi, 401)

        edges = np.empty(ages_ma.size + 1, float)
        edges[1:-1] = 0.5 * (ages_ma[:-1] + ages_ma[1:])
        step = ages_ma[1] - ages_ma[0] if ages_ma.size > 1 else 1.0
        edges[0]  = ages_ma[0]  - 0.5 * step
        edges[-1] = ages_ma[-1] + 0.5 * step
        eps = 1e-9 * max(1.0, step)

        winners_ma = np.clip(winners_ma, edges[0] + eps, edges[-1] - eps)
        counts, _ = np.histogram(winners_ma, bins=edges)
        y = counts.astype(float); s = float(y.sum())
        if s > 0: y /= s

        self.update_curve(ages_ma, y)
