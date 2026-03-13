import time
import numpy as np
from typing import Optional

from model.settings.type import SettingsType
from utils.settings import Settings
from view.axes.heatmapAxis import HeatmapAxis
from view.axes.goodnessAxis import GoodnessAxis
from view.figures.abstractFigure import AbstractFigure


class SampleOutputFigure(AbstractFigure):
    """
    Top: KS distance map.
    Bottom: Goodness (median of 1 - D*) with automatic peak markers / CI windows.
    """

    def __init__(self, controller, sample):
        super().__init__()
        self.sample = sample
        self.lastDrawTime = None

        # --- subplots once ---
        ax_hm = self.fig.add_subplot(211)
        ax_gd = self.fig.add_subplot(212)

        self.heatmapAxis  = HeatmapAxis(ax_hm, self.canvas, self.fig)
        self.goodnessAxis = GoodnessAxis(ax_gd)

        self._ens_ages_ma = None   # cached ensemble x from processing
        self._ens_S_view  = None   # cached ensemble curve from processing
        self._ens_label   = None   # optional label "RAW"/"PEN" if you pass one later

        ax_hm.tick_params(labelbottom=False)
        ax_hm.set_xlabel("")
        self.fig.subplots_adjust(hspace=0.28)
        self.fig.set_constrained_layout(True)

        # wire signals
        sample.signals.concordancyCalculated.connect(self._onSampleConcordancyCalculated)
        sample.signals.monteCarloRunAdded.connect(self._onMonteCarloRunAdded)
        sample.signals.processingCleared.connect(self._onProcessingCleared)
        sample.signals.optimalAgeCalculated.connect(self._onOptimalAgeCalculated)
        try:
            if hasattr(sample.signals, "summedKS"):
                sample.signals.summedKS.connect(self._onSummedKS)
        except Exception:
            pass

        self.processingComplete = False
        self.mouseOnStatisticsAxes = False

    def highlight_catalogue_row(self, row_index: Optional[int] = None):
        """Called by the results panel when a catalogue row is selected."""
        self.goodnessAxis.highlight_peak(row_index if isinstance(row_index, int) else None)
        self.canvas.draw_idle()

    def _onSummedKS(self, payload):
        try:
            ages_ma, S_view, peaks, *rest = payload
        except Exception:
            return

        ages_ma = np.asarray(ages_ma, float)
        S_view  = np.asarray(S_view,  float)

        # basic guards
        if ages_ma.size == 0 or S_view.size == 0:
            print("[UI] summedKS payload is empty:", ages_ma.size, S_view.size)
            return
        if ages_ma.size != S_view.size:
            print("[UI] summedKS length mismatch: ages=", ages_ma.size, " S=", S_view.size)
            return
        if not np.isfinite(S_view).any():
            print("[UI] summedKS S_view has no finite values")
            return

        # optional CI windows
        wins = None
        if rest:
            ci_pairs = rest[0]
            if isinstance(ci_pairs, (list, tuple)) and ci_pairs and isinstance(ci_pairs[0], (list, tuple)):
                try:
                    wins = [(float(a), float(b)) for a, b in ci_pairs]
                except Exception:
                    wins = None

        # cache for refresh
        self._ens_ages_ma = ages_ma
        self._ens_S_view  = S_view
        self._ens_label   = None

        # draw: windows → curve → peaks; keep heatmap in sync
        if wins:
            self.goodnessAxis.set_windows(wins)

        # IMPORTANT: call update_curve to actually render the line
        self.goodnessAxis.update_curve(
            ages_ma, S_view,
            peaks=[float(p) for p in (peaks or [])] if peaks is not None else None,
            windows=wins
        )

        self.heatmapAxis.set_curve(ages_ma, S_view)
        self.heatmapAxis.set_peaks(peaks or [])
        self.canvas.draw_idle()


    # ----------------- helpers -----------------

    @staticmethod
    def _sanitise_catalogue(raw_rows):
        """
        Make the catalogue robust against legacy/duplicate/extended rows.
        - Accept list/tuple/dict or empty.
        - Keep strongest support per ~10 Ma bucket.
        - Cap to top 8 by support; final sort by age.
        """
        rows = raw_rows or []
        if isinstance(rows, dict):
            rows = [rows]
        if isinstance(rows, tuple):
            rows = list(rows)
        if not isinstance(rows, list):
            return []

        out = []
        for r in rows:
            if isinstance(r, dict):
                try:
                    out.append(dict(
                        age_ma  = float(r.get("age_ma")),
                        ci_low  = float(r.get("ci_low")),
                        ci_high = float(r.get("ci_high")),
                        support = float(r.get("support", float("nan"))),
                    ))
                except Exception:
                    pass
            elif isinstance(r, (list, tuple)) and len(r) >= 3:
                try:
                    age, lo, hi = r[:3]
                    sup = (r[3] if len(r) >= 4 else float("nan"))
                    out.append(dict(
                        age_ma  = float(age),
                        ci_low  = float(lo),
                        ci_high = float(hi),
                        support = float(sup),
                    ))
                except Exception:
                    pass

        # bucket within 10 Ma; keep max support per bucket
        buckets = {}
        for r in out:
            key = round(r["age_ma"] / 10.0)
            sup = r.get("support", float("nan"))
            if key not in buckets or (np.isfinite(sup) and sup > buckets[key][0]):
                buckets[key] = (sup, r)

        # compact list; keep at most 8 best-supported
        compact = [v[1] for v in buckets.values()]
        compact = sorted(compact, key=lambda d: float(d.get("support", 0.0)), reverse=True)[:8]
        compact.sort(key=lambda d: d["age_ma"])
        return compact

    def _display_rows_from_summedks(self):
        """Rows used for figure markers/windows from the plotted summedKS payload."""
        peaks = np.asarray(getattr(self.sample, "summedKS_peaks_Ma", None), float)
        lows = np.asarray(getattr(self.sample, "summedKS_ci_low_Ma", None), float)
        highs = np.asarray(getattr(self.sample, "summedKS_ci_high_Ma", None), float)

        peaks = np.atleast_1d(peaks)
        lows = np.atleast_1d(lows)
        highs = np.atleast_1d(highs)

        # Handle scalar/empty placeholders (e.g., np.array(nan))
        if peaks.size == 1 and (not np.isfinite(peaks[0])):
            return []

        if peaks.size == 0:
            return []

        rows = []
        for i, a in enumerate(peaks):
            if not np.isfinite(a):
                continue
            lo = float(lows[i]) if i < lows.size and np.isfinite(lows[i]) else float(a)
            hi = float(highs[i]) if i < highs.size and np.isfinite(highs[i]) else float(a)
            if hi < lo:
                lo, hi = hi, lo
            rows.append(dict(age_ma=float(a), ci_low=lo, ci_high=hi, support=float("nan")))
        return rows

    # ----------------- lifecycle / clearing -----------------

    def clearProcessingResults(self):
        self.processingComplete = False
        self.lastDrawTime = None
        self._ens_ages_ma = None
        self._ens_S_view = None
        self._ens_label = None
        self.heatmapAxis.clearAll()
        self.goodnessAxis.clear()

    def clearInputData(self):
        self.clearProcessingResults()

    def _onProcessingCleared(self):
        self.clearProcessingResults()
        self.canvas.draw_idle()

    def _onSampleConcordancyCalculated(self):
        return

    # ----------------- optimal-age path (catalogue-first) -----------------

    def _onOptimalAgeCalculated(self):
        # Update catalogue + windows only; curve already pushed via _onSummedKS
        # Update catalogue + windows only if the toggle is ON; curve already came via _onSummedKS
        st = getattr(self.sample, "calculationSettings", None)
        if st is not None and getattr(self.sample, "monteCarloRuns", None):
            # Final optimal-age delivery can change the authoritative surface
            # (e.g. clustered fallback to global_all), so refresh the heatmap
            # from the final per-run cached columns now that processing is done.
            self.heatmapAxis.plotRuns(self.sample.monteCarloRuns, st)
        if not getattr(st, "enable_ensemble_peak_picking", False):
            self.goodnessAxis.set_peak_catalogue([])
            self.goodnessAxis.set_windows([])
            self.canvas.draw_idle()
            return

        rows = self._display_rows_from_summedks()
        if not rows:
            rows = self._sanitise_catalogue(getattr(self.sample, "peak_catalogue", []) or [])
        if rows:
            self.goodnessAxis.set_peak_catalogue(rows)
            wins = [(float(r["ci_low"]), float(r["ci_high"])) for r in rows]
            self.goodnessAxis.set_windows(wins)
        else:
            self.goodnessAxis.set_windows([])
        self.canvas.draw_idle()

    # ----------------- incremental updates -----------------

    def _onMonteCarloRunAdded(self):
        # Prefer the frozen settings that produced the runs
        settings = getattr(self.sample, "calculationSettings", None) or Settings.get(SettingsType.CALCULATION)
        now = time.time()
        if len(self.sample.monteCarloRuns) == settings.monteCarloRuns or \
        self.lastDrawTime is None or \
        (now - self.lastDrawTime) >= 0.75:

            self.lastDrawTime = now

            # Heatmap refresh during processing. Curve/markers/windows are only
            # updated from _onSummedKS to avoid preview-vs-final flicker/races.
            self.heatmapAxis.plotRuns(self.sample.monteCarloRuns, settings)

            self.canvas.draw_idle()
