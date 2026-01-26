from copy import deepcopy
from PyQt5.QtCore import pyqtSignal, QObject
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from model.settings.calculation import ConcordiaMode  # you already use this elsewhere

@dataclass
class ModeResults:
    # sample-level outputs
    optimalAge: Optional[float] = None
    optimalAgeLowerBound: Optional[float] = None
    optimalAgeUpperBound: Optional[float] = None
    optimalAgeDValue: Optional[float] = None
    optimalAgePValue: Optional[float] = None
    optimalAgeNumberOfInvalidPoints: Optional[float] = None
    optimalAgeScore: Optional[float] = None

    peak_catalogue: List[Any] = field(default_factory=list)
    disc_cluster_labels: Any = None
    disc_cluster_summary: List[Any] = field(default_factory=list)

    monteCarloRuns: List[Any] = field(default_factory=list)

    processed: List[bool] = field(default_factory=list)
    concordant: List[Optional[bool]] = field(default_factory=list)
    discordance: List[Optional[float]] = field(default_factory=list)
    reverse: List[bool] = field(default_factory=list)

class Sample:
    def __init__(self, id, name, spots):
        self.id = id
        self.name = name
        self.spots = spots

        self.peak_catalogue = []

        self.disc_cluster_labels = None        
        self.disc_cluster_summary = []      

        self.validSpots = [spot for spot in self.spots if spot.valid]
        self.invalidSpots = [spot for spot in self.spots if not spot.valid]

        self.signals = SampleSignals()

        self.calculationSettings = None
        self.optimalAge = None
        self.optimalAgeLowerBound = None
        self.optimalAgeUpperBound = None
        self.optimalAgeDValue = None
        self.optimalAgePValue = None
        self.optimalAgeNumberOfInvalidPoints = None
        self.optimalAgeScore = None
        self.monteCarloRuns = []

        self._results_by_mode: Dict[ConcordiaMode, ModeResults] = {}
        self.activeConcordiaMode: ConcordiaMode = ConcordiaMode.TW

        # Goodness curve cache (for exporting curve values)
        self.summedKS_ages_Ma = None       # np.ndarray shape (n,)
        self.summedKS_goodness = None      # np.ndarray shape (n,)

        self.skip_reason = None

    @property
    def peak_catalogue(self):
        return getattr(self, "_peak_catalogue", [])

    @peak_catalogue.setter
    def peak_catalogue(self, val):
        if not isinstance(val, list):
            import traceback
            print("[CDC] BAD peak_catalogue assignment:", type(val), repr(val))
            traceback.print_stack(limit=6)
            val = []
        self._peak_catalogue = val

    def concordantSpots(self):
        # Use explicit bool() checks so numpy.bool_ values are handled and skip unprocessed spots.
        return [
            spot for spot in self.validSpots
            if spot.processed and bool(spot.concordant)
        ]

    def discordantSpots(self):
        out = []
        for spot in self.validSpots:
            c = getattr(spot, "concordant", None)
            if c is None:
                continue  # not classified yet
            if (not bool(c)) and (not getattr(spot, "reverseDiscordant", False)):
                out.append(spot)
        return out


    def reverseDiscordantSpots(self):
        return [
            spot for spot in self.validSpots
            if spot.processed and getattr(spot, "reverseDiscordant", False)
        ]

    def setSkipReason(self, reason):
        self.skip_reason = reason
        if self.signals:
            self.signals.skipped.emit()

    ##################
    ## Calculations ##
    ##################

    def startCalculation(self, calculationSettings):
        self.calculationSettings = calculationSettings
        mode = getattr(calculationSettings, "concordiaMode", None)
        if mode is not None:
            self.activeConcordiaMode = ConcordiaMode.coerce(mode)

    def clearCalculation(self):
        mode = self._current_mode()
        self._results_by_mode.pop(mode, None)

        self.optimalAge = None
        self.monteCarloRuns = []
        self.peak_catalogue = []
        self.disc_cluster_labels = None
        self.disc_cluster_summary = []

        for spot in self.spots:
            spot.clear()

        self.signals.processingCleared.emit()
        self.summedKS_ages_Ma = None
        self.summedKS_goodness = None


    def updateConcordance(self, concordancy, discordances, reverse_flags=None):
        for i, (spot, conc, disc) in enumerate(zip(self.validSpots, concordancy, discordances)):
            spot.updateConcordance(conc, disc)
            if reverse_flags is not None and i < len(reverse_flags):
                spot.reverseDiscordant = bool(reverse_flags[i])
        if self.signals:
            self._snapshot_mode_results() 
            self.signals.concordancyCalculated.emit()

    def addMonteCarloRun(self, run):
        self.monteCarloRuns.append(run)
        if self.signals:
            self.signals.monteCarloRunAdded.emit()

    def setOptimalAge(self, args):
        self.optimalAge = args[0]
        self.optimalAgeLowerBound = args[1]
        self.optimalAgeUpperBound = args[2]
        self.optimalAgeDValue = args[3]
        self.optimalAgePValue = args[4]
        self.optimalAgeNumberOfInvalidPoints = args[5]
        self.optimalAgeScore = args[6]

        idx_catalogue = None
        if len(args) >= 9 and isinstance(args[8], (list, tuple)):
            idx_catalogue = 8                      
        elif len(args) >= 8 and isinstance(args[7], (list, tuple)):
            idx_catalogue = 7                       

        if idx_catalogue is not None:
            self.peak_catalogue = list(args[idx_catalogue] or [])
        else:
            pass

        disc_idx = (idx_catalogue + 1) if idx_catalogue is not None else 9
        if len(args) > disc_idx:
            payload = args[disc_idx]
            if isinstance(payload, dict):
                self.disc_cluster_labels = payload.get("labels")
                self.disc_cluster_summary = payload.get("summary", [])
            else:
                self.disc_cluster_labels = payload
                self.disc_cluster_summary = args[disc_idx + 1] if len(args) > disc_idx + 1 else []

        if self.signals:
            self._snapshot_mode_results()
            self.signals.optimalAgeCalculated.emit()

    def createProcessingCopy(self):
        signals = self.signals
        saved_results = getattr(self, "_results_by_mode", None)
        saved_active  = getattr(self, "activeConcordiaMode", None)

        self.signals = None
        self._results_by_mode = {}  # do not deepcopy cached results
        copy = deepcopy(self)

        self.signals = signals
        self._results_by_mode = saved_results if saved_results is not None else {}
        self.activeConcordiaMode = saved_active

        return copy


    def getMonteCarloRuns(self):
        return self.monteCarloRuns

    def _current_mode(self) -> ConcordiaMode:
        mode = getattr(self.calculationSettings, "concordiaMode", None)
        return ConcordiaMode.coerce(mode) if mode is not None else self.activeConcordiaMode

    def _snapshot_mode_results(self, mode: Optional[ConcordiaMode] = None):
        mode = ConcordiaMode.coerce(mode) if mode is not None else self._current_mode()

        res = ModeResults()
        res.optimalAge = self.optimalAge
        res.optimalAgeLowerBound = self.optimalAgeLowerBound
        res.optimalAgeUpperBound = self.optimalAgeUpperBound
        res.optimalAgeDValue = self.optimalAgeDValue
        res.optimalAgePValue = self.optimalAgePValue
        res.optimalAgeNumberOfInvalidPoints = self.optimalAgeNumberOfInvalidPoints
        res.optimalAgeScore = self.optimalAgeScore

        # keep whatever your catalogue type is (tuples/dicts) as-is
        res.peak_catalogue = deepcopy(self.peak_catalogue) if self.peak_catalogue else []
        res.disc_cluster_labels = deepcopy(self.disc_cluster_labels)
        res.disc_cluster_summary = deepcopy(self.disc_cluster_summary) if self.disc_cluster_summary else []

        res.monteCarloRuns = list(self.monteCarloRuns) if self.monteCarloRuns else []

        # snapshot spot classification (do NOT copy raw values; only classification state)
        res.processed  = [bool(getattr(s, "processed", False)) for s in self.validSpots]
        res.concordant = [
            (bool(getattr(s, "concordant", False)) if bool(getattr(s, "processed", False)) else None)
            for s in self.validSpots
        ]
        res.discordance = [getattr(s, "discordance", None) for s in self.validSpots]
        res.reverse = [bool(getattr(s, "reverseDiscordant", False)) for s in self.validSpots]

        self._results_by_mode[mode] = res
        self.activeConcordiaMode = mode

    def _clear_spot_classification_only(self):
        # IMPORTANT: do NOT call spot.clear() here (that might wipe imported values)
        for s in self.validSpots:
            if hasattr(s, "processed"):
                s.processed = False
            if hasattr(s, "concordant"):
                s.concordant = None
            if hasattr(s, "discordance"):
                s.discordance = None
            if hasattr(s, "reverseDiscordant"):
                s.reverseDiscordant = False

    def activate_mode(self, mode: ConcordiaMode):
        mode = ConcordiaMode.coerce(mode)
        self.activeConcordiaMode = mode

        res = self._results_by_mode.get(mode, None)

        if res is None:
            # no results for this mode => show "unprocessed" state for this mode
            self.optimalAge = None
            self.optimalAgeLowerBound = None
            self.optimalAgeUpperBound = None
            self.optimalAgeDValue = None
            self.optimalAgePValue = None
            self.optimalAgeNumberOfInvalidPoints = None
            self.optimalAgeScore = None
            self.monteCarloRuns = []
            self.peak_catalogue = []
            self.disc_cluster_labels = None
            self.disc_cluster_summary = []

            self._clear_spot_classification_only()

            # refresh the UI (tables + plots)
            if self.signals:
                self.signals.concordancyCalculated.emit()
                self.signals.optimalAgeCalculated.emit()
            return

        # restore sample-level values
        self.optimalAge = res.optimalAge
        self.optimalAgeLowerBound = res.optimalAgeLowerBound
        self.optimalAgeUpperBound = res.optimalAgeUpperBound
        self.optimalAgeDValue = res.optimalAgeDValue
        self.optimalAgePValue = res.optimalAgePValue
        self.optimalAgeNumberOfInvalidPoints = res.optimalAgeNumberOfInvalidPoints
        self.optimalAgeScore = res.optimalAgeScore

        self.peak_catalogue = deepcopy(res.peak_catalogue) if res.peak_catalogue else []
        self.disc_cluster_labels = deepcopy(res.disc_cluster_labels)
        self.disc_cluster_summary = deepcopy(res.disc_cluster_summary) if res.disc_cluster_summary else []
        self.monteCarloRuns = list(res.monteCarloRuns) if res.monteCarloRuns else []

        # restore spot classification
        self._clear_spot_classification_only()
        for i, s in enumerate(self.validSpots):
            if i >= len(res.processed) or not res.processed[i]:
                continue

            conc = res.concordant[i] if i < len(res.concordant) else None
            disc = res.discordance[i] if i < len(res.discordance) else None

            # spot.updateConcordance is your canonical setter
            if conc is not None and disc is not None:
                s.updateConcordance(bool(conc), disc)

            if i < len(res.reverse):
                s.reverseDiscordant = bool(res.reverse[i])

        # refresh the UI
        if self.signals:
            self.signals.concordancyCalculated.emit()
            self.signals.optimalAgeCalculated.emit()

    def mode_results(self, mode: ConcordiaMode) -> Optional[ModeResults]:
        mode = ConcordiaMode.coerce(mode)
        return self._results_by_mode.get(mode)

    def mode_peak_catalogue(self, mode: ConcordiaMode):
        r = self.mode_results(mode)
        return r.peak_catalogue if r is not None else []

class SampleSignals(QObject):
    summedKS = pyqtSignal(object)
    processingCleared = pyqtSignal()
    concordancyCalculated = pyqtSignal()
    monteCarloRunAdded = pyqtSignal()
    optimalAgeCalculated = pyqtSignal()
    skipped = pyqtSignal()
