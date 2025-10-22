from copy import deepcopy
from PyQt5.QtCore import pyqtSignal, QObject


class Sample:
    def __init__(self, id, name, spots):
        self.id = id
        self.name = name
        self.spots = spots

        self.peak_catalogue = []

        # Discordant clustering (experimental)
        self.disc_cluster_labels = None        # list[int] aligned to discordantSpots()
        self.disc_cluster_summary = []         # list[dict]: {cluster_id, n, median_ma}

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
        return [spot for spot in self.validSpots if spot.concordant]

    def discordantSpots(self):
        return [spot for spot in self.validSpots if not spot.concordant]

    def setSkipReason(self, reason):
        self.skip_reason = reason
        if self.signals:
            self.signals.skipped.emit()

    ##################
    ## Calculations ##
    ##################

    def startCalculation(self, calculationSettings):
        self.calculationSettings = calculationSettings

    def clearCalculation(self):
        self.optimalAge = None
        self.monteCarloRuns = []
        self.peak_catalogue = []
        self.disc_cluster_labels = None
        self.disc_cluster_summary = []
        for spot in self.spots:
            spot.clear()
        self.signals.processingCleared.emit()

    def updateConcordance(self, concordancy, discordances):
        for spot, concordant, discordance in zip(self.validSpots, concordancy, discordances):
            spot.updateConcordance(concordant, discordance)
        if self.signals:
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

        # ---- robust catalogue extraction (handles both old and new orders) ----
        idx_catalogue = None
        if len(args) >= 9 and isinstance(args[8], (list, tuple)):
            idx_catalogue = 8                         # new: (..., peak_str, catalogue)
        elif len(args) >= 8 and isinstance(args[7], (list, tuple)):
            idx_catalogue = 7                         # old: (..., catalogue)

        if idx_catalogue is not None:
            self.peak_catalogue = list(args[idx_catalogue] or [])
        else:
            # keep whatever was set by processing (or clear if you prefer)
            pass

        # ---- optional discordant clustering payload (shifted accordingly) ----
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
            self.signals.optimalAgeCalculated.emit()


    def createProcessingCopy(self):
        signals = self.signals
        self.signals = None
        copy = deepcopy(self)
        self.signals = signals
        return copy

    def getMonteCarloRuns(self):
        return self.monteCarloRuns

class SampleSignals(QObject):
    summedKS = pyqtSignal(object)  # (ages_ma, S_view, peaks[, ci_pairs, support])
    processingCleared = pyqtSignal()
    concordancyCalculated = pyqtSignal()
    monteCarloRunAdded = pyqtSignal()
    optimalAgeCalculated = pyqtSignal()
    skipped = pyqtSignal()

