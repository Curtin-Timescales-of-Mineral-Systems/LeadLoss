from copy import deepcopy
from PyQt5.QtCore import pyqtSignal, QObject


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

    def clearCalculation(self):
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
    summedKS = pyqtSignal(object)
    processingCleared = pyqtSignal()
    concordancyCalculated = pyqtSignal()
    monteCarloRunAdded = pyqtSignal()
    optimalAgeCalculated = pyqtSignal()
    skipped = pyqtSignal()
