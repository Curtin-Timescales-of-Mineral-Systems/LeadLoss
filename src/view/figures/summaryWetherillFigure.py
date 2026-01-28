import matplotlib.pyplot as plt

from view.axes.concordia.summaryWetherillConcordiaAxis import SummaryWetherillConcordiaAxis
from view.figures.abstractFigure import AbstractFigure
from utils import resourceUtils


class SummaryWetherillFigure(AbstractFigure):

    def __init__(self, controller, samples):
        super().__init__()

        self.concordiaPlot = SummaryWetherillConcordiaAxis(self.fig.add_subplot(111), samples)
        self.fig.subplots_adjust(hspace=0.7, wspace=0.4)


        self.canvasHost.setObjectName("ConcordiaHost")
        self.set_watermark(resourceUtils.getResourcePath("zircon.png"))

        self.concordiaPlot = SummaryWetherillConcordiaAxis(self.fig.add_subplot(111), samples)

        controller.signals.samplesSelected.connect(self._onSamplesSelected)
        for sample in samples:
            sample.signals.processingCleared.connect(lambda s=sample: self._onSampleChanged(s))
            sample.signals.concordancyCalculated.connect(lambda s=sample: self._onSampleChanged(s))
            sample.signals.optimalAgeCalculated.connect(lambda s=sample: self._onSampleChanged(s))

        self.redraw()

    def _onSampleChanged(self, sample):
        self.concordiaPlot.refreshSample(sample)
        self.canvas.draw()

    def _onSamplesSelected(self, selectedSamples, _unselectedSamples):
        # Keep it simple for now; selection styling can be added later if you want parity.
        for sample in selectedSamples:
            self.concordiaPlot.refreshSample(sample)
        self.canvas.draw()
