import matplotlib.pyplot as plt

from view.axes.concordia.summaryWetherillConcordiaAxis import SummaryWetherillConcordiaAxis
from view.figures.abstractFigure import AbstractFigure

class SummaryWetherillFigure(AbstractFigure):
    def __init__(self, controller, samples):
        super().__init__()

        ax = self.fig.add_subplot(111)

        self.concordiaPlot = SummaryWetherillConcordiaAxis(ax, samples)

        controller.signals.samplesSelected.connect(self._onSamplesSelected)

        for sample in samples:
            sample.signals.processingCleared.connect(
                lambda s=sample: self._onSampleChanged(s))
            sample.signals.concordancyCalculated.connect(
                lambda s=sample: self._onSampleChanged(s))
            sample.signals.optimalAgeCalculated.connect(
                lambda s=sample: self._onSampleChanged(s))

        self.canvas.draw()

    def _onSampleChanged(self, sample):
        self.concordiaPlot.refreshSample(sample)
        self.canvas.draw()

    def _onSamplesSelected(self, selectedSamples, unselectedSamples):
        self.concordiaPlot.selectSamples(selectedSamples, unselectedSamples)
        self.canvas.draw()
