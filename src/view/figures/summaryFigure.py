import matplotlib.pyplot as plt

from view.axes.concordia.summaryConcordiaAxis import SummaryConcordiaAxis
from view.figures.abstractFigure import AbstractFigure


class SummaryFigure(AbstractFigure):

    def __init__(self, controller, samples):
        super().__init__()

        self.concordiaPlot = SummaryConcordiaAxis(plt.subplot(111), samples)
        plt.subplots_adjust(hspace=0.7, wspace=0.4)

        controller.signals.samplesSelected.connect(self._onSamplesSelected)
        for sample in samples:
            sample.signals.processingCleared.connect(lambda s=sample: self._onSampleChanged(s))
            sample.signals.concordancyCalculated.connect(lambda s=sample: self._onSampleChanged(s))
            sample.signals.optimalAgeCalculated.connect(lambda s=sample: self._onSampleChanged(s))

        self.canvas.draw()

    def _onSampleChanged(self, sample):
        self.concordiaPlot.refreshSample(sample)
        self.canvas.draw()

    def _onSamplesSelected(self, selectedSamples, unselectedSamples):
        self.concordiaPlot.selectSamples(selectedSamples, unselectedSamples)
        self.canvas.draw()