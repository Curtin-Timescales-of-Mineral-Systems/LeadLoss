import matplotlib.pyplot as plt

from view.axes.concordia.summaryWetherillConcordiaAxis import SummaryWetherillConcordiaAxis
from view.figures.abstractFigure import AbstractFigure


class SampleInputWetherillFigure(AbstractFigure):

    def __init__(self, sample):
        super().__init__()

        self.concordiaPlot = SummaryWetherillConcordiaAxis(self.fig.add_subplot(111), [sample])
        self.fig.subplots_adjust(hspace=0.7, wspace=0.4)

        sample.signals.processingCleared.connect(lambda s=sample: self._onSampleChanged(s))
        sample.signals.concordancyCalculated.connect(lambda s=sample: self._onSampleChanged(s))
        sample.signals.optimalAgeCalculated.connect(lambda s=sample: self._onSampleChanged(s))

        self.redraw()

    def _onSampleChanged(self, sample):
        self.concordiaPlot.refreshSample(sample)
        self.canvas.draw()
