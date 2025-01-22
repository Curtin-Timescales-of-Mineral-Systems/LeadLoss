import matplotlib.pyplot as plt

from view.figures.abstractFigure import AbstractFigure
from view.axes.concordia.summaryWetherillConcordiaAxis import SummaryWetherillConcordiaAxis

class SampleInputWetherillFigure(AbstractFigure):
    """
    Shows a single sample's 'unprocessed' spots in Wetherill space,
    using the same unclassified/concordant/discordant color logic.
    """

    def __init__(self, sample):
        super().__init__()

        # Create a subplot
        ax = self.fig.add_subplot(111)

        # Pass just [sample] so it only plots that one sample’s data
        self.concordiaPlot = SummaryWetherillConcordiaAxis(ax, [sample])

        # Hook up signals so that if the user re-runs processing or clears it,
        # we re-plot the sample as "unclassified" etc.
        sample.signals.processingCleared.connect(lambda s=sample: self._onSampleChanged(s))
        sample.signals.concordancyCalculated.connect(lambda s=sample: self._onSampleChanged(s))
        sample.signals.optimalAgeCalculated.connect(lambda s=sample: self._onSampleChanged(s))

        self.fig.tight_layout()
        self.canvas.draw()

    def _onSampleChanged(self, sample):
        # The Wetherill axis calls refreshSample
        self.concordiaPlot.refreshSample(sample)
        self.canvas.draw()
