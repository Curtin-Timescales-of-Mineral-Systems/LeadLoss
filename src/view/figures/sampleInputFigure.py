from utils import resourceUtils 
from view.axes.concordia.summaryConcordiaAxis import SummaryConcordiaAxis
from view.figures.abstractFigure import AbstractFigure


class SampleInputFigure(AbstractFigure):

    def __init__(self, sample):
        super().__init__()

        self.concordiaPlot = SummaryConcordiaAxis(self.fig.add_subplot(111), [sample])
        self.fig.subplots_adjust(hspace=0.7, wspace=0.4)

        self.canvasHost.setObjectName("ConcordiaHost")
        self.set_watermark(resourceUtils.getResourcePath("concordia_bg.png"))

        sample.signals.processingCleared.connect(lambda s=sample: self._onSampleChanged(s))
        sample.signals.concordancyCalculated.connect(lambda s=sample: self._onSampleChanged(s))
        sample.signals.optimalAgeCalculated.connect(lambda s=sample: self._onSampleChanged(s))

        self.redraw()

    def _onSampleChanged(self, sample):
        self.concordiaPlot.refreshSample(sample)
        self.canvas.draw()