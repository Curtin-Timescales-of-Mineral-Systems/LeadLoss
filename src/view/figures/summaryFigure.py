import matplotlib.pyplot as plt

from view.axes.concordia.summaryConcordiaAxis import SummaryConcordiaAxis
from view.figures.abstractFigure import AbstractFigure


class SummaryFigure(AbstractFigure):

    def __init__(self, controller, samples):
        super().__init__()

        self.concordiaPlot = SummaryConcordiaAxis(plt.subplot(111), samples)
        plt.subplots_adjust(hspace=0.7, wspace=0.4)

        self.canvas.draw()