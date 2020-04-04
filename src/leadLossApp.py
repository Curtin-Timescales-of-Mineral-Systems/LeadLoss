from apps.abstract.application import AbstractApplication
from apps.leadLoss.controller import LeadLossTabController
from utils import config


class LeadLossApplication(AbstractApplication):
    def getControllers(self):
        return [self.createController()]

    @staticmethod
    def getTitle():
        return config.LEAD_LOSS_TITLE

    @staticmethod
    def getVersion():
        return "0.1"

    @staticmethod
    def getIcon():
        return "../resources/icon.png"

    @staticmethod
    def createController():
        return LeadLossTabController()


if __name__ == '__main__':
    app = LeadLossApplication()
    app.createGUI()
