from apps.abstract.application import AbstractApplication
from apps.unmix.controller import UnmixTabController
from utils import config


class UPbUnmixerApplication(AbstractApplication):

    def getControllers(self):
        return [self.createController()]

    @staticmethod
    def getTitle():
        return config.U_PB_UNMIXER_TITLE

    @staticmethod
    def getVersion():
        return "0.10"

    @staticmethod
    def getIcon():
        return "../resources/unmix_icon.ico"

    @staticmethod
    def createController():
        return UnmixTabController()


if __name__ == '__main__':
    app = UPbUnmixerApplication()
    app.createGUI()
