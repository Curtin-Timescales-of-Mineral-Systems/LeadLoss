from apps.abstract.application import AbstractApplication
from leadLossApp import LeadLossApplication
from unmixApp import UPbUnmixerApplication

class MultiApplication(AbstractApplication):

    applications = [
        UPbUnmixerApplication,
        LeadLossApplication
    ]

    def __init__(self):
        super().__init__("All applications")

    def getTitle(self):
        return " & ".join([application.getTitle() for application in self.applications])

    def getVersion(self):
        return " ".join([application.getVersion() for application in self.applications])

    def getIcon(self):
        return "../resources/icon.png"

    def getControllers(self):
        return [application.createController() for application in self.applications]


if __name__ == '__main__':
    app = MultiApplication()
    app.createGUI()
