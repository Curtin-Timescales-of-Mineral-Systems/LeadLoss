from PyQt5.QtWidgets import QWidget, QStyle

class Icons:
    __style = None

    @staticmethod
    def get(icon):
        if Icons.__style is None:
            Icons.__style = QWidget().style()
        style = Icons.__style
        return style.standardIcon(icon)

    @staticmethod
    def help():
        return Icons.get(QStyle.SP_MessageBoxQuestion)

    @staticmethod
    def importCSV():
        return Icons.get(QStyle.SP_DialogOpenButton)

    @staticmethod
    def process():
        return Icons.get(QStyle.SP_ArrowForward)

    @staticmethod
    def warning():
        return Icons.get(QStyle.SP_MessageBoxWarning)

    @staticmethod
    def export():
        return Icons.get(QStyle.SP_DialogSaveButton)

    @staticmethod
    def cancel():
        return Icons.get(QStyle.SP_DialogCancelButton)

    @staticmethod
    def close():
        return Icons.get(QStyle.SP_DialogCloseButton)