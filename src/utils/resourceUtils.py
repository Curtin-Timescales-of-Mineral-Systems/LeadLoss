import os
import sys
import pathlib

def getResourcePath(relativePath):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS + "/resources/"
    except Exception:
        import __main__
        base_path = str(pathlib.Path(__main__.__file__).parent.absolute()) + "/../resources/"
    return os.path.join(base_path, relativePath)
