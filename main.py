from PySide6.QtWidgets import QApplication
from filebrowserapp import FileBrowserApp

if __name__ == '__main__':
    if not QApplication.instance():
        app = QApplication([])
    else:
        app = QApplication.instance()
    window = FileBrowserApp()
    window.show()
    app.exec()