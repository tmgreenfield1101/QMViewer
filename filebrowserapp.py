from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel
from main_window import MainWindow

class FileBrowserApp(QWidget):
    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        open_folder_button = QPushButton('Open Folder', self)
        open_folder_button.clicked.connect(self.show_file_dialog)

        layout.addWidget(open_folder_button)

        self.setLayout(layout)
        self.setWindowTitle('File Browser App')

        # Set the fixed size of the window (width, height)
        self.setFixedSize(400, 200)

        # Apply a stylesheet to customize the appearance
        # self.setStyleSheet("""
        #     QWidget {
        #         background-color: lightblue;
        #     }
        # """)
        self.setStyleSheet(open("MacOS.qss", "r").read())

    def show_file_dialog(self):
        folder_path = QFileDialog.getExistingDirectory(self, 'Open Folder', 
                                                       '/space/tg286/iceland/reykjanes/qmigrate/may21/nov23_dyke_tight')
        if folder_path:
            print(f'Selected Folder: {folder_path}')
            self.open_new_window(folder_path)

    def open_new_window(self, folder_path):
        self.new_window = MainWindow(folder_path, self)
        self.new_window.show()
        self.close()