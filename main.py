import sys
from PyQt5.QtWidgets import QApplication
from gui.main_window import MainWindow
from config.config import *

if __name__ == '__main__':
    if not check_restrictions():
        sys.exit(1)

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())