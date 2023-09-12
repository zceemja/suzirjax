import sys

import jax
from PyQt5.QtWidgets import *

from suzirjax.application import ApplicationWidget



if __name__ == '__main__':
    class ApplicationWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle('Suzirjax')
            self.setCentralWidget(ApplicationWidget(self))
            self.statusbar = self.statusBar()
            self.statusbar.showMessage("Device: " + jax.devices()[0].device_kind)

            centre_point = QDesktopWidget().availableGeometry().center()
            geom = self.frameGeometry()
            geom.moveCenter(centre_point)
            self.move(geom.topLeft())


    app = QApplication(sys.argv)
    window = ApplicationWindow()
    window.show()
    app.exec()
