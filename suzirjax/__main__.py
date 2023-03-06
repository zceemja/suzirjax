import sys
import time

import jax
from PyQt5.QtCore import Qt, QThread
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QDialog, QDesktopWidget
from suzirjax import utils
from suzirjax.gui import ApplicationWidget

LOGO = utils.get_resource('logo.png')


#
# class LoadingThread(QThread):
#     def __init__(self):
#         QThread.__init__(self)
#
#     def run(self):
#         # This takes the most time
#         import jax
#         jax.devices()
#         self.terminate()
#
#
# class ApplicationWindow(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.splash = SplashWindow()
#
#         thread = LoadingThread()
#         thread.finished.connect(self._done)
#         thread.start()
#         self.splash.show()
#
#     def _done(self):
#         print('Done loading!')
#         import gui
#         self.setWindowTitle('OFC Demo')
#         self.setWindowIcon(QIcon(LOGO))
#         self.setCentralWidget(gui.ApplicationWidget(self))
#         self.show()
#         self.splash.close()
#
#
# class SplashWindow(QDialog):
#     def __init__(self, parent=None):
#         super().__init__(parent)
#         label = QLabel(self)
#         pixmap = QPixmap(LOGO)
#         label.setPixmap(pixmap)
#
#         text = QLabel(self)
#         text.setText('Loading JAX..')
#
#         text.setStyleSheet('font-size: 24px;')
#
#         self.setWindowTitle('Suzirjax')
#         self.setWindowIcon(QIcon(LOGO))
#         self.setGeometry(0, 0, pixmap.width(), pixmap.height())
#         self.resize(pixmap.width(), pixmap.height())
#         self.setWindowFlags(Qt.WindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint))
#
#         centre_point = QDesktopWidget().availableGeometry().center()
#         geom = self.frameGeometry()
#         geom.moveCenter(centre_point)
#         self.move(geom.topLeft())


class ApplicationWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Suzirjax')
        self.setWindowIcon(QIcon(LOGO))
        self.setCentralWidget(ApplicationWidget(self))
        self.statusbar = self.statusBar()
        self.statusbar.showMessage("Device: " + jax.devices()[0].device_kind)

        centre_point = QDesktopWidget().availableGeometry().center()
        geom = self.frameGeometry()
        geom.moveCenter(centre_point)
        self.move(geom.topLeft())


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ApplicationWindow()
    window.show()
    app.exec()
