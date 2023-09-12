import sys

import jax
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QDesktopWidget, QDialogButtonBox, QScrollArea
from suzirjax import utils
from suzirjax.gui import make_dialog

jax.config.update("jax_enable_x64", True)
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
    def __init__(self, const_fname=None):
        from suzirjax.gui import ApplicationWidget

        super().__init__()
        self.setWindowTitle('Suzirjax')
        self.setWindowIcon(QIcon(LOGO))
        self.widget = ApplicationWidget(self, const_name=const_fname)
        self.setCentralWidget(self.widget)
        self.statusbar = self.statusBar()
        self.statusbar.showMessage("Device: " + jax.devices()[0].device_kind)

        centre_point = QDesktopWidget().availableGeometry().center()
        geom = self.frameGeometry()
        geom.moveCenter(centre_point)
        self.move(geom.topLeft())

    def except_hook(self, exc_type, exc_value, exc_tb):
        import traceback
        tb = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        print(tb, file=sys.stderr)
        error_area = QScrollArea()
        error_area.setWidget(QLabel(tb))
        if not make_dialog("Unexpected Exception", error_area, parent=self,
                           buttons=QDialogButtonBox.Ignore | QDialogButtonBox.Abort).exec():
            app.quit()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ApplicationWindow(sys.argv[1] if len(sys.argv) == 2 else None)
    sys.excepthook = window.except_hook
    window.show()
    app.exec()
