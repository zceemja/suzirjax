"""
Window that shows NDFF (National Dark Fibre Facility, https://www.ndff.ac.uk/)
map with optical power at each point, and allowing reconfiguration optical path.
"""

from suzirjax import utils
from suzirjax.gui.helpers import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *


class NDFFWindow(QDialog):
    LOCATIONS = "CONNET", "Telehouse", "Powergate", "Reading", "Southampton"

    def __init__(self, data: Connector, parent=None):
        super().__init__(parent)
        self.data = data
        self.label = QLabel(self)
        self.MAP_IMGS = [
            QPixmap(utils.get_resource(f'map{i}.png')) for i in range(5)
        ]
        self.label.setPixmap(self.MAP_IMGS[0])

        power_labels = []
        for i, v in enumerate(self.LOCATIONS):
            power_labels.append((v + ' in: ',
                                 make_label(bind=self.data.bind(f'ndff_pow{i}_in', -50), formatting='{:.2f}dBm')))
            power_labels.append((v + ' out: ',
                                 make_label(bind=self.data.bind(f'ndff_pow{i}_out', -50), formatting='{:.2f}dBm')))

        layout = HLayout(
            self.label,
            VLayout(
                make_radio_buttons(*[(v, i) for i, v in enumerate(self.LOCATIONS)], bind=self.data.bind('ndff_route', 0)),
                FLayout(*power_labels), parent=self)
            , parent=self, widget_class=None)
        self.setLayout(layout)

        self.data.on('ndff_route', self.update_route)
        self.setWindowTitle('Suzirjax NDFF')
        self.setWindowIcon(QIcon(utils.get_resource('logo.png')))

    def update_route(self, i):
        self.label.setPixmap(self.MAP_IMGS[i])


if __name__ == '__main__':
    import sys

    class ApplicationWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.splash = NDFFWindow(Connector())
            self.splash.show()
    app = QApplication(sys.argv)
    window = ApplicationWindow()
    window.show()
    app.exec()
