from matplotlib import pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

from suzirjax import utils
from suzirjax.gui_helpers import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import numpy as np


class GMIHistoryWindow(QDialog):
    def __init__(self, data: Connector, parent=None):
        super().__init__(parent)
        self.canvas = GMIHistoryCanvas(data)
        self.setLayout(VLayout(
            self.canvas,
            # make_button('Add', lambda _: self.canvas.update_data(0.13 + np.random.normal())),
            make_button('Reset', lambda _: self.canvas.clear()),
            parent=self, widget_class=None))
        self.setWindowTitle('Suzirjax GMI History')
        self.setWindowIcon(QIcon(utils.get_resource('logo.png')))
        data.on('gmi', lambda gmi: self.canvas.update_data(gmi), now=False)

    # def _update(self, gmi):
    #     self.canvas.gmi.append(gmi)


class GMIHistoryCanvas(FigureCanvasQTAgg):
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.ax = self.figure.subplots()
        self.figure.patch.set_facecolor('black')
        self.ax.set_facecolor('black')
        self.ax.tick_params(axis='x', labelcolor='white')
        self.ax.tick_params(axis='y', labelcolor='white')
        self.ax.spines['left'].set_color('white')
        self.ax.spines['bottom'].set_color('white')
        self.ax.set_title('GMI over time', color='white')
        self.ax.grid(axis='y')

        self.gmi = []
        self.im, = self.ax.plot(np.arange(len(self.gmi)), self.gmi, c='tab:blue', lw=1.6)
        self.ax.set_xlabel('Iterations', color='white')
        self.ax.set_ylabel('GMI (bit/2D symbol)', color='white')
        self.figure.tight_layout()
        # self.bg = self.figure.canvas.copy_from_bbox(self.figure.bbox)
        # self.ax.draw_artist(self.im)
        # self.figure.canvas.blit(self.figure.bbox)

    def clear(self):
        self.gmi = []
        # self.render_graph()

    def render_graph(self):
        self.im.set_data(np.arange(len(self.gmi)), self.gmi)
        self.ax.set_ylim([np.round(np.min(self.gmi) - 0.1, 1), np.round(np.max(self.gmi) + 0.1, 1)])
        self.ax.set_xlim([0, len(self.gmi)])
        self.ax.draw_artist(self.im)
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def update_data(self, gmi):
        if np.isfinite(gmi):
            self.gmi.append(gmi)
            self.render_graph()


if __name__ == '__main__':
    import sys

    class ApplicationWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.splash = GMIHistoryWindow(Connector())
            self.splash.show()
    app = QApplication(sys.argv)
    window = ApplicationWindow()
    window.show()
    app.exec()
