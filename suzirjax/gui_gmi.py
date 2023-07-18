from collections import deque

from matplotlib.axes import Axes

from suzirjax import utils
from suzirjax.gui_helpers import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import numpy as np
import math


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
        data.on('gmi', lambda gmi: self.canvas.update_data(gmi, self.isVisible()), now=False)


class GMIHistoryCanvas(FigureCanvas):
    MAX_HIST = 2000

    def __init__(self, data):
        super().__init__()
        self.data: Connector = data
        self.ax: Axes = self.figure.subplots()
        self.figure.patch.set_facecolor('black')
        self.ax.set_facecolor('black')
        self.ax.tick_params(axis='x', labelcolor='white')
        self.ax.tick_params(axis='y', labelcolor='white')
        self.ax.spines['left'].set_color('white')
        self.ax.spines['bottom'].set_color('white')
        self.ax.set_title('GMI over time', color='white')
        self.ax.grid(axis='y')

        self.ax2: Axes = self.ax.twinx()
        self.ax2.set_facecolor('black')
        self.ax2.spines['right'].set_color('white')
        self.ax2.tick_params(axis='y', labelcolor='white')

        self.gmi = deque(maxlen=self.MAX_HIST)
        self.throughput = deque(maxlen=self.MAX_HIST)
        self.iteration_offset = 0
        self.im, = self.ax.plot(np.arange(len(self.gmi)), self.gmi, c='tab:blue', lw=1.6)
        self.im2, = self.ax2.plot(np.arange(len(self.throughput)), self.throughput, c='tab:orange', lw=1.6)
        self.ax.set_xlabel('Iterations', color='white')
        self.ax.set_ylabel('GMI (bit/2D symbol)', color='tab:blue')
        self.ax2.set_ylabel('Throughput (Gbps)', color='tab:orange')
        self.figure.tight_layout()

        add_right_clk_menu(
            self,
            ("Clear", self.clear),
            ("Save figure", self.save),
            ("Copy figure", self.copy_image),
            ("Save data", lambda: self.save_data(const=self.const)),
            parent=self,
        )
        # self.bg = self.figure.canvas.copy_from_bbox(self.figure.bbox)
        # self.ax.draw_artist(self.im)
        # self.figure.canvas.blit(self.figure.bbox)

    def clear(self):
        self.gmi.clear()
        self.throughput.clear()
        self.iteration_offset = 0
        self.im.set_data([0], [0])
        self.im2.set_data([0], [0])
        self.ax.set_xlim(xmin=0, xmax=1)
        self.ax.draw_artist(self.im)
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def render_graph(self):
        self.im.set_data(np.arange(len(self.gmi)) + self.iteration_offset, self.gmi)
        self.im2.set_data(np.arange(len(self.gmi)) + self.iteration_offset, self.throughput)
        self.ax.set_ylim(ymin=float(np.round(np.min(self.gmi) - 0.1, 1)), ymax=float(np.round(np.max(self.gmi) + 0.1, 1)))
        self.ax.set_xlim(xmin=self.iteration_offset, xmax=len(self.gmi) + self.iteration_offset)
        self.ax2.set_ylim(ymin=float(np.round(np.min(self.throughput) - 1, 10)), ymax=float(np.round(np.max(self.throughput) + 1, 10)))
        if self.ax.get_renderer_cache() is not None:
            self.ax.draw_artist(self.im)
            self.ax2.draw_artist(self.im2)
            self.figure.canvas.draw()
            self.figure.canvas.flush_events()

    def update_data(self, gmi, render=True):
        if np.isfinite(gmi):
            if len(self.gmi) >= self.MAX_HIST:
                self.iteration_offset += 1
            self.gmi.append(gmi)
            self.throughput.append(np.maximum(gmi, 0) * self.data.get('throughput_factor') / 1e9)  # Gbps
            if render:
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
