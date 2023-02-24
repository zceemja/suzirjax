import math
import sys
import time

from PyQt5.QtCore import QTimer

from gui_helpers import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
from channels import CHANNELS
from modulation import MODULATIONS, get_modulation
from optimiser import OPTIMISERS
from simulation import SimulationWorker
from utils import register_cmaps

matplotlib.use('Qt5Agg')
register_cmaps()


class ApplicationWidget(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.data = Connector()
        self.data.set('running', False)  # Play/pause simulation
        self.data.set('channel_type', "AWGN")
        self.data.set('mod_points', 32)
        self.data.set('mod_name', "Random")

        self.ssfm_data = Connector()

        layout = QVBoxLayout()
        self.setLayout(layout)
        self.channels = {ch.NAME: ch(self) for ch in CHANNELS}
        self.optimisers = {opt.NAME: opt(self.data) for opt in OPTIMISERS}

        # self.key, key0 = jax.random.split(jax.random.PRNGKey(time.time_ns()), 2)
        self.quitting = False
        self.const_canvas = ConstellationCanvas(self.data)
        self.progress = QProgressBar(self)
        self.progress_timer = QTimer()
        self.sim: SimulationWorker = None

        self.start_stop_btn = make_button("Start", self.start_stop, self)
        self.control_widget = VLayout(
            FLayout(
                ("Seq. Length (2^n)", make_int_input(11, 14, bind=self.data.bind("seq_length", 11))),
                ("Channel", make_combo_dict(
                    self.channels, bind=self.data.bind("channel",  self.channels[CHANNELS[0].NAME]))),
                ("Optimiser", make_combo_dict(
                    self.optimisers, bind=self.data.bind("optimiser",  self.optimisers[OPTIMISERS[0].NAME]))),
                ("Show received", make_checkbox(bind=self.data.bind("show_rx", True))),
                ("Show constellation", make_checkbox(bind=self.data.bind("show_c", True))),
                ("Show bitmap", make_checkbox(bind=self.data.bind("show_bmap", True))),
            ),
            *[
                make_hidden_group(channel.make_gui(), bind=self.data.bind("channel"), bind_value=channel)[0]
                for channel in self.channels.values()
            ],
            *[
                make_hidden_group(opt.make_gui(), bind=self.data.bind("optimiser"), bind_value=opt)[0]
                for opt in self.optimisers.values()
            ],
            make_button("New Constellation", self.new_constellation, self),
            make_button("Single", lambda _: self.sim.single(), self),
            self.start_stop_btn,
            parent=self
        )
        self.data.on("channel", lambda ch: ch.initialise())
        QApplication.instance().aboutToQuit.connect(self.before_quit)

        self.initialise_sim()
        self.shortcuts = [
            QShortcut(QKeySequence("Ctrl+Space"), self),
            QShortcut(QKeySequence("Escape"), self),
        ]
        self.shortcuts[0].activated.connect(self.sim.single)
        self.shortcuts[1].activated.connect(self.stop)

        layout.addWidget(HLayout(
            self.control_widget, self.const_canvas, parent=self
        ))
        layout.addWidget(self.progress)
        self.sim.single()

    def _progress_stop(self):
        self.progress_timer.stop()
        self.progress.setTextVisible(False)
        self.progress.setValue(0)

    def _progress_update(self):
        self.progress.setValue(self.progress.value() + 1)
        if self.progress.value() >= self.progress.maximum():
            self.progress_timer.stop()

    def _progress_start(self, f):
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        interval = int(1 / 60 * 1000)
        self.progress_timer.start(interval)
        self.progress.setMaximum(int(f * 1000 / interval))

    def initialise_sim(self):
        if self.quitting:
            return
        if self.sim is not None:
            self.sim.wait()
            init_const = self.sim._const.copy()
        else:
            init_const = np.random.rand(32, 2) * 2 - 1
        self.sim = SimulationWorker(self.data, init_const, parent=self)
        self.sim.signal.result.connect(self.const_canvas.update_data)
        self.sim.signal.start.connect(self._progress_start)
        self.sim.signal.complete.connect(self._progress_stop)
        self.progress_timer.timeout.connect(self._progress_update)
        self.sim.finished.connect(self.initialise_sim)
        self.sim.start()

    def before_quit(self):
        self.quitting = True
        self.sim.close()
        self.sim.quit()
        self.sim.wait()

    def stop(self):
        self.data['running'] = False
        self.sim.close()
        self.start_stop_btn.setText("Start")

    def start_stop(self, btn: QPushButton):
        btn.setText("Start" if self.data['running'] else "Stop")
        self.data['running'] = not self.data['running']
        self.sim.resume() if self.data['running'] else self.sim.pause()

    def new_constellation(self, _):
        dlgc = Connector()
        dlg = make_dialog("New Constellation", FLayout(
            ("Modulation", make_combo(
                *MODULATIONS, "Random",
                bind=dlgc.bind("mod_name", self.data['mod_name']))
             ),
            ("Points", make_combo_dict(
                {str(n): int(n) for n in 2 ** np.arange(2, 11)},
                bind=dlgc.bind("mod_points", self.data['mod_points']))
             ),
        ), parent=self)
        if dlg.exec():
            if dlgc['mod_name'] == 'Random':
                a = np.random.rand(dlgc['mod_points'], 2) * 2 - 1
                self.sim.set_const(a)
            else:
                a = get_modulation(str(dlgc['mod_points']) + dlgc['mod_name'])
                a = np.array([a.real, a.imag])
                self.sim.set_const(a.T)
            self.data['mod_name'] = dlgc['mod_name']
            self.data['mod_points'] = dlgc['mod_points']


class ConstellationCanvas(FigureCanvasQTAgg):
    LIM = 1 + 2 ** -.5  # Should match simulation.py HIST_LIM
    TEXT_OFFSET = 0.05

    def __init__(self, data: Connector):
        super().__init__()
        self.data = data
        self.ax = self.figure.subplots()
        self.ax.set_ylim(ymin=-self.LIM, ymax=self.LIM)
        self.ax.set_xlim(xmin=-self.LIM, xmax=self.LIM)
        self.ax.set_aspect('equal', 'box')

        self.im = None
        self.const_plt = None
        self.last_const = None

        self.const_text = []

        self.data.on('show_rx', lambda v: self.set_comp_visible(self.im, v))
        self.data.on('show_c', lambda v: self.set_comp_visible(self.const_plt, v))
        self.data.on('show_bmap', lambda v: ([t.set_visible(v) for t in self.const_text], self.figure.canvas.draw()))

    def set_comp_visible(self, comp, v):
        if comp is not None:
            comp.set_visible(v)
            self.figure.canvas.draw()

    def _update_points(self):
        m = self.last_const.shape[0]
        format_str = f'{{0:0{int(np.log2(m))}b}}'
        self.bit_text = [format_str.format(i) for i in range(m)]
        if len(self.const_text) > 0:
            for text in self.const_text:
                text.remove()
        self.const_text = []
        for i, text in enumerate(self.bit_text):
            self.const_text.append(self.ax.text(
                self.last_const[i, 0] + self.TEXT_OFFSET, self.last_const[i, 1] + self.TEXT_OFFSET, text, c='magenta'))


    def run(self, val):
        ...
        # self.anim.resume() if val else self.anim.pause()

    def update_data(self, const, hist):
        if self.im is None:
            self.im = self.ax.imshow(
                hist, interpolation='bicubic', extent=[-self.LIM, self.LIM, -self.LIM, self.LIM],
                origin='lower', cmap='sillekens')  # kindlmann, sillekens
            self.im.set_visible(self.data['show_rx'])
        else:
            self.im.set_data(hist)
        if self.last_const is None or const.shape[0] != self.last_const.shape[0]:
            self.last_const = const
            self._update_points()

        if self.const_plt is None:
            self.const_plt, = self.ax.plot(const[:, 0], const[:, 1], '.', c='magenta')
            self.const_plt.set_visible(self.data['show_c'])
        else:
            self.const_plt.set_data(const[:, 0], const[:, 1])

        for i, text in enumerate(self.const_text):
            if np.all(np.isfinite(const[i])):
                text.set_x(const[i, 0] + self.TEXT_OFFSET)
                text.set_y(const[i, 1] + self.TEXT_OFFSET)
        self.figure.canvas.draw()

    # def _update(self, i):
    #     tx, rx = self._simulate()
    #     self.im.set_data(self._make_hist(rx))
    #     self.const_plt.set_data(self.const[:, 0], self.const[:, 1])
    #     for i, text in enumerate(self.const_text):
    #         if jnp.all(jnp.isfinite(self.const[i])):
    #             text.set_x(self.const[i, 0] + self.TEXT_OFFSET)
    #             text.set_y(self.const[i, 1] + self.TEXT_OFFSET)
    #     return self.const_plt, self.im


if __name__ == '__main__':
    class ApplicationWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle('Suzirjax')
            self.setCentralWidget(ApplicationWidget(self))

            centre_point = QDesktopWidget().availableGeometry().center()
            geom = self.frameGeometry()
            geom.moveCenter(centre_point)
            self.move(geom.topLeft())


    app = QApplication(sys.argv)
    window = ApplicationWindow()
    window.show()
    app.exec()
