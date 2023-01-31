import sys
import time

from PyQt5.QtWidgets import *
from jax import numpy as jnp
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib import pyplot as plt
import numpy as np
from gui_helpers import *
from channels import CHANNELS
from modulation import MODULATIONS, get_modulation
from optimiser import OPTIMISERS
from utils import register_cmaps

register_cmaps()


class ApplicationWidget(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.data = Connector()
        self.data.set('running', True)  # Play/pause simulation
        self.data.set('channel_type', "AWGN")
        self.data.set('mod_points', 32)
        self.data.set('mod_name', "Random")

        self.ssfm_data = Connector()

        layout = QHBoxLayout()
        self.setLayout(layout)
        self.channels = {ch.NAME: ch() for ch in CHANNELS}
        self.optimisers = {opt.NAME: opt(self.data) for opt in OPTIMISERS}

        self.control_widget = VLayout(
            FLayout(
                ("Seq. Length (2^n)", make_int_input(4, 20, bind=self.data.bind("seq_length", 10))),
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
            make_button("Stop", self.start_stop, self),
            parent=self
        )
        layout.addWidget(self.control_widget)

        self.const_canvas = ConstellationCanvas(self.data)
        layout.addWidget(self.const_canvas)

    def start_stop(self, btn: QPushButton):
        btn.setText("Start" if self.data['running'] else "Stop")
        self.data['running'] = not self.data['running']

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
                self.const_canvas.const = a
            else:
                a = get_modulation(str(dlgc['mod_points']) + dlgc['mod_name'])
                a = np.array([a.real, a.imag])
                self.const_canvas.const = a.T
            self.data['mod_name'] = dlgc['mod_name']
            self.data['mod_points'] = dlgc['mod_points']


class ConstellationCanvas(FigureCanvasQTAgg):
    LIM = 1 + 2 ** -.5
    TEXT_OFFSET = 0.05

    def __init__(self, data: Connector):
        super().__init__(plt.Figure())
        self.data = data
        self.ax = self.figure.subplots()
        self.ax.set_ylim(ymin=-self.LIM, ymax=self.LIM)
        self.ax.set_xlim(xmin=-self.LIM, xmax=self.LIM)
        self.ax.set_aspect('equal', 'box')
        self.const = np.random.rand(32, 2) * 2 - 1

        tx, rx = self._simulate()
        self.im = self.ax.imshow(
            self._make_hist(rx), interpolation='bicubic', extent=[-self.LIM, self.LIM, -self.LIM, self.LIM],
            origin='lower', cmap='sillekens')  # kindlmann, sillekens
        self.const_plt, = self.ax.plot(self.const[:, 0], self.const[:, 1], '.', c='magenta')
        self.const_text = []
        self.data.on('mod_points', self._update_points)
        self.data.on('show_rx', lambda v: (self.im.set_visible(v), self.figure.canvas.draw()))
        self.data.on('show_c', lambda v: (self.const_plt.set_visible(v), self.figure.canvas.draw()))
        self.data.on('show_bmap', lambda v: ([t.set_visible(v) for t in self.const_text], self.figure.canvas.draw()))

        self.anim = FuncAnimation(self.figure, self._update, init_func=self._init, interval=20, blit=False)
        data.on('running', self.run)

    def _update_points(self, m):
        format_str = f'{{0:0{int(np.log2(m))}b}}'
        self.bit_text = [format_str.format(i) for i in range(m)]
        if len(self.const_text) > 0:
            for text in self.const_text:
                text.remove()
        self.const_text = []
        for i, text in enumerate(self.bit_text):
            self.const_text.append(self.ax.text(
                self.const[i, 0] + self.TEXT_OFFSET, self.const[i, 1] + self.TEXT_OFFSET, text, c='magenta'))

    def _make_hist(self, rx):
        d, _, _ = jnp.histogram2d(
            rx[:, 0], rx[:, 1], bins=128, range=np.array([[-self.LIM, self.LIM], [-self.LIM, self.LIM]]))
        # d[d < 1] = np.nan
        d /= jnp.log2(rx.shape[0]) * self.data['mod_points']
        return d.T

    def run(self, val):
        self.anim.resume() if val else self.anim.pause()

    def _init(self):
        return self.const_plt, self.im

    def _simulate(self):
        seq = (1 << (self.data['seq_length'])) // self.data['mod_points']
        tx = jnp.tile(self.const, (seq, 1))
        rx, snr = self.data['channel'].propagate(tx)
        self.const = self.data['optimiser'].update(self.const, rx, snr)
        return tx, rx

    def _update(self, i):
        tx, rx = self._simulate()
        self.im.set_data(self._make_hist(rx))
        self.const_plt.set_data(self.const[:, 0], self.const[:, 1])
        for i, text in enumerate(self.const_text):
            if jnp.all(jnp.isfinite(self.const[i])):
                text.set_x(self.const[i, 0] + self.TEXT_OFFSET)
                text.set_y(self.const[i, 1] + self.TEXT_OFFSET)
        return self.const_plt, self.im


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
