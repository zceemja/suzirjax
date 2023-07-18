import sys

import jax

from suzirjax.gui_gmi import GMIHistoryWindow
from suzirjax.gui_const import ConstellationCanvas
from suzirjax.gui_helpers import *
from suzirjax.channels import CHANNELS
from suzirjax.modulation import MODULATIONS, get_modulation, relabel
from suzirjax.optimiser import OPTIMISERS
from suzirjax.simulation import Simulation
from suzirjax.utils import register_cmaps

from PyQt5.QtCore import QT_VERSION_STR
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import matplotlib
import numpy as np
from jax import numpy as jnp

matplotlib.use('Qt5Agg')
register_cmaps()


class ApplicationWidget(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.data = Connector()
        self.data.set('sim_running', False)  # Play/pause simulation
        self.data.set('channel_type', "AWGN")
        self.data.set('mod_points', 32)
        self.data.set('mod_name', "Random")
        self.data.set('throughput_factor', 1)

        self.ssfm_data = Connector()

        layout = QVBoxLayout()
        self.setLayout(layout)
        self.channels = {ch.NAME: ch(self) for ch in CHANNELS}
        self.optimisers = {opt.NAME: opt(self.data) for opt in OPTIMISERS}

        self.quitting = False
        self.const_canvas = ConstellationCanvas(self.data)

        ## Simulation
        init_const = np.random.rand(32, 2) * 2 - 1
        self.sim = Simulation(self.data, init_const, parent=self)

        self.sim.signal.result.connect(self.const_canvas.update_data)

        # This lets GUI thread to catch up with events before running next sim
        self.sim.signal.result.connect(lambda _: self.sim.cont())

        self.sim_btn = make_button("", lambda _: self.sim.toggle_pause(), self)
        self.data.on('sim_running', lambda r: self.sim_btn.setText('Stop' if r else 'Start'))
        self.data.on('channel', lambda _, c: c.terminate(), now=False, call_on_none=False)
        self.gmi_hist = GMIHistoryWindow(self.data, self)
        self.data['show_rx'] = True

        self.control_widget = VLayout(
            FLayout(
                ("Seq. Length (2^n)", make_int_input(11, 17, bind=self.data.bind("seq_length", 11))),
                ("Channel", make_combo_dict(
                    self.channels, bind=self.data.bind("channel", self.channels[CHANNELS[0].NAME]))),
                ("Optimiser", make_combo_dict(
                    self.optimisers, bind=self.data.bind("optimiser", self.optimisers[OPTIMISERS[0].NAME]))),
                ("GMI History", make_button('Show', lambda _: self.gmi_hist.show())),
                # ("", make_button('Relabel', self._relabel)),
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
            self.sim_btn,
            parent=self
        )
        self.control_widget.setMaximumWidth(250)
        self.data.on("channel", lambda ch: ch.initialise())
        QApplication.instance().aboutToQuit.connect(self.before_quit)

        self.shortcuts = [
            QShortcut(QKeySequence("Ctrl+Space"), self),
            QShortcut(QKeySequence("Escape"), self),
        ]
        self.shortcuts[0].activated.connect(self.sim.single)
        self.shortcuts[1].activated.connect(self.sim.pause)

        layout.addWidget(HLayout(
            self.control_widget, self.const_canvas,
            parent=self
        ))
        # layout.addWidget(self.progress)
        if isinstance(parent, QMainWindow):
            parent.setMenuBar(make_menubar({
                "&File": [
                    ("Save &Constellation", self.const_canvas.save),
                    ("Save &GMI History", self.gmi_hist.canvas.save),
                    ("&Exit", QApplication.instance().quit),
                ],
                "&Help": [
                    ("&About", self.about_dialog),
                ]
            }, parent=parent))
        self.sim.start()

    # def _relabel(self, _):
    #     const = relabel(self.sim.const)
    #     # const = self.sim.const
    #     const /= np.sqrt((abs(const) ** 2).mean() * 2)
    #     self.sim.set_const(const)
    #     self.const_canvas.update_data(const, None)

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

    def before_quit(self):
        self.quitting = True
        self.data['channel'].terminate()
        self.sim.close()
        self.sim.wait()
        print("Shutting down")

    def about_dialog(self, _):
        make_dialog("About", VLayout(
            "Suzirjax",
            "Python: " + sys.version.split('\n')[0],
            "JAX: " + jax.__version__,
            "QT: " + QT_VERSION_STR,
            "Matplotlib: " + matplotlib.__version__,
        ), buttons=QDialogButtonBox.Close, parent=self).exec()

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
            self.const_canvas.update_data(self.sim.const, None)


if __name__ == '__main__':
    class ApplicationWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle('Suzirjax')
            self.setCentralWidget(ApplicationWidget(self))
            self.setWindowTitle("Device: " + jax.devices()[0].device_kind)

            centre_point = QDesktopWidget().availableGeometry().center()
            geom = self.frameGeometry()
            geom.moveCenter(centre_point)
            self.move(geom.topLeft())


    app = QApplication(sys.argv)
    window = ApplicationWindow()
    window.show()
    app.exec()
