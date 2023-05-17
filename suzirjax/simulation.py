import threading
import time

import jax
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QWidget
from jax import numpy as jnp
import numpy as np

from suzirjax.gui_helpers import Connector


def norm(x: jnp.ndarray) -> jnp.ndarray:
    p = jnp.sqrt(jnp.mean(x ** 2))
    return x / p


# Will be different for GPU/CPU capable machines
jax_arr_type = type(jnp.array([]))


class SimulationSignal(QObject):
    result = pyqtSignal(jax_arr_type, jax_arr_type)
    start = pyqtSignal(float)
    complete = pyqtSignal()


class Simulation(QWidget):
    HIST_LIM = 1.69
    HIST_BINS = 128

    def __init__(self, data: Connector, const: jnp.ndarray, parent=None):
        super().__init__(parent)
        self.signal = SimulationSignal()
        self._running = True
        self._const = const
        self.data = data
        self.rng_key = 312
        self._paused = threading.Event()
        if not self.data['sim_running']:
            self._paused.clear()
        self._thread = threading.Thread(name='simulation_loop', target=self._loop)
        self._single = False

    @property
    def const(self):
        return self._const

    def start(self):
        self._thread.start()

    def set_const(self, const):
        self._const = const

    def pause(self):
        self._paused.clear()
        self.data['sim_running'] = False

    def resume(self):
        self._paused.set()
        self.data['sim_running'] = True

    def cont(self):
        if self.data['sim_running']:
            self._paused.set()

    def toggle_pause(self):
        self.pause() if self._paused.isSet() else self.resume()

    def single(self):
        self._single = True
        self._paused.set()
        self.data['sim_running'] = False

    def wait(self, timeout=None):
        self._thread.join(timeout)

    def close(self):
        self._running = False
        self._paused.set()

    def _make_hist(self, rx):
        d, _, _ = jnp.histogram2d(
            rx[:, 0], rx[:, 1], bins=self.HIST_BINS, range=np.array([
                [-self.HIST_LIM, self.HIST_LIM], [-self.HIST_LIM, self.HIST_LIM]
            ]))
        return d.T

    def _simulate(self):
        data = self.data.data.copy()
        const = self._const.copy()

        rx, snr = data['channel'].propagate(const, self.rng_key, 1 << data['seq_length'])
        tx, _ = data['channel'].get_tx(const, self.rng_key, 1 << data['seq_length'])
        const = jax.device_put(data['optimiser'].update(const, rx, snr, tx[0]))
        self._const = const.copy()

        const /= np.sqrt((abs(const)**2).mean() * 2)
        rx /= np.sqrt((abs(rx) ** 2).mean() * 2)
        hist = self._make_hist(rx)
        self._paused.clear()
        self.signal.result.emit(const, hist)

    def _loop(self):
        self._paused.wait()
        while self._running:
            self._simulate()
            if self._single:
                self._single = False
            self._paused.wait()
        print("Closing simulation loop")
