import queue
import time

import jax
from PyQt5.QtCore import QThread, QObject, pyqtSignal, QTimer
from jax import numpy as jnp
import numpy as np
from queue import Queue

from gui_helpers import Connector


def norm(x: jnp.ndarray) -> jnp.ndarray:
    p = jnp.sqrt(jnp.mean(x ** 2))
    return x / p


# Will be different for GPU/CPU capable machines
jax_arr_type = type(jnp.array([]))


class SimulationSignal(QObject):
    result = pyqtSignal(jax_arr_type, jax_arr_type)
    start = pyqtSignal(float)
    complete = pyqtSignal()


class SimulationWorker(QThread):
    HIST_LIM = 2
    HIST_BINS = 128

    def __init__(self, data: Connector, const: jnp.ndarray, parent=None):
        super().__init__()

        self.signal = SimulationSignal()
        self.data = data
        self._const = const
        self.const_new = False
        self.fps = 20
        self._request_queue = Queue(maxsize=1)
        self.paused = True
        self.timing = 1.5  # approx time for single simulation in ms

        self.progress_complete = 0
        self.progress_timer = QTimer()
        self.progress_timer.timeout.connect(self._update_progress)

    def _update_progress(self):
        self.progress_complete = min(self.progress_complete + 1, 100)
        self.signal.update.emit(self.progress_complete)

    def set_const(self, const):
        self._const = const
        self.const_new = True
        self.single()

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False
        self.single()

    def single(self):
        try:
            self._request_queue.put(True, block=False)
        except queue.Full:
            pass

    def _make_hist(self, rx):
        d, _, _ = jnp.histogram2d(
            rx[:, 0], rx[:, 1], bins=self.HIST_BINS, range=np.array([
                [-self.HIST_LIM, self.HIST_LIM], [-self.HIST_LIM, self.HIST_LIM]
            ]))
        # d[d < 1] = np.nan
        d /= jnp.log2(rx.shape[0]) * self.data['mod_points']
        return d.T

    def simulate(self):
        self.data.copy()
        const = self._const.copy()

        seq = (1 << (self.data['seq_length'])) // self.data['mod_points']  # Do random instead?
        tx = jnp.tile(const, (seq, 1))
        rx, snr = self.data['channel'].propagate(const, tx)
        const = jax.device_put(self.data['optimiser'].update(const, rx, snr))

        # Something is going terribly wrong
        if jnp.any(jnp.isnan(const)):
            const = self._const.copy()

        hist = self._make_hist(rx)
        self.signal.result.emit(const, hist)

        # Just in case of out if sync
        if not self.const_new:
            self._const = const
        self.const_new = False

    def run(self):
        while True:
            if not self._request_queue.get():
                break
            self.progress_complete = 0
            self.signal.start.emit(self.timing)
            t = time.time_ns()
            self.simulate()
            t = (time.time_ns() - t) / 1e9
            self.timing = t
            self.signal.complete.emit()
            interval = int((1 / self.fps - t) * 1e3)
            if not self.paused:
                if interval > 0:
                    QThread.msleep(interval)
                try:
                    self.single()
                except ValueError:
                    break  # Was done after sleep

    def close(self):
        self._request_queue.put(False)
        self._request_queue.task_done()
