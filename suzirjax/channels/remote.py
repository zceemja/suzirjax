import os
import queue
import time

import jax
import socketio

from gui_helpers import *
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import *
from .channel import Channel
from jax import numpy as jnp
from typing import Tuple


class RemoteChannel(Channel):

    NAME = 'Remote'

    def __init__(self, *args):
        super().__init__(*args)
        self.sio = socketio.Client()
        self.pinger = QTimer()
        self.pinger.timeout.connect(self._ping)
        self.btn_connect = QPushButton('Connect', self.parent)
        self.btn_connect.clicked.connect(self._btn_connect)
        self.queue = queue.Queue(maxsize=1)

        self.data.on('linewidth', lambda val: self._send_config('linewidth', val * 1e3), now=False)
        self.data.on('snr', lambda val: self._send_config('snr', val), now=False)

        @self.sio.event
        def connect():
            self.data['conn'] = True
            self.sio.emit('config', {'linewidth': self.data['linewidth'] * 1e3, 'snr': self.data['snr']})
            self.btn_connect.setDisabled(False)
            self.btn_connect.setText('Disconnect')

        @self.sio.event
        def disconnect():
            self.data['conn'] = False
            self.btn_connect.setText('Connect')

        # @self.sio.event
        # def connect_error(data):
        #     make_dialog("Remote connection", QLabel("The connection failed!"), parent=self.parent, buttons=QDialogButtonBox.Close).exec()

        @self.sio.event
        def pong(data):
            t = int.from_bytes(data, 'little')
            t = (time.time_ns() - t) / 1e6
            self.data["ping"] = f'{t:.1f}ms' if t > 1. else f'{t * 1e3:.0f}µs'

        @self.sio.event
        def rx(rx_data, snr):
            self.queue.put((bytes_to_array(rx_data), bytes_to_array(snr)))

        @self.sio.on('*')
        def catch_all(event, data):
            pass

    def _send_config(self, name, val):
        if self.sio.connected:
            self.sio.emit('config', {name: val})

    def _btn_connect(self, _):
        if self.sio.connected:
            self.sio.disconnect()
        else:
            self.initialise()

    def _ping(self):
        if self.sio.connected:
            self.sio.emit('ping', time.time_ns().to_bytes(8, 'little'))
        else:
            self.data["ping"] = f'-'

    def initialise(self):
        if self.sio.connected:
            return
        if 'http_proxy' in os.environ:
            del os.environ['http_proxy']  # Please, stop.
        try:
            self.sio.connect('http://localhost:49922', wait=False)
            self.btn_connect.setDisabled(True)
            self.pinger.start(1000)
        except socketio.exceptions.ConnectionError as e:
            text = QLabel()
            if 'Max retries exceeded with url' in str(e):
                text.setText('Failed to establish a new connection')
            else:
                text.setText(str(e))
            make_dialog("Remote connection", text, parent=self.parent, buttons=QDialogButtonBox.Close).exec()
            self.btn_connect.setDisabled(False)

    def make_gui(self) -> QWidget:
        return FLayout(
            ('', self.btn_connect),
            ("Ping", make_label(bind=self.data.bind("ping", '-'))),
            ("Sampling Rate (GHz)", make_int_input(1, 60, 1, bind=self.data.bind("fs", 15))),
            ("Symbol Rate",  QLabel("90GHz")),
            ("Power (dBm)", make_float_input(-50, 20, 1, bind=self.data.bind("power", 0))),
            ("Additive Noise (dB)", make_float_input(-10, 300, 1, bind=self.data.bind("snr", 30))),
            # ("Additive Phase Noise (kHz)", make_float_input(1, 1e5, 10, bind=self.data.bind("linewidth", 100))),
            ("Route", make_combo(
                "B2B", "45km ULL", "NDFF CONNET", "NDFF Telehouse", "NDFF Powerhouse", "NDFF Reading",
                bind=self.data.bind("route", "B2B"))
             ),
        )

    def propagate(self, const: jnp.ndarray, rng_key: int, seq_len: int) -> Tuple[jnp.ndarray, float]:
        tx = self.get_tx(const, rng_key, seq_len)[1]
        if not self.sio.connected:
            return tx[0], 0
        self.sio.emit('tx', data=(array_to_bytes(const), rng_key, seq_len))
        rx, snr = self.queue.get(True)
        self.data['snr_est'] = f'{snr[0]:.1f} / {snr[1]:.1f}'
        return rx, snr[0]


def array_to_bytes(arr: jnp.ndarray) -> bytes:
    header = f"{str(arr.dtype)}&{','.join([str(a) for a in arr.shape])}".encode('ascii')
    header = len(header).to_bytes(4, 'little') + header
    return header + arr.ravel().tobytes()


def bytes_to_array(data: bytes) -> jnp.ndarray:
    header_len = int.from_bytes(data[:4], 'little')
    header = data[4:4+header_len].decode('ascii').split('&')
    dtype = header[0]
    shape = tuple(int(i) for i in header[1].split(','))
    return jnp.frombuffer(data[4+header_len:], dtype).reshape(shape)
