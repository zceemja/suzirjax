import os
import queue
import time

import socketio

from suzirjax.gui import *
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import *
from .channel import Channel, register_channel
from jax import numpy as jnp
from typing import Tuple

from suzirjax.gui.ndff import NDFFWindow


@register_channel
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
        self.ndff_window = NDFFWindow(self.data)

        self.data.on('linewidth', lambda val: self._send_config('linewidth', val * 1e3), now=False)
        self.data.on('snr', lambda val: self._send_config('snr', val), now=False)
        self.data.on('route', lambda val: self._send_config('mode', val), now=False)
        self.data.on('power', lambda val: self._send_config('power', val), now=False)
        self.data.on('fb', lambda val: self._send_config('fb', val * 1e9), now=False)
        self.data.on('impairments', lambda val: self._send_config('impairments', val), now=False)
        self.data.on('impairments_snr', lambda val: self._send_config('impairments_snr', val), now=False)
        self.data.on('impairments_linewidth', lambda val: self._send_config('impairments_linewidth', val), now=False)

        @self.sio.event
        def connect():
            print("Socket connected")
            self.data['conn'] = True
            self.sio.emit('config', {'fb': self.data['fb'] * 1e9})
            self.btn_connect.setDisabled(False)
            self.btn_connect.setText('Disconnect')

        @self.sio.event
        def disconnect():
            print("Socket disconnected")
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

        @self.sio.event
        def status(data):
            self.data['power_mon_in'] = data['power_in']
            self.data['power_mon_out'] = data['power_out']
            self.data['ndff_pow0_in'] = data['ndff_pow0_in']
            self.data['ndff_pow0_out'] = data['ndff_pow0_out']
            self.data['ndff_pow1_in'] = data['ndff_pow1_in']
            self.data['ndff_pow1_out'] = data['ndff_pow1_out']
            self.data['route'] = data['mode']
            self.data['ndff_route'] = data['ndff_route']
            self.data['fibre_len'] = data['distance'] / 1e3
            self.data['impairments'] = data['impairments']
            self.data['impairments_snr'] = data['impairments_snr']
            self.data['impairments_linewidth'] = data['impairments_linewidth']
            # self.data['fb'] = float(np.round(data['fb'] * 1e-9, 1))

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
            self.sio.emit('status')
        else:
            self.data["ping"] = f'-'

    def initialise(self):
        self.pinger.start()
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

    def terminate(self):
        if self.sio.connected:
            self.sio.disconnect()
        self.pinger.stop()

    def make_gui(self) -> QWidget:
        return FLayout(
            ('', self.btn_connect),
            ("Ping", make_label(bind=self.data.bind("ping", '-'))),
            ("Sampling Rate (GBaud)", make_float_input(1, 60, 1, bind=self.data.bind("fb", 60.0))),
            ("Symbol Rate",  QLabel("90GHz")),
            ("Power target (dBm)", make_float_input(-50, 20, 1, bind=self.data.bind("power", 0))),
            ("Input power", make_label(formatting="{:.2f}dBm", bind=self.data.bind("power_mon_in", -50))),
            ("Output power", make_label(formatting="{:.2f}dBm", bind=self.data.bind("power_mon_out", -50))),
            ("Impairments", make_checkbox(bind=self.data.bind("impairments", False))),
            ("Target SNR (dB)", make_float_input(-10, 50, 1, bind=self.data.bind("impairments_snr", 50))),
            ("Phase Noise (kHz)", make_float_input(1, 1e5, 10, bind=self.data.bind("impairments_linewidth", 0))),
            ("Route", make_combo("B2B", "ULL", "NDFF", bind=self.data.bind("route", "B2B"))),
            ("NDFF Window", make_button('Open', lambda _: self.ndff_window.show())),
            ("Fibre length", make_label(formatting="{:.2f}km", bind=self.data.bind("fibre_len", 0))),
            ("SNR x/y (dB)", make_label(bind=self.data.bind("snr_est", '- / -'))),
        )

    def propagate(self, const: jnp.ndarray, rng_key: int, seq_len: int) -> Tuple[jnp.ndarray, float]:
        tx = self.get_tx(const, rng_key, seq_len)[1]
        if not self.sio.connected:
            return tx[0], 0
        self.sio.emit('tx', data=(array_to_bytes(const), rng_key, seq_len))
        try:
            rx, snr = self.queue.get(True, timeout=40)
            self.data['snr_est'] = f'{snr[0]:.1f} / {snr[1]:.1f}'
            if jnp.iscomplexobj(rx):
                rx = jnp.array([rx[0].real, rx[0].imag])
            return rx, snr[0]
        except queue.Empty:
            return tx[0], 0


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
