import os
import time

import socketio

from gui_helpers import *
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import *
from .channel import Channel
from jax import numpy as jnp


class RemoteChannel(Channel):

    NAME = 'Remote'

    def __init__(self, *args):
        super().__init__(*args)
        self.sio = socketio.Client()
        self.pinger = QTimer()
        self.pinger.timeout.connect(self._ping)

        @self.sio.event
        def connect():
            self.data['conn'] = True

        @self.sio.event
        def disconnect():
            self.data['conn'] = False

        @self.sio.event
        def connect_error(data):
            print("The connection failed!")

        @self.sio.event
        def pong(data):
            print(data)

        @self.sio.on('*')
        def catch_all(event, data):
            pass

    def _ping(self):
        if self.sio.connected:

            self.sio.emit('ping', time.time_ns().to_bytes(8, 'little'))
            # t = (time.time_ns() - t) / 1e6
            # if t > 1.:
            #     self.data["ping"] = f'{t:.1f}ms'
            # else:
            #     self.data["ping"] = f'{t * 1e3:.0f}µs'
        else:
            self.data["ping"] = f'-'

    def initialise(self):
        if self.sio.connected:
            return
        if 'http_proxy' in os.environ:
            del os.environ['http_proxy']  # Please, stop.
        try:
            self.sio.connect('http://localhost:49922', wait=False)
            self.pinger.start(1000)
        except socketio.exceptions.ConnectionError as e:
            text = QLabel()
            if 'Max retries exceeded with url' in str(e):
                text.setText('Failed to establish a new connection')
            else:
                text.setText(str(e))
            make_dialog("Remote connection", text, parent=self.parent, buttons=QDialogButtonBox.Close).exec()

    def make_gui(self) -> QWidget:
        return FLayout(
            ("Ping", make_label(bind=self.data.bind("ping", '-'))),
        )

    def propagate(self, const: jnp.ndarray, tx: jnp.ndarray) -> Tuple[jnp.ndarray, float]:
        if self.sio.connected:
            return tx, 0
