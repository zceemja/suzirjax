from typing import Tuple

from PyQt5.QtWidgets import QWidget
from jax import numpy as jnp
from gui_helpers import Connector


class Channel:
    NAME: str

    def __init__(self, parent):
        self.parent = parent
        self.data = Connector()

    def initialise(self):
        pass

    def make_gui(self) -> QWidget:
        raise NotImplemented

    def propagate(self, const: jnp.ndarray, signal: jnp.ndarray) -> Tuple[jnp.ndarray, float]:
        return signal, 0

