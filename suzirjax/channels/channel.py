from typing import Tuple
import jax
from jax import numpy as jnp

from suzirjax.gui_helpers import Connector
from PyQt5.QtWidgets import QWidget


class Channel:
    NAME: str

    def __init__(self, parent):
        self.parent = parent
        self.data = Connector()

    def initialise(self):
        pass

    def terminate(self):
        pass

    def make_gui(self) -> QWidget:
        raise NotImplemented

    @staticmethod
    def get_tx(const, rng_key, seq_len):
        if not jnp.iscomplexobj(const):
            const = jnp.array([const[:, 0] + 1j * const[:, 1]])
        if isinstance(rng_key, int):
            rng_key = jax.random.PRNGKey(rng_key)
        sym_idx = jax.random.randint(rng_key, (2, seq_len), minval=0, maxval=const.size)
        sym = jnp.take(const, sym_idx)
        return sym_idx, sym

    def propagate(self, const: jnp.ndarray, rng_key: jnp.ndarray, seq_len: int) -> Tuple[jnp.ndarray, float]:
        return self.get_tx(const, rng_key, seq_len)[1][0], 0.

