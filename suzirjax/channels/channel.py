from typing import Tuple
import jax
from jax import numpy as jnp

from suzirjax.gui_helpers import Connector
from PyQt5.QtWidgets import QWidget

# TX idx, RX sig, 2-pol SNR
ch_out = Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]

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
        key1, key2 = jax.random.split(rng_key)
        # Make sure we have at least one instance of each symbol, fill rest with random and shuffle
        sym_idx = jnp.tile(jnp.arange(const.size), (2, 1))
        sym_idx_rng = jax.random.randint(key1, (2, seq_len - const.size), minval=0, maxval=const.size)
        sym_idx = jnp.append(sym_idx, sym_idx_rng, axis=1)
        sym_idx = jax.random.permutation(key2, sym_idx, axis=1, independent=True)
        sym = jnp.take(const, sym_idx)
        return sym_idx, sym

    def propagate(self, const: jnp.ndarray, rng_key: jnp.ndarray, seq_len: int) -> ch_out:
        tx_idx, tx = self.get_tx(const, rng_key, seq_len)
        return tx_idx, tx, (0., 0.)
