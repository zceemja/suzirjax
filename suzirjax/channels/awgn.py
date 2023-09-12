from .channel import Channel, ch_out, register_channel
from suzirjax.gui import *
from PyQt5.QtWidgets import QWidget

import jax
from jax import numpy as jnp
import time

from ..modulation import Modulation


@register_channel
class AWGNChannel(Channel):
    NAME = 'AWGN'

    def __init__(self, *args):
        super().__init__(*args)
        self.key = jax.random.PRNGKey(time.time_ns())
        self.sigma = 0
        self.data.bind('ase', 12).on(self._set_sigma)
        self.parent.data.set('throughput_factor', jnp.nan)

    def _set_sigma(self, snr):
        self.sigma = 10 ** (-snr / 20)

    def make_gui(self) -> QWidget:
        layout = FLayout(
            ("Symbol Rate (GBaud)", make_float_input(0.01, 300, 1, bind=self.data.bind("fb", 20))),
            ("Target SNR (dB)", make_float_input(-50, 50, 0.1, bind=self.data.bind("ase"))),
            ("SNR x/y (dB)", make_label(formatting='{:.3f} / {:.3f}', bind=self.data.bind("snr", (-jnp.nan, -jnp.nan)))),
        )
        self.data.on('fb', lambda fb: self.parent.data.set('throughput_factor', fb * 1e9 * 2))
        return layout

    def propagate(self, const: Modulation, rng_key: int, seq_len: int) -> ch_out:
        self.key, key = jax.random.split(self.key)
        tx_idx, tx = self.get_tx(const, rng_key, seq_len)

        p = jnp.sqrt(jnp.mean(jnp.abs(const.complex) ** 2) * tx.shape[0])
        noise = p * jax.random.normal(key, tx.shape, dtype=complex) * self.sigma
        rx = tx + noise
        snr = 10 * jnp.log10(
                    jnp.sum(jnp.abs(tx) ** 2, axis=-1) /
                    jnp.sum(jnp.abs(rx - tx) ** 2, axis=-1)
        )
        self.data['snr'] = snr[0], snr[-1]
        return tx_idx, rx, snr


@register_channel
class PCAWGNChannel(Channel):
    """
    Partially Coherent Additive White Gaussian Noise

    Source:
    Sales-Llopis, M., & Savory, S. J. (2019).
    Approximating the Partially Coherent Additive White Gaussian Noise Channel in Polar Coordinates.
    IEEE Photonics Technology Letters, 31 (11), 833-836. https://doi.org/10.1109/lpt.2019.2909803
    """
    NAME = 'PC-AWGN'

    def __init__(self, *args):
        super().__init__(*args)
        self.key = jax.random.PRNGKey(time.time_ns())
        self.sigma = 0
        self.data.bind('ase', 12).on(self._set_sigma)
        self.data.bind('linewidth', 50).on(self._set_std, False)
        self.data.bind('fs', 25).on(self._set_std)

    def _set_sigma(self, snr):
        self.sigma = 10 ** (-snr / 20)

    def _set_std(self, _):
        self.std = jnp.sqrt(2 * jnp.pi * (1 / (self.data['fs'] * 1e9 + 1)) * self.data['linewidth'] * 1e3)

    def make_gui(self) -> QWidget:
        layout = FLayout(
            ("Target SNR (dB)", make_float_input(-50, 50, 0.1, bind=self.data.bind("ase"))),
            ("Sample Rate (GHz)", make_float_input(0.001, 500, 1, bind=self.data.bind("fs", 25))),
            ("Linewidth (kHz)", make_float_input(1e-3, 1e6, 10, bind=self.data.bind("linewidth", ))),
            ("SNR", make_label(formatting='{:.3f}dB', bind=self.data.bind("snr", 0.))),
        )
        self.data.on('fs', lambda fs: self.parent.data.set('throughput_factor', fs * 1e9))  # assume 2 samples per symbol
        return layout

    def propagate(self, const: Modulation, rng_key: int, seq_len: int) -> ch_out:
        self.key, key1, key2 = jax.random.split(self.key, num=3)
        tx_idx, tx = self.get_tx(const, rng_key, seq_len)
        noise = jnp.sqrt(jnp.mean(jnp.abs(const.complex) ** 2)) * jax.random.normal(key1, shape=tx[0].shape, dtype=tx.dtype) * self.sigma * (2 ** -.5)
        # phase = jnp.cumsum(, axis=-1)
        phase = jnp.exp(1j * jax.random.normal(key2, shape=tx[0].shape) * self.std)
        rx = tx[0] * phase + noise
        snr = 10 * jnp.log10(
                    jnp.sum(jnp.abs(tx[0]) ** 2, axis=-1) /
                    jnp.sum(jnp.abs(rx - tx[0]) ** 2, axis=-1)
            )
        self.data['snr'] = snr
        rx = jnp.array([rx.real, rx.imag]).T
        return tx_idx, rx, snr
