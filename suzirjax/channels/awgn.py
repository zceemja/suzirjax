from .channel import Channel
from gui_helpers import *
from PyQt5.QtWidgets import QWidget

import jax
from jax import numpy as jnp
import time
from typing import Tuple


class AWGNChannel(Channel):
    NAME = 'AWGN'

    def __init__(self, *args):
        super().__init__(*args)
        self.key = jax.random.PRNGKey(time.time_ns())
        self.sigma = 0
        self.data.bind('snr', 12).on(self._set_sigma)

    def _set_sigma(self, snr):
        self.sigma = 10 ** (-snr / 20)

    def make_gui(self) -> QWidget:
        return FLayout(
            ("SNR (dB)", make_float_input(-50, 50, 0.1, bind=self.data.bind("snr"))),
            ("SNR", make_label(formatting='{:.3f}dB', bind=self.data.bind("snr_msq", 0.))),
        )

    def propagate(self, const: jnp.ndarray, tx: jnp.ndarray) -> Tuple[jnp.ndarray, float]:
        self.key, key = jax.random.split(self.key)
        noise = jax.random.normal(key, tx.shape) * self.sigma
        rx = tx + noise * (2 ** -.5)
        snr = (
                    jnp.sum(jnp.abs(tx.flatten()) ** 2, axis=-1) /
                    jnp.sum(jnp.abs(rx.flatten() - tx.flatten()) ** 2, axis=-1)
            ).mean()
        self.data['snr_msq'] = 10 * jnp.log10(snr)
        return rx, self.data['snr_msq']


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
        self.data.bind('noise', 12).on(self._set_sigma)
        self.data.bind('linewidth', 50).on(self._set_std, False)
        self.data.bind('fs', 25).on(self._set_std)

    def _set_sigma(self, snr):
        self.sigma = 10 ** (-snr / 20)

    def _set_std(self, _):
        self.std = jnp.sqrt(2 * jnp.pi * (1 / (self.data['fs'] * 1e9 + 1)) * self.data['linewidth'] * 1e3)

    def make_gui(self) -> QWidget:
        return FLayout(
            ("Noise Figure (dB)", make_float_input(-50, 50, 0.1, bind=self.data.bind("noise"))),
            ("Sample Rate (GHz)", make_float_input(0.001, 500, 1, bind=self.data.bind("fs", 25))),
            ("Linewidth (kHz)", make_float_input(1e-3, 1e6, 10, bind=self.data.bind("linewidth", ))),
            ("SNR", make_label(formatting='{:.3f}dB', bind=self.data.bind("snr_msq", 0.))),
        )

    def propagate(self,  const: jnp.ndarray, tx: jnp.ndarray) -> Tuple[jnp.ndarray, float]:
        self.key, key1, key2 = jax.random.split(self.key, num=3)
        tx = tx[:, 0] + 1j * tx[:, 1]
        noise = jax.random.normal(key1, shape=tx.shape, dtype=tx.dtype) * self.sigma * (2 ** -.5)
        # phase = jnp.cumsum(, axis=-1)
        phase = jnp.exp(1j * jax.random.normal(key2, shape=tx.shape) * self.std)
        rx = tx * phase + noise
        snr = (
                    jnp.sum(jnp.abs(tx) ** 2, axis=-1) /
                    jnp.sum(jnp.abs(rx - tx) ** 2, axis=-1)
            )
        self.data['snr_msq'] = 10 * jnp.log10(snr)
        return jnp.array([rx.real, rx.imag]).T, self.data['snr_msq']
