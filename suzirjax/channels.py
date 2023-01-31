import time

import jax
from jax import numpy as jnp
from gui_helpers import *
from typing import Tuple
from PyQt5.QtWidgets import QWidget


class Channel:
    NAME: str

    def __init__(self):
        self.data = Connector()

    def make_gui(self) -> QWidget:
        raise NotImplemented

    def propagate(self, signal: jnp.ndarray) -> Tuple[jnp.ndarray, float]:
        return signal, 0


class AWGNChannel(Channel):
    NAME = 'AWGN'

    def __init__(self):
        super().__init__()
        self.key = jax.random.PRNGKey(time.time_ns())
        self.sigma = 0
        self.data.bind('snr', 12).on(self._set_sigma)

    def _set_sigma(self, snr):
        self.sigma = 10 ** (-snr / 20)

    def make_gui(self) -> QWidget:
        return FLayout(
            ("SNR (dB)", make_float_input(-50, 50, 0.1, bind=self.data.bind("snr"))),
        )

    def propagate(self, tx: jnp.ndarray) -> Tuple[jnp.ndarray, float]:
        self.key, key = jax.random.split(self.key)
        noise = jax.random.normal(key, tx.shape) * self.sigma
        return tx + noise * (2 ** -.5), self.data['snr']


class PCAWGNChannel(Channel):
    """
    Partially Coherent Additive White Gaussian Noise

    Source:
    Sales-Llopis, M., & Savory, S. J. (2019).
    Approximating the Partially Coherent Additive White Gaussian Noise Channel in Polar Coordinates.
    IEEE Photonics Technology Letters, 31 (11), 833-836. https://doi.org/10.1109/lpt.2019.2909803
    """
    NAME = 'PC-AWGN'

    def __init__(self):
        super().__init__()
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
            ("SNR", make_label(formatting='{:.3f}dB', bind=self.data.bind("snr", 0.))),

        )

    def propagate(self, tx: jnp.ndarray) -> Tuple[jnp.ndarray, float]:
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
        self.data['snr'] = 10 * jnp.log10(snr)
        return jnp.array([rx.real, rx.imag]).T, self.data['snr']


class FibreChannel(Channel):
    NAME = 'NL-Fibre'

    def make_gui(self) -> QWidget:
        return FLayout(
            ("Distance (km)",
             make_float_input(1, 300, 1, bind=self.data.bind("length", 50))),
            ("Spans",
             make_int_input(1, 100, bind=self.data.bind("num_spans", 1))),
            ("Attenuation (dB/km)",
             make_float_input(0.01, 100, 0.1, bind=self.data.bind("attenuation", 0.2))),
            ("Gamma (/W/km)",
             make_float_input(0.1, 100, 0.1, bind=self.data.bind("nonlinear_coeff", 1.21))),
            ("Dispersion (ps/nm/km)",
             make_float_input(0.1, 100, 0.1, bind=self.data.bind("dispersion", 17))),
            ("Ref. Wavelength (nm)",
             make_float_input(1000, 2000, 1, bind=self.data.bind("ref_lambda", 1550))),
            ("Channels",
             make_int_input(1, 1000, bind=self.data.bind("num_channels", 1))),
            ("Symbol Rate (GBaud)",
             make_float_input(0.1, 500, 1, bind=self.data.bind("symbol_rate", 96))),
            ("Ch. Spacing (GHz)",
             make_float_input(0.1, 500, 1, bind=self.data.bind("ch_spacing", 100))),
            ("Ch. Power (dBm)",
             make_float_input(-50, 50, 1, bind=self.data.bind("ch_power_dBm", 0))),
            ("Max. NL-Phase",
             make_float_input(0.00001, 0.1, 0.00001, bind=self.data.bind("max_phi", 0.001))),
        )

    def split_step(self, signal):
        alpha = self.data['attenuation'] * 1e3
        length = self.data['length'] * 1e3
        gamma = self.data['gamma'] / 1e3
        max_phi = self.data['max_phi']
        fs = self.data['symbol_rate'] * 1e9
        dispersion = self.data['dispersion'] * 1e9
        ref_lambda = self.data['ref_lambda'] * 1e9
        c = 299792458.0

        ff = jnp.fft.fftfreq(signal.shape[-1], d=1 / fs)
        beta2 = -dispersion * ref_lambda ** 2 / (2 * jnp.pi * c)
        d = 0.5j * beta2 * (2. * jnp.pi * ff) ** 2

        jnp.fft.fft(signal)
        dz0 = max_phi / (jnp.abs(gamma) * jnp.max(jnp.abs(signal) ** 2))

        def _step(args):
            E_f, dist, dz = args
            dz = jnp.minimum(dz, length - dist)  # do not overshoot distance
            dist += dz

            dz_eff = (1 - jnp.exp(-alpha * dz)) / alpha
            nl_factor = 1j * dz_eff * gamma

            # Step
            D = jnp.ones((2, 1)) * jnp.exp(dz / 2 * d)
            E_t = jnp.fft.ifft(E_f * D)
            # Nonlinear operator SPM & XPM
            power = jnp.abs(E_t) ** 2  # Single polarisation
            if len(E_f.shape) > 1 and E_f.shape[0] > 1:  # Multiple polarisation
                power = jnp.sum(power, axis=0)
                nl_factor *= 8. / 9.  # Manakov factor
            E_t *= jnp.exp(nl_factor * power - (dz * alpha / 2))
            power_peak = jnp.max(E_t.real ** 2 + E_t.imag ** 2)
            E_f = jnp.fft.fft(E_t) * D
            dz = max_phi / (abs(gamma) * power_peak)

            # if callable(self.callback):
            #     self.callback(dz)

            return E_f, dist, dz

        signal_fd, _, _ = jax.lax.while_loop(
            lambda arg: arg[1] < length,
            _step,
            (jnp.fft.fft(signal), 0.0, dz0)
        )
        return jnp.fft.ifft(signal_fd)

    def propagate(self, tx: jnp.ndarray) -> Tuple[jnp.ndarray, float]:
        self.key, key = jax.random.split(self.key)
        noise = jax.random.normal(key, tx.shape) * self.sigma
        return tx + noise * (2 ** -.5), self.data['snr']


CHANNELS = [AWGNChannel, PCAWGNChannel, FibreChannel]
