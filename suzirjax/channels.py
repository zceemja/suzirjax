# import time
#
# import jax
# from PyQt5.QtCore import QTimer
# from jax import numpy as jnp
# from gui_helpers import *
# from typing import Tuple
# from PyQt5.QtWidgets import *
# from scipy.constants import c
# import os
# import socketio
#
#
# class FibreChannel(Channel):
#     NAME = 'NL-Fibre'
#
#     def make_gui(self) -> QWidget:
#         return FLayout(
#             ("Distance (km)",
#              make_float_input(1, 300, 1, bind=self.data.bind("length", 50))),
#             ("Spans",
#              make_int_input(1, 100, bind=self.data.bind("num_spans", 1))),
#             ("Attenuation (dB/km)",
#              make_float_input(0.01, 100, 0.1, bind=self.data.bind("attenuation", 0.2))),
#             ("Gamma (/W/km)",
#              make_float_input(0.1, 100, 0.1, bind=self.data.bind("nonlinear_coeff", 1.21))),
#             ("Dispersion (ps/nm/km)",
#              make_float_input(0.1, 100, 0.1, bind=self.data.bind("dispersion", 17))),
#             ("Ref. Wavelength (nm)",
#              make_float_input(1000, 2000, 1, bind=self.data.bind("ref_lambda", 1550))),
#             ("Channels",
#              make_int_input(1, 1000, bind=self.data.bind("num_channels", 1))),
#             ("Symbol Rate (GBaud)",
#              make_float_input(0.1, 500, 1, bind=self.data.bind("symbol_rate", 96))),
#             ("Ch. Spacing (GHz)",
#              make_float_input(0.1, 500, 1, bind=self.data.bind("ch_spacing", 100))),
#             ("Ch. Power (dBm)",
#              make_float_input(-50, 50, 1, bind=self.data.bind("ch_power_dBm", 0))),
#             ("Max. NL-Phase",
#              make_float_input(0.00001, 0.1, 0.00001, bind=self.data.bind("max_phi", 0.001))),
#         )
#
#     def split_step(self, signal):
#         alpha = self.data['attenuation'] * 1e3
#         length = self.data['length'] * 1e3
#         gamma = self.data['gamma'] / 1e3
#         max_phi = self.data['max_phi']
#         fs = self.data['symbol_rate'] * 1e9
#         dispersion = self.data['dispersion'] * 1e9
#         ref_lambda = self.data['ref_lambda'] * 1e9
#
#         ff = jnp.fft.fftfreq(signal.shape[-1], d=1 / fs)
#         beta2 = -dispersion * ref_lambda ** 2 / (2 * jnp.pi * c)
#         d = 0.5j * beta2 * (2. * jnp.pi * ff) ** 2
#
#         jnp.fft.fft(signal)
#         dz0 = max_phi / (jnp.abs(gamma) * jnp.max(jnp.abs(signal) ** 2))
#
#         def _step(args):
#             E_f, dist, dz = args
#             dz = jnp.minimum(dz, length - dist)  # do not overshoot distance
#             dist += dz
#
#             dz_eff = (1 - jnp.exp(-alpha * dz)) / alpha
#             nl_factor = 1j * dz_eff * gamma
#
#             # Step
#             D = jnp.ones((2, 1)) * jnp.exp(dz / 2 * d)
#             E_t = jnp.fft.ifft(E_f * D)
#             # Nonlinear operator SPM & XPM
#             power = jnp.abs(E_t) ** 2  # Single polarisation
#             if len(E_f.shape) > 1 and E_f.shape[0] > 1:  # Multiple polarisation
#                 power = jnp.sum(power, axis=0)
#                 nl_factor *= 8. / 9.  # Manakov factor
#             E_t *= jnp.exp(nl_factor * power - (dz * alpha / 2))
#             power_peak = jnp.max(E_t.real ** 2 + E_t.imag ** 2)
#             E_f = jnp.fft.fft(E_t) * D
#             dz = max_phi / (abs(gamma) * power_peak)
#
#             # if callable(self.callback):
#             #     self.callback(dz)
#
#             return E_f, dist, dz
#
#         signal_fd, _, _ = jax.lax.while_loop(
#             lambda arg: arg[1] < length,
#             _step,
#             (jnp.fft.fft(signal), 0.0, dz0)
#         )
#         return jnp.fft.ifft(signal_fd)
#
#     def propagate(self, const: jnp.ndarray, tx: jnp.ndarray) -> Tuple[jnp.ndarray, float]:
#         self.key, key = jax.random.split(self.key)
#         noise = jax.random.normal(key, tx.shape) * self.sigma
#         return tx + noise * (2 ** -.5), self.data['snr']
#
#
#
# CHANNELS = [AWGNChannel, PCAWGNChannel, FibreChannel, RemoteChannel]
