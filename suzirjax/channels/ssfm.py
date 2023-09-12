
import jax
import time
from jax import numpy as jnp
from qampy import impairments
from functools import partial

from qampy.core.impairments import quantize_signal
from qampy.signals import SignalBase
from scipy.constants import pi, c, h

from suzirjax.channels import ChalmersQAMpy
from suzirjax.channels.channel import register_channel
from suzirjax.gui import *
from PyQt5.QtWidgets import *
import numpy as np


@partial(jax.jit, static_argnums=[1, 2, 3, 4, 5, 6, 7, 8])
def split_step_fourier_method(sig, fs, l_power, distance, step_size, dispersion, wavelength, alpha, gamma):
    # Step parameters
    nz = jnp.ceil(distance / step_size).astype(int)
    dz = distance / nz

    # Precompute params
    beta2 = -dispersion * wavelength ** 2 / (2 * pi * c)
    dz_eff = (1 - jnp.exp(-alpha * dz)) / alpha
    nl_factor = gamma * dz_eff * 1j
    dza = dz * alpha / 2

    ff = jnp.fft.fftfreq(sig.shape[-1], d=1 / fs)
    d = 0.5j * beta2 * (2. * pi * ff) ** 2
    D = jnp.ones((2, 1)) * jnp.exp(dz / 2 * d)  # Dispersion

    init_power = 10 * jnp.log10((jnp.abs(sig) ** 2).mean() * sig.shape[0]) + 30
    sig *= 10 ** ((l_power - init_power) / 20)
    # start in frequency domain
    sig = jnp.fft.fft(sig)

    def _step(i, _sig):
        _sig *= D
        _sig = jnp.fft.ifft(_sig)  # in time domain
        power = jnp.sum(jnp.abs(_sig) ** 2, axis=0)
        _sig *= jnp.exp(8. / 9. * nl_factor * power - dza)  # Manakov function
        _sig = jnp.fft.fft(_sig)  # in frequency domain
        _sig *= D
        return _sig
    sig = jax.lax.fori_loop(0, nz, _step, sig)

    # finish in time domain
    sig = jnp.fft.ifft(sig)
    return sig


@partial(jax.jit, static_argnums=[1, 2, 3, 4])
def dispersion_comp(sig, fs, distance, dispersion, wavelength):
    beta2 = -dispersion * wavelength ** 2 / (2 * pi * c)
    ff = jnp.fft.fftfreq(sig.shape[-1], d=1 / fs)
    d = 0.5j * beta2 * (2. * pi * ff) ** 2
    return jnp.fft.ifft(jnp.fft.fft(sig) * jnp.exp(distance * -d))


@partial(jax.jit, static_argnums=[3, 4])
def amplifier(sig, gain, nf, fs, wavelength, key):
    sig *= jnp.sqrt(gain)
    n = (nf * gain - 1) / (2 * (gain - 1))
    p_ase = 2 * n * (gain - 1) * h * (c / wavelength) * fs
    sigma = jnp.sqrt(0.5 / sig.shape[0] * p_ase)
    noise = jax.random.normal(key, sig.shape, dtype='complex') * jnp.sqrt(2) * sigma
    sig += noise
    return sig


@register_channel
class NonLinearChannel(ChalmersQAMpy):
    NAME = 'Non-Linear'

    def __init__(self, *args):
        super().__init__(*args)
        self.fs = 0
        self.key = jax.random.PRNGKey(time.time_ns())

        self.wavelength = 0
        self.distance = 0
        self.gamma = 0
        self.alpha = 0
        self.dispersion = 0
        self.step_size = 0
        self.power = 0
        self.sigma = 0
        self.amp_gain = 0
        self.amp_nf = 0

        self._ssfm = None
        self.wx = None

    def make_gui(self) -> QWidget:
        layout = FLayout(
            ("Sample Rate (GHz)", make_float_input(1e-3, 300, 1, bind=self.data.bind("fs", 50))),
            ("Symbol Rate (GBaud)", make_float_input(1e-3, 300, 1, bind=self.data.bind("fb", 25))),
            ("Linewidth (kHz)", make_float_input(1, 1e6, 1, bind=self.data.bind("linewidth", 100))),

            ("TX SNR (dB)", make_float_input(-10, 60, 1, bind=self.data.bind("tx_noise", 20))),
            ("RX SNR (dB)", make_float_input(-10, 60, 1, bind=self.data.bind("rx_noise", 20))),
            ("Amp. NF (dB)", make_float_input(-10, 50, 1, bind=self.data.bind("amp_nf", 6))),
            ("Amp. Gain (dB)", make_float_input(-10, 50, 1, bind=self.data.bind("amp_gain", 20))),
            ("DAC Res. (bits)", make_int_input(2, 32, 1, bind=self.data.bind("dac_res", 6))),
            ("Wavelength (nm)", make_float_input(800, 2200, 1, bind=self.data.bind("wavelength", 1550))),
            ("Distance (km)", make_float_input(0.01, 1e6, 1, bind=self.data.bind("distance", 100))),
            ("Dispersion (ps/nm/km)", make_float_input(0, 500, 0.1, bind=self.data.bind("dispersion", 17))),
            ("Attenuation (dB/km)", make_float_input(0.005, 100, 0.1, bind=self.data.bind("attenuation", 0.2))),
            ("Gamma (1/W/km)", make_float_input(0, 500, 0.01, bind=self.data.bind("gamma", 1.16))),
            ("TX Power (dBm)", make_float_input(-50, 30, 0.1, bind=self.data.bind("power", 0))),
            ("Fibre Power (dBm)", make_label(formatting='{:.3f}', bind=self.data.bind("power_2", -np.inf))),
            ("RX Power (dBm)", make_label(formatting='{:.3f}', bind=self.data.bind("power_rx", -np.inf))),

            ("Step-Size (m)", make_float_input(0.01, 1e6, 10, bind=self.data.bind("step_size", 100))),

            ("Sync taps", make_int_input(1, 300, 1, bind=self.data.bind("ntaps", 17))),
            ("Enable CPE", make_checkbox(bind=self.data.bind("cpe_en", True))),
            ("Frame Synced", make_label(bind=self.data.bind("synced", False))),
            ("SNR (dB)", make_label(bind=self.data.bind("snr", '- / -'))),
        )
        self.data.on('fs', lambda val: setattr(self, 'fs', val * 1e9))
        self.data.on('wavelength', lambda val: setattr(self, 'wavelength', val * 1e-9))
        self.data.on('distance', lambda val: setattr(self, 'distance', val * 1e3))
        self.data.on('gamma', lambda val: setattr(self, 'gamma', val * 1e-3))
        self.data.on('attenuation', lambda val: setattr(self, 'alpha', val * 1e-3 * float(jnp.log(10) / 10)))
        self.data.on('dispersion', lambda val: setattr(self, 'dispersion', val * 1e-6))
        self.data.on('step_size', lambda val: setattr(self, 'step_size', val))
        self.data.on('power', lambda val: setattr(self, 'power', val))
        self.data.on('tx_noise', lambda val: setattr(self, 'sigma', 10 ** (val / 20)))
        self.data.on('amp_gain', lambda val: setattr(self, 'amp_gain', 10 ** (val / 10)))
        self.data.on('amp_nf', lambda val: setattr(self, 'amp_nf', 10 ** (val / 10)))
        self.data.on('fb', lambda fb: self.parent.data.set('throughput_factor', fb * 1e9 * 2))
        return layout

    def impairments(self, sig: SignalBase):
        m = sig.shape[0]  # num of polarisation
        sig = impairments.rotate_field(sig, pi / 0.1)
        sig = impairments.change_snr(sig, self.data['tx_noise'])

        # Transmission
        self.key, key = jax.random.split(self.key, num=2)
        sig_ = split_step_fourier_method(
            sig, self.fs, self.power, self.distance, self.step_size, self.dispersion, self.wavelength, self.alpha, self.gamma)
        self.data['power_2'] = 10 * np.log10((abs(sig_) ** 2).mean() * m) + 30

        # ASE noise from amplifier
        # sig_ = amplifier(sig, self.amp_gain, self.amp_nf, self.fs, self.wavelength, key)
        self.data['power_rx'] = 10 * np.log10((abs(sig_) ** 2).mean() * m) + 30

        sig_ = dispersion_comp(
            sig_, self.fs, self.distance, self.dispersion, self.wavelength
        )
        sig = sig.recreate_from_np_array(np.array(sig_))
        sig = impairments.apply_phase_noise(sig, self.data['linewidth'] * 1e3)
        sig = sig.recreate_from_np_array(quantize_signal(sig, nbits=self.data['dac_res']))
        sig = impairments.change_snr(sig, self.data['rx_noise'])
        return sig
