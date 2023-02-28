
import warnings

import jax
from qampy import signals, impairments, equalisation, phaserec, helpers, analog_frontend
from qampy.core.pythran_dsp import estimate_snr
from qampy.signals import SignalQAMGrayCoded

from typing import Tuple
from .channel import Channel
from gui_helpers import *
from PyQt5.QtWidgets import QWidget
from jax import numpy as jnp
import numpy as np


class ArbritarySignal(SignalQAMGrayCoded):

    @classmethod
    def from_symbol_array(cls, symbs, coded_symbols, fb=1, dtype=None):
        """
        Generate signal from a given symbol array.

        Parameters
        ----------
        symbs : subclass of SignalBase
            symbol array to base on
        coded_symbols  :
            complex type constellation
        fb : float, optional
            symbol rate
        dtype : np.dtype, optional
            dtype for the signal. The default of None means use the dtype from symbols
        Returns
        -------
        output : SignalQAMGrayCoded
            output signal based on symbol array
        """
        if dtype is not None:
            assert dtype in [np.complex128, np.complex64], "only np.complex128 and np.complex64  or None dtypes are supported"
        symbs = np.atleast_2d(symbs)
        P = (abs(np.unique(coded_symbols))**2).mean()
        if not np.isclose(P, 1):
            warnings.warn("Power of symbols is not normalized to 1, this might cause issues later")
        M = coded_symbols.size
        m = int(np.log2(M))
        # scale = np.sqrt(theory.cal_scaling_factor_qam(M)) / np.sqrt((abs(np.unique(symbs)) ** 2).mean())
        # coded_symbols, graycode, encoding, bitmap_mtx = cls._generate_mapping(M, scale, dtype=dtype)
        # out = np.empty_like(symbs).astype(dtype)
        # for i in range(symbs.shape[0]):
        #     out[i], _, idx = make_decision(np.copy(symbs[i]), coded_symbols) # need a copy to avoid a pythran error
        # bits = cls._demodulate(idx, encoding)
        encoding = (jnp.arange(M)[:, None] >> jnp.arange(m-1, -1, -1) & 1).astype(bool)
        obj = np.asarray(symbs).view(cls)
        obj._M = M
        obj._fb = fb
        obj._fs = fb
        # obj._bits = bits
        obj._encoding = np.asarray(encoding, np.complex64)
        obj._code = np.arange(M)
        # obj._bitmap_mtx = bitmap_mtx
        # obj._bitmap_mtx = bitmap_mtx
        obj._coded_symbols = np.asarray(coded_symbols.flatten(), np.complex64)
        obj._symbols = obj.copy()
        return obj


class ChalmersQAMpy(Channel):
    NAME = 'QAMpy'

    def __init__(self, *args):
        super().__init__(*args)
        self.wx = None

    def make_gui(self) -> QWidget:
        return FLayout(
            ("Sample Rate (GHz)", make_float_input(1, 300, 1, bind=self.data.bind("fb", 25))),
            ("Linewidth (kHz)", make_float_input(1, 1e6, 1, bind=self.data.bind("linewidth", 100))),
            ("AES (dB)", make_float_input(-10, 40, 1, bind=self.data.bind("ase", 15))),
            ("Sync taps", make_int_input(1, 300, 1, bind=self.data.bind("ntaps", 17))),
            ("Enable CPE", make_checkbox(bind=self.data.bind("cpe_en", True))),
            ("Frame Synced", make_label(bind=self.data.bind("synced", False))),
            ("SNR (dB)", make_label(bind=self.data.bind("snr", '- / -'))),
        )

    def propagate(self, const: jnp.ndarray, rng_key: int, seq_len: int) -> Tuple[jnp.ndarray, float]:
        tx = self.get_tx(const, rng_key, seq_len)[1]

        pilot_seq_len = 2 ** 12
        pilot_ins_rat = 5
        payload_size = tx.shape[-1]

        # (frame_length - pilot_seq_len)/pilot_ins_rat
        N = int((payload_size * pilot_ins_rat / (pilot_ins_rat - 1)) + pilot_seq_len)

        # # TO CHECK
        # def calc_n(p, pr, ps, n=None):
        #     N = p * pr / (pr - 1) + ps
        #     return N, ((n or N) - ps) / pr
        scale = np.sqrt(2) if (abs(const)**2).mean() == 2 else 1
        const /= scale
        tx /= scale

        M = const.size
        payload = ArbritarySignal.from_symbol_array(tx, const, fb=self.data['fb'] * 1e9)
        # payload = SignalQAMGrayCoded.from_symbol_array(tx, const.size, fb=self.data['fb'] * 1e9)
        sig = signals.SignalWithPilots(M, N, pilot_seq_len, pilot_ins_rat, nmodes=2, nframes=4, fb=self.data['fb'] * 1e9)
        sig = sig.from_symbol_array(payload, N, pilot_seq_len, pilot_ins_rat, pilots=None, nframes=4, fb=self.data['fb'] * 1e9)
        sig = sig.resample(2 * sig.fb, beta=0.01, renormalise=True)

        synced = False
        failed = -1
        while not synced:
            if failed > 100:
                return tx, 0
            failed += 1
            sig2 = impairments.apply_phase_noise(sig, self.data['linewidth'] * 1e3)
            # sig2 = impairments.change_snr(sig2, self.data['ase'])
            sig2 = impairments.rotate_field(sig2, np.pi / 0.1)

            sig2 = analog_frontend.orthonormalize_signal(sig2)
            sig2 = helpers.normalise_and_center(sig2)
            synced = sig2.sync2frame()

        if failed > 0:
            print(f"Failed to sync {failed} times")
        self.data['synced'] = synced
        sig2 = helpers.normalise_and_center(sig2)
        sig2 = analog_frontend.orthonormalize_signal(sig2)
        self.wx, s1 = equalisation.pilot_equaliser(sig2, (50e-6, 40e-5), self.data['ntaps'], apply=True, wxinit=self.wx, adaptive_stepsize=True)
        s1 = analog_frontend.orthonormalize_signal(s1)
        s1, ph = phaserec.pilot_cpe(s1, nframes=4)
        # data = s1.get_data()
        rx = s1.get_data()
        snr = 10 * jnp.log10(
                    jnp.sum(jnp.abs(tx) ** 2, axis=-1) /
                    jnp.sum(jnp.abs(rx - tx) ** 2, axis=-1)
        )
        self.data['snr'] = f'{snr[0]:.2f} / {snr[1]:.2f}'

        rx1 = rx[0] * scale
        res = jnp.array([rx1.real, rx1.imag]).T
        return res, snr[0]
