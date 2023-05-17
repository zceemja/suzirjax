import jax
import numpy as np
from jax import numpy as jnp
import optax
from suzirjax.gui_helpers import *
from PyQt5.QtWidgets import QWidget
from typing import List, Tuple


def bitmap_indices(bmap: jnp.ndarray) -> jnp.ndarray:
    a, b = jnp.nonzero(bmap.T)
    return jnp.stack([b[a == i] for i in jnp.arange(bmap.shape[1])]).T


class Optimiser:
    NAME: str = 'None'

    def __init__(self, data: Connector):
        self.data = Connector()
        self.D = 2
        self.parent_data = data
        self._prev = None
        data.on('mod_points', self._update_points)
        data.on('mod_name', self._update_points, False)

    def _update_points(self, _):
        self.M = self.parent_data['mod_points']
        m = int(jnp.log2(self.M))
        self.bmap = (jnp.arange(self.M)[:, None] >> jnp.arange(m) & 1).astype(bool)
        self.bmap_idx_true = bitmap_indices(self.bmap)
        self.bmap_idx_false = bitmap_indices(~self.bmap)
        self.data['gmi'] = -np.Inf

    def gmi_log_sum(self, const: jnp.ndarray, nx: jnp.ndarray, tx_seq: jnp.ndarray, snr: float) -> jnp.ndarray:
        """
        Computes the generalized mutual information (GMI) using the log sum approximation.
        """
        sigma = 10 ** (-snr / 20)

        scaling = jnp.sqrt(jnp.mean(jnp.sum(const ** 2, axis=1) / (self.D / 2)))
        # scaling = jnp.sqrt((jnp.sum(const**2, axis=1))) * self.D
        const /= scaling
        nx /= scaling

        tx_bits = jnp.take(self.bmap, tx_seq, axis=0)
        tx = jnp.take(const, tx_seq, axis=0)

        rx = tx + nx

        # Compute the squared distance between the received and constellation points
        squared_distance = ((rx[:, None, :] - const[None, :, :]) ** 2).sum(axis=-1)

        # Compute the symbol likelihood based on the squared distance and the variance sigma
        symbol_log_likelihood = -squared_distance / sigma ** 2

        # Compute the log likelihood ratios for each sample
        max_true = jnp.log(jnp.sum(jnp.exp(symbol_log_likelihood[:, self.bmap_idx_true]), axis=1))
        max_false = jnp.log(jnp.sum(jnp.exp(symbol_log_likelihood[:, self.bmap_idx_false]), axis=1))
        log_likelihood_ratios = max_true - max_false

        # Compute the information loss for each sample
        information_loss = jnp.mean(jnp.log1p(jnp.exp((1 - 2 * tx_bits) * log_likelihood_ratios)), axis=0)

        # the GMI is the
        return (1 - information_loss / jnp.log(2)).sum()

    def gmi_max_log(self, const: jnp.ndarray, nx: jnp.ndarray, tx_seq: jnp.ndarray, snr: float) -> jnp.ndarray:
        """
        Computes the generalized mutual information (GMI) using the max-log approximation.
        """
        scaling = jnp.sqrt(jnp.mean(jnp.sum(const ** 2, axis=1) / (self.D / 2)))
        const /= scaling
        nx /= scaling

        tx_bits = jnp.take(self.bmap, tx_seq, axis=0)
        tx = jnp.take(const, tx_seq, axis=0)
        rx = tx + nx

        sigma = 10 ** (-snr / 20)
        # Compute the squared distance between the received and constellation points
        squared_distance = ((rx[:, None, :] - const[None, :, :]) ** 2).sum(axis=-1)

        # Compute the symbol likelihood based on the squared distance and the variance sigma
        symbol_log_likelihood = -squared_distance / sigma ** 2

        # Compute the log likelihood ratios for each sample
        max_true = jnp.max(symbol_log_likelihood[:, self.bmap_idx_true], axis=1)
        max_false = jnp.max(symbol_log_likelihood[:, self.bmap_idx_false], axis=1)
        log_likelihood_ratios = max_true - max_false

        # Compute the information loss for each sample
        information_loss = jnp.mean(jnp.log1p(jnp.exp((1 - 2 * tx_bits) * log_likelihood_ratios)), axis=0)

        # the GMI is the
        return (1 - information_loss / jnp.log(2)).sum()

    def make_gui(self) -> QWidget:
        return FLayout(
            ("GMI", make_label(formatting='{:.5f}', bind=self.data.bind("gmi", -np.Inf))),
            *self._extra_gui_elements(),
            ("Allow decrease", make_checkbox(bind=self.data.bind("allow_decrease", True))),
            # ("GMI Method", make_combo_dict({
            #     'Log-Sum': self.gmi_log_sum,
            #     'Max-Log': self.gmi_max_log,
            # }, bind=self.data.bind("gmi_method", self.gmi_log_sum))),
        )

    def _extra_gui_elements(self) -> List[Tuple[str, QWidget]]:
        return []

    def optimise(self, const, rx, tx_seq, snr) -> Tuple[jnp.ndarray, float]:
        if jnp.iscomplexobj(rx):
            rx = jnp.array([rx[0].real, rx[0].imag]).T
        if jnp.iscomplexobj(tx_seq):
            tx_seq = jnp.array([tx_seq[0].real, tx_seq[0].imag]).T
        nx = rx - jnp.take(const, tx_seq, axis=0)
        gmi = jax.jit(self.gmi_log_sum)(const, nx, tx_seq, snr)
        return const, gmi

    def update(self, const: jnp.ndarray, rx: jnp.ndarray, snr: float, tx_seq: jnp.ndarray = None) -> jnp.ndarray:
        if tx_seq is None:
            tx_seq = jnp.tile(jnp.arange(self.M), (rx.shape[0] // self.M))

        new_const, gmi = self.optimise(const, rx, tx_seq, snr)
        new_const /= jnp.sqrt(jnp.mean(jnp.abs(const) ** 2))
        if jnp.isnan(gmi):
            return const
        if not self.data['allow_decrease'] and gmi < self.data.get('gmi'):
            return const
        self.data['gmi'] = gmi
        self.parent_data['gmi'] = gmi
        return new_const


class GradientDescentOpt(Optimiser):
    NAME = 'Gradient Descent'

    def _extra_gui_elements(self) -> List[Tuple[str, QWidget]]:
        return [
            # ("GMI (precise)", make_label(formatting='{:.5f}', bind=self.data.bind("gmi_f32", -np.Inf))),
            # ("GMI error", make_label(formatting='{:.6f}', bind=self.data.bind("gmi_delta", -np.Inf))),
            ("||grad GMI||", make_label(formatting='{:.3f}', bind=self.data.bind("grad_gmi", 0.))),
            ("Learning rate (10^n)", make_float_input(-6, 2, 0.1, bind=self.data.bind("learning_rate", -1.))),
            ("Gradient Symmetry", make_checkbox(bind=self.data.bind("symmetry", False))),
        ]

    def optimise(self, const, rx, tx_seq, snr) -> Tuple[jnp.ndarray, float]:
        # dtype = jnp.float16
        # gmi, gmi_grad = jax.value_and_grad(self.gmi_max_log)(
        #     const.astype(dtype),
        #     rx.astype(dtype),
        #     tx_bits,
        #     snr.astype(dtype)
        # )
        # gmi = gmi.astype(jnp.float32)
        # gmi_grad = gmi_grad.astype(jnp.float32)
        # self.data['gmi_f32'] = jax.jit(self.gmi_max_log)(const, rx, tx_bits, snr)
        # self.data['gmi_delta'] = abs(self.data['gmi_f32'] - gmi)

        nx = rx - jnp.take(const, tx_seq, axis=0)
        gmi, gmi_grad = jax.value_and_grad(self.gmi_log_sum)(const, nx, tx_seq, snr)

        if jnp.any(jnp.isnan(gmi_grad)):
            self.parent_data['gmi_grad'] = gmi_grad
            return const, gmi

            # gmi_grad = gmi_grad

        if self.data['symmetry']:
            inv = jnp.array([[1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=float)
            gmi_grad = gmi_grad[:gmi_grad.shape[0] // 4]
            gmi_grad = inv[None, :, :] * gmi_grad[:, None, :]
            gmi_grad = gmi_grad.transpose((1, 0, 2)).reshape(-1, 2)
            const = const[:const.shape[0] // 4]
            const = inv[None, :, :] * const[:, None, :]
            const = const.transpose((1, 0, 2)).reshape(-1, 2)

        self.parent_data['gmi_grad'] = gmi_grad
        const += 10 ** self.data['learning_rate'] * gmi_grad
        # const /= jnp.sqrt(jnp.mean(jnp.sum(const ** 2, axis=1)))
        self.data['grad_gmi'] = jnp.sqrt((gmi_grad ** 2).sum())
        return const, gmi


class AdamOpt(Optimiser):
    NAME = 'ADAM'

    def __init__(self, data: Connector, learning_rate=0.2):
        super().__init__(data)
        self.optimiser = optax.adam(learning_rate)
        self.opt_state = None

    def _update_points(self, M):
        super(AdamOpt, self)._update_points(M)
        self.opt_state = None

    def _extra_gui_elements(self) -> List[Tuple[str, QWidget]]:
        return [
            ("||grad GMI||", make_label(formatting='{:.3f}', bind=self.data.bind("grad_gmi", 0.)))
        ]

    def optimise(self, const, rx, tx_seq, snr) -> Tuple[jnp.ndarray, float]:
        if self.opt_state is None:
            self.opt_state = self.optimiser.init(const)

        # const, self.opt_state = jax.jit(self._opt_update)(const, self.opt_state, rx, tx_bits, snr)
        # gmi = self.gmi_max_log(const, rx, tx_bits, snr)

        # scaling = np.sqrt(jnp.mean(jnp.sum(const ** 2, axis=1) / (self.D / 2)))
        nx = rx - jnp.take(const, tx_seq, axis=0)  # TODO: move
        # tx_bits = jnp.take(self.bmap, tx_seq, axis=0)

        gmi, gmi_grad = jax.value_and_grad(self.gmi_log_sum)(const, nx, tx_seq, snr)
        self.parent_data['gmi_grad'] = gmi_grad

        # gmi, gmi_grad = jax.value_and_grad(self.gmi_max_log)(const, rx, tx_bits, snr)
        if jnp.any(jnp.isnan(gmi_grad)):
            return const, gmi
        # gmi_grad /= scaling

        updates, self.opt_state = self.optimiser.update(-gmi_grad, self.opt_state, const)
        const = optax.apply_updates(const, updates)

        self.data['grad_gmi'] = np.sqrt((gmi_grad ** 2).sum())
        return const, gmi


class RMSPropOpt(AdamOpt):
    NAME = 'RMSProp'

    def __init__(self, data: Connector, learning_rate=0.01):
        super().__init__(data)
        self.optimiser = optax.rmsprop(learning_rate)


class YogiOpt(AdamOpt):
    NAME = 'Yogi'

    def __init__(self, data: Connector, learning_rate=0.01):
        super().__init__(data)
        self.optimiser = optax.yogi(learning_rate)


# class AdaFactorOpt(AdamOpt):
#     NAME = 'AdaFactor'
#
#     def __init__(self, data: Connector):
#         super().__init__(data)
#         self.optimiser = optax.adafactor()


OPTIMISERS = [Optimiser, GradientDescentOpt, AdamOpt, RMSPropOpt, YogiOpt]
