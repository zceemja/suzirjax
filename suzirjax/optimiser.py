import jax
from jax import numpy as jnp
import optax
from suzirjax.gui import *
from PyQt5.QtWidgets import QWidget
from typing import List, Tuple

from suzirjax.modulation import Modulation

# import matplotlib
# matplotlib.use('module://backend_interagg')
# from matplotlib import pyplot as plt

OPTIMISERS = []


def register_optimiser(cls):
    OPTIMISERS.append(cls)
    return cls


@register_optimiser
class Optimiser:
    NAME: str = 'None'

    def __init__(self, data: Connector):
        self.data = Connector()
        self.D = 2
        self.parent_data = data
        self._prev = None

    def make_gui(self) -> QWidget:
        return FLayout(
            (make_label(formatting='{0.NAME} ({0.UNIT})', bind=self.parent_data.bind("metric_method")),
             make_label(formatting='{:.5f}', bind=self.data.bind("metric", np.nan))),
            *self._extra_gui_elements(),
            ("Allow decrease", make_checkbox(bind=self.data.bind("allow_decrease", True))),
        )

    def _extra_gui_elements(self) -> List[Tuple[str, QWidget]]:
        return []

    def optimise(self, const: Modulation, rx, tx_seq, snr) -> Tuple[Modulation, float]:
        if jnp.iscomplexobj(rx):
            rx = jnp.array([rx[0].real, rx[0].imag]).T
        tx = jnp.take(const.regular[:, :2], tx_seq[0], axis=0)
        nx = rx - tx
        # metric = jax.jit(self.parent_data['metric_method'])(const, nx, tx_seq, snr)
        metric = self.parent_data['metric_method'](const, nx, tx_seq, snr)
        return const, metric

    def update(self, const: Modulation, rx: jnp.ndarray, snr: float, tx_seq: jnp.ndarray = None) -> Modulation:
        if tx_seq is None:
            tx_seq = jnp.tile(jnp.arange(self.M), (rx.shape[0] // self.M))

        new_const, metric = self.optimise(const, rx, tx_seq, snr)
        if jnp.isnan(metric):
            return const
        if not self.data['allow_decrease'] and metric < self.data.get('metric'):
            return const

        self.data['metric'] = metric
        self.parent_data['metric'] = metric
        return new_const


@register_optimiser
class GradientDescentOpt(Optimiser):
    NAME = 'Gradient Descent'

    def _extra_gui_elements(self) -> List[Tuple[str, QWidget]]:
        return [
            # ("GMI (precise)", make_label(formatting='{:.5f}', bind=self.data.bind("gmi_f32", -np.Inf))),
            # ("GMI error", make_label(formatting='{:.6f}', bind=self.data.bind("gmi_delta", -np.Inf))),
            ("||grad||", make_label(formatting='{:.3f}', bind=self.data.bind("grad_size", 0.))),
            ("Learning rate (10^n)", make_float_input(-6, 2, 0.1, bind=self.data.bind("learning_rate", -1.))),
            ("Gradient Symmetry", make_checkbox(bind=self.data.bind("symmetry", False))),
        ]

    def optimise(self, const: Modulation, rx, tx_seq, snr) -> Tuple[Modulation, float]:
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

        nx = rx - jnp.take(const.regular, tx_seq, axis=0)
        metric, grad = jax.value_and_grad(self.parent_data['metric_method'])(const, nx, tx_seq, snr)

        if jnp.any(jnp.isnan(grad)) or metric < 0:
            self.parent_data['gmi_grad'] = grad
            return const, metric if metric > 0 else jnp.nan

            # gmi_grad = gmi_grad

        # if self.data['symmetry']:
        #     inv = jnp.array([[1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=float)
        #     gmi_grad = gmi_grad[:gmi_grad.shape[0] // 4]
        #     gmi_grad = inv[None, :, :] * gmi_grad[:, None, :]
        #     gmi_grad = gmi_grad.transpose((1, 0, 2)).reshape(-1, 2)
        #     const = const[:const.shape[0] // 4]
        #     const = inv[None, :, :] * const[:, None, :]
        #     const = const.transpose((1, 0, 2)).reshape(-1, 2)

        self.data['metric'] = metric
        self.parent_data['metric'] = metric
        const = Modulation(const.regular + 10 ** self.data['learning_rate'] * grad)
        # const /= jnp.sqrt(jnp.mean(jnp.sum(const ** 2, axis=1)))
        self.data['grad_size'] = jnp.sqrt((grad ** 2).sum())
        return const, metric


@register_optimiser
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
            ("||grad||", make_label(formatting='{:.3f}', bind=self.data.bind("grad_size", 0.)))
        ]

    def optimise(self, const: Modulation, rx, tx_seq, snr) -> Tuple[Modulation, float]:
        if self.opt_state is None:
            self.opt_state = self.optimiser.init(const)

        # const, self.opt_state = jax.jit(self._opt_update)(const, self.opt_state, rx, tx_bits, snr)
        # gmi = self.gmi_max_log(const, rx, tx_bits, snr)

        # scaling = np.sqrt(jnp.mean(jnp.sum(const ** 2, axis=1) / (self.D / 2)))
        nx = rx - jnp.take(const.regular, tx_seq, axis=0)  # TODO: move
        # tx_bits = jnp.take(self.bmap, tx_seq, axis=0)

        metric, grad = jax.value_and_grad(self.parent_data['metric_method'])(const, nx, tx_seq, snr)

        # gmi, gmi_grad = jax.value_and_grad(self.gmi_max_log)(const, rx, tx_bits, snr)
        if jnp.any(jnp.isnan(grad)) or metric < 0:
            return const, metric if metric > 0 else jnp.nan
        # gmi_grad /= scaling
        self.data['metric'] = metric
        self.parent_data['metric'] = metric
        self.parent_data['gmi_grad'] = grad
        updates, self.opt_state = self.optimiser.update(-grad, self.opt_state, const.regular)
        const = Modulation(optax.apply_updates(const.regular, updates))

        self.data['grad_size'] = np.sqrt((grad ** 2).sum())
        return const, metric


@register_optimiser
class RMSPropOpt(AdamOpt):
    NAME = 'RMSProp'

    def __init__(self, data: Connector, learning_rate=0.01):
        super().__init__(data)
        self.optimiser = optax.rmsprop(learning_rate)


@register_optimiser
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
