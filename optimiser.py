import jax
import numpy as np
from PyQt5.QtWidgets import QWidget
from jax import numpy as jnp
from jax.scipy import optimize
import optax
from gui_helpers import *
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

    def gmi_max_log(self, const: jnp.ndarray, rx: jnp.ndarray, tx_bits: jnp.ndarray, snr: float) -> float:
        """
        Computes the generalized mutual information (GMI) using the max-log approximation.
        """
        a = (abs(const)**2).mean() * self.D
        const /= a
        # rx /= a

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
        )

    def _extra_gui_elements(self) -> List[Tuple[str, QWidget]]:
        return []

    def optimise(self, const, rx, tx_bits, snr) -> Tuple[jnp.ndarray, float]:
        hashed = hash(const.tobytes()) ^ hash(float(snr))
        if self._prev is None or self._prev[0] != hashed:
            gmi = jax.jit(self.gmi_max_log)(const, rx, tx_bits, snr)
            self._prev = hashed, gmi
            return const, gmi
        return const, self._prev[1]

    def update(self, const: jnp.ndarray, rx: jnp.ndarray, snr: float, tx_seq: jnp.ndarray = None) -> jnp.ndarray:
        if tx_seq is None:
            # Assume just sending all same bits in repeat
            tx_bits = jnp.tile(self.bmap, (rx.shape[0] // self.M, 1))
        else:
            tx_bits = jnp.take(self.bmap, tx_seq)

        new_const, gmi = self.optimise(const, rx, tx_bits, snr)
        if not self.data['allow_decrease'] and gmi < self.data.get('gmi'):
            return const
        self.data['gmi'] = gmi
        return new_const


class GradientDescentOpt(Optimiser):
    NAME = 'Gradient Descent'

    def make_gui(self) -> QWidget:
        return FLayout(
            ("GMI", make_label(formatting='{:.3f}', bind=self.data.bind("gmi", -np.Inf))),
            ("||grad GMI||", make_label(formatting='{:.3f}', bind=self.data.bind("grad_gmi", 0.))),
            ("Learning rate (10^n)", make_float_input(-6, 2, 0.1, bind=self.data.bind("learning_rate", -1.))),
            ("Allow decrease", make_checkbox(bind=self.data.bind("allow_decrease", True))),
        )

    def _extra_gui_elements(self) -> List[Tuple[str, QWidget]]:
        return [
            ("||grad GMI||", make_label(formatting='{:.3f}', bind=self.data.bind("grad_gmi", 0.))),
            ("Learning rate (10^n)", make_float_input(-6, 2, 0.1, bind=self.data.bind("learning_rate", -1.))),
        ]

    def optimise(self, const, rx, tx_bits, snr) -> Tuple[jnp.ndarray, float]:
        gmi, gmi_grad = jax.value_and_grad(self.gmi_max_log)(const, rx, tx_bits, snr)
        self.data['grad_gmi'] = jnp.sqrt((gmi_grad ** 2).sum())
        return const + (10 ** self.data['learning_rate'] * gmi_grad), gmi


class AdamOpt(Optimiser):
    NAME = 'ADAM'

    def __init__(self, data: Connector):
        super().__init__(data)
        self.optimiser = optax.adam(0.01)
        self.opt_state = None

    def _update_points(self, M):
        super(AdamOpt, self)._update_points(M)
        self.opt_state = None

    def _extra_gui_elements(self) -> List[Tuple[str, QWidget]]:
        return [
            ("||grad GMI||", make_label(formatting='{:.3f}', bind=self.data.bind("grad_gmi", 0.)))
        ]

    def optimise(self, const, rx, tx_bits, snr) -> Tuple[jnp.ndarray, float]:
        if self.opt_state is None:
            self.opt_state = self.optimiser.init(const)
        gmi, gmi_grad = jax.value_and_grad(self.gmi_max_log)(const, rx, tx_bits, snr)
        updates, self.opt_state = self.optimiser.update(-gmi_grad, self.opt_state, const)
        const = optax.apply_updates(const, updates)
        self.data['grad_gmi'] = jnp.sqrt((gmi_grad ** 2).sum())
        return const, gmi


class SM3Opt(AdamOpt):
    NAME = 'SM3'

    def __init__(self, data: Connector):
        super().__init__(data)
        self.optimiser = optax.sm3(0.01)


class AdaFactorOpt(AdamOpt):
    NAME = 'AdaFactor'

    def __init__(self, data: Connector):
        super().__init__(data)
        self.optimiser = optax.adafactor()


class BFGSOpt(Optimiser):
    NAME = 'BFGS'

    def optimise(self, const, rx, tx_bits, snr) -> Tuple[jnp.ndarray, float]:
        res = optimize.minimize(
            lambda x, *args: -self.gmi_max_log(x.reshape((-1, 2)), *args), const.flatten(),
            args=(rx, tx_bits, snr), method='BFGS')
        const = res.x.reshape((-1, 2))
        return const, -res.fun


OPTIMISERS = [Optimiser, GradientDescentOpt, AdamOpt, SM3Opt, AdaFactorOpt, BFGSOpt]
