import time

import jax
import matplotlib
import numpy as np
from jax import numpy as jnp
import logging
from matplotlib import pyplot as plt
from numpy.polynomial.hermite import hermgauss
from matplotlib.animation import FuncAnimation

from utils import kindlmann_cmap
from ong.utils import alphabet

from ong.utils.jax_utils import rand

matplotlib.use('Qt5Agg')
matplotlib.colormaps.register(kindlmann_cmap)


log = logging.getLogger(__name__)


def bitmap_indices(bmap: jnp.ndarray) -> jnp.ndarray:
    a, b = jnp.nonzero(bmap.T)
    return jnp.stack([b[a == i] for i in jnp.arange(bmap.shape[1])]).T


class ConstellationShaper:

    def __init__(self, points, dimensions=2, snr=15, max_iter=1000, min_region=1e-6, prng_seed=None):
        self.M = points
        self.m = int(jnp.log2(self.M))

        self.D = dimensions
        self.SNR = snr
        self.max_iter = max_iter
        self.min_region = min_region

        if prng_seed is None:
            prng_seed = time.time_ns()
        self.keygen = rand.sequence(seed=prng_seed)

        # Quarter of constellation
        self.x = jax.random.uniform(next(self.keygen), shape=(self.D, self.M // (2 ** self.D)))
        # self.x = jnp.array([
        #     [0.4980, -0.7779],
        #     [2.7891, 0.3160],
        #     [0.7276, 1.4065],
        #     [-0.7731, 0.4011],
        #     [0.8366, 0.9297],
        #     [-1.1283, -1.6058],
        #     [-1.4245, 0.6615],
        #     [0.7174, 2.1385],
        # ]).T
        self.logM = int(jnp.log2(self.M))
        self.bmap = (jnp.arange(self.M)[:, None] >> jnp.arange(self.m) & 1).astype(bool)
        self.bmap_idx_true = bitmap_indices(self.bmap)
        self.bmap_idx_false = bitmap_indices(~self.bmap)

    @property
    def const(self):
        flip = 1 - 2 * jnp.eye(self.D)
        x = self.x
        for d in range(self.D):
            x = jnp.append(x, (flip[d] * x.T).T, axis=1)
        return x

    def make_sequence(self, n):
        idx = jax.random.randint(next(self.keygen), shape=n, minval=0, maxval=self.M)
        return idx, jnp.take(self.const, idx)

    def plot_constellation(self):
        c = self.const.reshape((self.D // 2, 2, self.M))
        fig, axs = plt.subplots(1, self.D // 2)
        if self.D // 2 == 1:
            axs = [axs]
        for i, ax in enumerate(axs):
            ax.scatter(c[i, 0], c[i, 1])
            ax.set_aspect('equal', 'box')
            ax.grid()
        fig.tight_layout()
        return plt


class ConstellationShaper2D(ConstellationShaper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, dimensions=2, **kwargs)

    def monte_carlo_gmi(self, n=1e6):
        x = self.const
        x /= jnp.sqrt(jnp.mean(x ** 2, axis=0))
        # normalise to sigma = 1;
        x /= 10 ** (-self.SNR / 20)

        i = jnp.sum(jax.random.uniform(next(self.keygen), shape=(int(n),)) > 1, axis=1)
        y = x[i] + jax.random.uniform(next(self.keygen), shape=(int(n),))
        b = self.bmap[i]

        # squared euclidean distance
        d = (y - x.T) ** 2

        # unnormalised symbol probabilities
        ll = jnp.exp(-d / 2 * jnp.log(1 / self.M))

        # bit log likelihood ratios
        llr = jnp.log(ll @ self.bmap) - jnp.log(ll @ ~self.bmap)

        # GMI is the transmitted entropy minus the information loss
        gmi = self.m - jnp.sum(jnp.mean(jnp.log1p(jnp.exp((1 - 2 * b) * llr)))) / jnp.log(2)
        return gmi

    def gmi(self, L=10):
        input_scaling = jnp.sqrt(jnp.mean(jnp.sum(self.const ** 2, axis=0) / (self.D / 2)))
        sigma_z = 10 ** (-self.SNR / 20)
        x = self.const
        x /= input_scaling
        x /= sigma_z
        pos_idx = jnp.ones(self.M)
        sym_bits = jnp.zeros(self.m)
        for d in range(self.D):
            sym_bit = jnp.argmax(jnp.abs(x[d] @ self.bmap))
            flip = jnp.ones((self.D, 1))
            flip = flip.at[d].set(-1)
            idx = self.bmap[:, sym_bit] == 0
            comp = jnp.sum(jnp.abs(x[:, idx] - flip * x[:, ~idx]) ** 2, axis=0) < np.spacing(2 * self.M / sigma_z)
            if jnp.alltrue(comp):
                pos_idx = pos_idx.at[~idx].set(0)
                sym_bits = sym_bits.at[sym_bit].set(d)
        pos_idx = jnp.nonzero(pos_idx)[0]
        xi, al = hermgauss(L)
        z, alpha = self._grid_nD(xi, al)

        def _sum_i_fun(i):
            dij = x[:, i, None] - x[:, jnp.arange(self.M) != i]
            mapi = (self.bmap == self.bmap[i]).at[i].set(False)
            exp_n = jnp.exp(z.T @ (-2 * dij) - jnp.sum(dij ** 2, axis=0)).T
            sum_j = jnp.sum(exp_n, axis=0)
            _sum_i = alpha @ jnp.log1p(sum_j)

            # Gradient
            # den = exp_n / (1 + sum_j)
            # temp_j = -2 * alpha @ (den * (z + dij))

            def _sum_p_fun(k):
                exp_k = exp_n[mapi[jnp.arange(self.M) != i, k]]
                sum_j2 = jnp.sum(exp_k, axis=0)
                return alpha @ jnp.log1p(sum_j2)

            _sum_p = jax.lax.map(_sum_p_fun, jnp.arange(self.m))
            return _sum_i, _sum_p

        with jax.disable_jit():
            sum_i, sum_p = jax.lax.map(_sum_i_fun, pos_idx)

        GMI = (
                jnp.log2(self.M) -
                self.m / pos_idx.size / jnp.pi ** (self.D / 2) * jnp.sum(sum_i) / jnp.log(2) +
                1 / pos_idx.size / jnp.pi ** (self.D / 2) * jnp.sum(sum_p) / jnp.log(2)
        )
        return GMI

    def _grid_nD(self, xi, al):
        L = len(xi)
        z = xi
        alpha = al
        for d in range(1, self.D):
            z = jnp.vstack([jnp.tile(z, L), jnp.kron(xi, jnp.ones(L ** d))])
            alpha = jnp.tile(alpha, L) * jnp.kron(al, jnp.ones(L ** d))
        return z, alpha

    def gmi_max_log(self, const: jnp.ndarray, rx: jnp.ndarray, tx_bits: jnp.ndarray) -> float:
        """
        Computes the generalized mutual information (GMI) using the max-log approximation.
        """
        a = (abs(const)**2).mean() * self.D
        const /= a
        # rx /= a

        sigma = 10 ** (-self.SNR / 20)
        # Compute the squared distance between the received and constellation points
        squared_distance = ((rx[:, None, :] - const.T[None, :, :]) ** 2).sum(axis=-1)

        # Compute the symbol likelihood based on the squared distance and the variance sigma
        symbol_log_likelihood = -squared_distance / sigma ** 2

        # Compute the log likelihood ratios for each sample
        # max_true = jnp.max(jnp.stack([symbol_log_likelihood[:, b] for b in self.bmap.T], axis=-1), axis=1)
        # max_false = jnp.max(jnp.stack([symbol_log_likelihood[:, ~b] for b in self.bmap.T], axis=-1), axis=1)
        max_true = jnp.max(symbol_log_likelihood[:, self.bmap_idx_true], axis=1)
        max_false = jnp.max(symbol_log_likelihood[:, self.bmap_idx_false], axis=1)
        log_likelihood_ratios = max_true - max_false

        # Compute the information loss for each sample
        information_loss = jnp.mean(jnp.log1p(jnp.exp((1 - 2 * tx_bits) * log_likelihood_ratios)), axis=0)

        # the GMI is the
        return (1 - information_loss / jnp.log(2)).sum()

    def relabel(self, x):
        def _gray(o):
            j = jnp.arange(0, o, dtype=int)
            return j ^ (j >> 1)

        def _maprecursive(_x, _m):
            if _x.shape[0] == 1:
                return jnp.zeros(_x.size).at[jnp.argsort(_x)].set(_gray(2 ** _m[0]))
            else:
                _mmap = jnp.zeros(((2 ** _m.sum()).astype(int),), dtype=int)
                mmap_d = _gray(2 ** _m[0])
                argsort = jnp.lexsort(jnp.flipud(_x))
                for i in jnp.arange(2 ** _m[0]):
                    idx = argsort[(jnp.arange(2 ** _m[1:].sum()) + i * 2 ** _m[1:].sum()).astype(int)]
                    _mmap = _mmap.at[idx].set(2 ** _m[1:].sum() * mmap_d[i] + _maprecursive(_x[1:, idx], _m[1:]))
                return _mmap

        m = jnp.floor(self.m / self.D) * jnp.ones((self.D,))
        mlow = int(self.m - m.sum())
        m = m.at[-mlow:].set(m[-mlow:] + 1).astype(int)
        mmap = _maprecursive(x, m)
        # if abs(jnp.linalg.det(jnp.eye(self.M)[mmap])) != 1:
        #     raise ValueError('Failed to relabel the constellation points.')
        return jnp.take(x, mmap, axis=-1)


def rms(const: jnp.ndarray) -> float:
    """
    Computes the root-mean-square (RMS) of the constellation points.
    """
    return jnp.sqrt(jnp.square(const).sum(axis=-1)).mean()


def main():
    cs = ConstellationShaper2D(64, snr=10)
    const = jnp.array([
        [1.4143,    0.2813],
        [0.9713,    0.1932],
        [0.3584,    0.0713],
        [0.6724,    0.1337],
        [1.1990,    0.8011],
        [0.8235,    0.5502],
        [0.3038,    0.2030],
        [0.5700,    0.3809],
        [0.2813,    1.4143],
        [0.1932,    0.9713],
        [0.0713,    0.3584],
        [0.1337,    0.6724],
        [0.8011,    1.1990],
        [0.5502,    0.8235],
        [0.2030,    0.3038],
        [0.3809,    0.5700],
        [- 1.4143,    0.2813],
        [- 0.9713,    0.1932],
        [- 0.3584,    0.0713],
        [- 0.6724,    0.1337],
        [- 1.1990,    0.8011],
        [- 0.8235,    0.5502],
        [- 0.3038,    0.2030],
        [- 0.5700,    0.3809],
        [- 0.2813,    1.4143],
        [- 0.1932,    0.9713],
        [- 0.0713,    0.3584],
        [- 0.1337,    0.6724],
        [- 0.8011,    1.1990],
        [- 0.5502,    0.8235],
        [- 0.2030,    0.3038],
        [- 0.3809,    0.5700],
        [1.4143, - 0.2813],
        [0.9713, - 0.1932],
        [0.3584, - 0.0713],
        [0.6724, - 0.1337],
        [1.1990, - 0.8011],
        [0.8235, - 0.5502],
        [0.3038, - 0.2030],
        [0.5700, - 0.3809],
        [0.2813, - 1.4143],
        [0.1932, - 0.9713],
        [0.0713, - 0.3584],
        [0.1337, - 0.6724],
        [0.8011, - 1.1990],
        [0.5502, - 0.8235],
        [0.2030, - 0.3038],
        [0.3809, - 0.5700],
        [- 1.4143, - 0.2813],
        [- 0.9713, - 0.1932],
        [- 0.3584, - 0.0713],
        [- 0.6724, - 0.1337],
        [- 1.1990, - 0.8011],
        [- 0.8235, - 0.5502],
        [- 0.3038, - 0.2030],
        [- 0.5700, - 0.3809],
        [- 0.2813, - 1.4143],
        [- 0.1932, - 0.9713],
        [- 0.0713, - 0.3584],
        [- 0.1337, - 0.6724],
        [- 0.8011, - 1.1990],
        [- 0.5502, - 0.8235],
        [- 0.2030, - 0.3038],
        [- 0.3809, - 0.5700],
    ]).T

    # const = cs.relabel(cs.const)
    # const = cs.const
    # const = alphabet.get_alphabet('64QAM')
    # const = jnp.array([const.real, const.imag])
    # cs.plot_constellation().show()

    seq = 2**10
    sigma = 10 ** (-cs.SNR / 20)

    tx = jnp.tile(const.T, (seq, 1))
    tx_bits = jnp.tile(cs.bmap, (seq, 1))
    rx = tx + cs.D ** -.5 * jax.random.normal(next(cs.keygen), shape=tx.shape) * sigma
    print(f"Initial GMI={cs.gmi_max_log(const, rx, tx_bits):.3f} bit/{cs.D}D @{cs.SNR:.1f}dB")
    gmi_fun = jax.value_and_grad(cs.gmi_max_log)

    # plt.figure()
    # plt.hist2d(rx[:, 0], rx[:, 1], bins=128, cmin=1, range=np.sqrt(2) * np.array([[-1, 1], [-1, 1]]))
    # plt.plot(const[0], const[1], 'x', c='magenta')
    # plt.axis('equal')
    # plt.show()

    # plt.quiver(cs.const[0], cs.const[1], gmi_grad[0], gmi_grad[1], width=0.002, headwidth=4, color='b')
    # plt.plot(cs.const[0], cs.const[1], 'o')
    # plt.grid()
    # plt.axis('equal')
    # plt.show()

    fig = plt.figure()
    lim = 1 + 2 ** -.5
    ax = plt.axes(xlim=(-lim, lim), ylim=(-lim, lim))
    # ax.grid()
    ax.set_aspect('equal', 'box')
    ax.set_title(f"Starting..")

    def _make_hist(_rx):
        d, _, _ = np.histogram2d(_rx[:, 0], _rx[:, 1], bins=128, range=np.array([[-lim, lim], [-lim, lim]]))
        # d[d < 1] = np.nan
        return d.T

    im = plt.imshow(_make_hist(rx), interpolation='gaussian', extent=[-lim, lim, -lim, lim], origin='lower', cmap='kindlmann')
    line, = ax.plot(const[0], const[1], '.', c='white')

    # const = cs.const #/ rms(cs.const)
    # const /= rms(const)

    def init():
        line.set_data(const[0], const[1])
        return line,

    LEARNING_RATE = 0.2
    GMI = -np.inf
    FA = 0

    def animate(i):
        nonlocal const, GMI, FA, LEARNING_RATE
        tx = jnp.tile(const.T, (seq, 1))
        rx = tx + cs.D ** -.5 * jax.random.normal(next(cs.keygen), shape=tx.shape) * sigma
        gmi, gmi_grad = gmi_fun(const, rx, tx_bits)


        # const /= (abs(const)**2).mean() * cs.D
        if gmi > GMI and FA < 10:
            GMI = gmi
            const += LEARNING_RATE * gmi_grad
            FA = 0
        else:
            FA += 1
        if FA > 100:
            LEARNING_RATE = 5
        elif FA > 60:
            LEARNING_RATE = 4
        elif FA > 30:
            LEARNING_RATE = 3
        elif FA > 10:
            LEARNING_RATE = 2
        elif LEARNING_RATE != 0.2:
            LEARNING_RATE = 0.7

        ax.set_title(f"GMI={GMI:.3f} RMS={rms(const):.2f} $||\\nabla GMI||$={jnp.sqrt((gmi_grad ** 2).sum()):.2f} LR={LEARNING_RATE:.1f}")
        line.set_data(const[0], const[1])
        im.set_data(_make_hist(rx))
        return line, im,

    # plt.quiver(cs.const[0], cs.const[1], gmi_grad[0], gmi_grad[1], width=0.002, headwidth=4, color='b')

    anim = FuncAnimation(fig, animate, init_func=init, frames=200, interval=20, blit=False)
    plt.show()
    # anim.save('anim.mp4', writer='ffmpeg', fps=30)


if __name__ == '__main__':
    main()
