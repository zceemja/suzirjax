from functools import partial

import jax
from jax import numpy as jnp

from matplotlib import pyplot as plt
import numpy as np

# c = jnp.array(np.loadtxt('qam2_64.txt'))
from ong.utils import alphabet

c = alphabet.get_alphabet('64QAM')
c = jnp.array([c.real, c.imag]).T
# GMI =  3.7791 bit/2D @12dB

@jax.jit
def root_mean_square(c):
    squared_sum = jnp.sum(jnp.square(c), axis=-1)
    root = jnp.sqrt(squared_sum)
    rms = jnp.mean(root)
    return rms


def bitmap(length: int) -> jnp.ndarray:
    return jnp.bool_(jnp.arange(length)[:, jnp.newaxis] >> jnp.arange(jnp.int32(jnp.ceil(jnp.log2(length)))) & 1)


# %%
class NormalDistributionGenerator:
    def __init__(self, seed=0):
        self.key = jax.random.PRNGKey(seed)

    def __call__(self, size=(1,)):
        self.key, subkey = jax.random.split(self.key)
        return size[-1] ** (-1 / 2) * jax.random.normal(subkey, shape=size)


def generalized_mutual_information(constellation: jnp.ndarray, received: jnp.ndarray, transmitted_bits: jnp.ndarray,
                                   bitmap: jnp.ndarray, sigma: float) -> float:
    # Compute the squared distance between the received and constellation points
    squared_distance = ((received[:, None, :] - constellation[None, :, :]) ** 2).sum(axis=-1)

    # Compute the symbol likelihood based on the squared distance and the variance sigma
    symbol_likelihood = jnp.exp(-squared_distance / sigma ** 2)

    # Compute the log likelihood ratios for each sample
    log_likelihood_ratios = jnp.log(symbol_likelihood @ bitmap) - jnp.log(symbol_likelihood @ ~bitmap)

    # Compute the information loss for each sample
    information_loss = jnp.mean(jnp.log1p(jnp.exp((1 - 2 * transmitted_bits) * log_likelihood_ratios)), axis=0)

    # the GMI is the
    return (1 - information_loss / jnp.log(2)).sum()


def generalized_mutual_information_max_log(constellation: jnp.ndarray, received: jnp.ndarray,
                                           transmitted_bits: jnp.ndarray, bitmap: jnp.ndarray, sigma: float) -> float:
    # Compute the squared distance between the received and constellation points
    squared_distance = ((received[:, None, :] - constellation[None, :, :]) ** 2).sum(axis=-1)

    # Compute the symbol likelihood based on the squared distance and the variance sigma
    symbol_log_likelihood = -squared_distance / sigma ** 2

    # Compute the log likelihood ratios for each sample
    max_true = jnp.max(jnp.stack([symbol_log_likelihood[:, b] for b in bitmap.T], axis=-1), axis=1)
    max_false = jnp.max(jnp.stack([symbol_log_likelihood[:, ~b] for b in bitmap.T], axis=-1), axis=1)
    log_likelihood_ratios = max_true - max_false

    # Compute the information loss for each sample
    information_loss = jnp.mean(jnp.log1p(jnp.exp((1 - 2 * transmitted_bits) * log_likelihood_ratios)), axis=0)

    # the GMI is the
    return (1 - information_loss / jnp.log(2)).sum()


def main():
    randn = NormalDistributionGenerator(1701)

    nrep = 1024
    snr = 12

    M = c.shape[0]
    bmap = bitmap(M)
    sigma = 10 ** (-snr / 20)

    x = jnp.tile(c, (nrep, 1))
    b = jnp.tile(bmap, (nrep, 1))
    y = x + 10 ** (-snr / 20) * randn(x.shape)

    plt.figure()
    plt.hist2d(y[:, 0], y[:, 1], bins=128, cmin=1, range=np.sqrt(2) * np.array([[-1, 1], [-1, 1]]))
    plt.plot(c[:, 0], c[:, 1], 'x', c='magenta')
    plt.axis('equal')
    plt.show()

    gmi1 = generalized_mutual_information(c, y, b, bmap, sigma)
    print(f"GMI = {gmi1:.4f} bit/2D @{snr:.2f}dB")
    gmi2 = generalized_mutual_information_max_log(c, y, b, bmap, sigma)
    print(f"GMI Max-Log = {gmi2:.4f} bit/2D @{snr:.2f}dB")


if __name__ == '__main__':
    main()
