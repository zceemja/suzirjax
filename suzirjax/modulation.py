import os.path
from typing import Any, Union

import jax
import numpy as np
import re
from jax import numpy as jnp
from jax.tree_util import register_pytree_node
from numpy import ndarray
from scipy.io import loadmat

from suzirjax import utils

__MOD = {}

MODULATIONS = ["ASK", "PAM", "PSK", "QAM", "APSK"]


class Modulation:
    """ Modulation format, always with uniform power """
    def __init__(self, array, modes=2):
        self.format: jnp.ndarray = jnp.array(array)
        if self.format.ndim == 1:
            self.format = self.format[None, :]
        if self.format.shape[0] == 1 and modes > 1:
            self.format = jnp.tile(self.format, (modes, 1))
        self.format /= jnp.sqrt(jnp.mean(jnp.abs(self.format) ** 2 / (self.dim / 2), axis=1))[:, None]

    @property
    def complex(self):
        """ Return in complex format of [[modes...], [points...]]"""
        return self.format

    @property
    def real(self):
        """ Return real values in format of [[modes...], [points...]]"""
        return self.format.real

    @property
    def imag(self):
        """ Return imaginary values in format of [[modes...], [points...]]"""
        return self.format.imag

    @property
    def regular(self):
        """ Return in real-value format of [[points...], [mode 0 real, mode 0 imag, mode 1 real ...]]"""
        return jnp.array([self.format.real, self.format.imag]).transpose((2, 1, 0)).reshape(self.points, self.dim)

    @property
    def points(self):
        """ Constellation cardinality """
        return self.format.shape[1]

    @property
    def modes(self):
        """ Number of modes (polarisations) """
        return self.format.shape[0]

    @property
    def dim(self):
        """ Constellation dimension """
        return self.format.shape[0] * 2

    def copy(self) -> 'Modulation':
        return Modulation(self.format.copy())

    @classmethod
    def special_flatten(cls, v):
        return v.format, None

    @classmethod
    def special_unflatten(cls, aux_data, children):
        return Modulation(children)


register_pytree_node(Modulation, Modulation.special_flatten, Modulation.special_unflatten)


def graycode(o):
    j = jnp.arange(o, dtype=int)
    return j ^ (j >> 1)


def relabel(x: jnp.ndarray):
    d = x.shape[0]
    n = int(jnp.log2(x.shape[1]))

    def _map_recursive(_x, _m):
        if _x.shape[0] == 1:
            return jnp.zeros(_x.size, dtype=int).at[jnp.argsort(_x)].set(graycode(2 ** _m[0]))
        else:
            _mmap = jnp.zeros(((2 ** _m.sum()).astype(int),), dtype=int)
            mmap_d = graycode(2 ** _m[0])
            argsort = jnp.lexsort(jnp.flipud(_x))
            for i in jnp.arange(2 ** _m[0]):
                idx = argsort[(jnp.arange(2 ** _m[1:].sum()) + i * 2 ** _m[1:].sum()).astype(int)]
                _mmap = _mmap.at[idx].set(2 ** _m[1:].sum() * mmap_d[i] + _map_recursive(_x[1:, idx], _m[1:]))
            return _mmap

    m = jnp.floor(n / d) * jnp.ones((d,))
    mlow = int(n - m.sum())
    if mlow > 0:
        m = m.at[-mlow:].set(m[-mlow:] + 1)
    mmap = _map_recursive(x, m.astype(int))
    # if abs(jnp.linalg.det(jnp.eye(self.M)[mmap])) != 1:
    #     raise ValueError('Failed to relabel the constellation points.')
    return jnp.take(x, mmap, axis=-1)


def make_qam(order: int):
    # Can't be asked to properly implement this
    # File has precomputed output from matlab qammod function
    if len(__MOD) == 0:
        mods = np.load(utils.get_resource('mod.npz'))
        for i in range(2, 11):
            __MOD[f'{1 << i}QAM'] = mods[f'qam{1 << i}']
    return __MOD[str(order) + 'QAM']


def make_apsk(order: int):
    m = int(np.log2(order))
    mp = -(-m // 2)
    ma = m // 2
    if ma == mp:
        mp += 1
        ma -= 1
    mp = 1 << mp
    ma = 1 << ma

    ba = np.zeros((ma, 1))
    ba[graycode(ma), 0] = np.arange(ma, dtype=int)

    bp = np.zeros((mp, 1))
    bp[graycode(mp), 0] = np.arange(mp, dtype=int)

    ua = (0.5 + ba) / ma
    up = (0.5 + bp) / mp
    c = np.sqrt(-np.log(ua)).T * np.exp(2j * np.pi * up)
    return c.flatten()


def make_psk(order: int):
    c = np.exp(1j * np.linspace(-np.pi, np.pi, order + 1))[:-1]
    return c[graycode(order)]


def make_ask(order: int):
    c = np.linspace(0, 1, num=order)
    return c[graycode(order)]


def make_pam(order: int):
    c = np.linspace(-1, 1, num=order)
    return c[graycode(order)]


def get_modulation(name: str, unit_power=True) -> Modulation:
    """
    Returns modulation format
    """
    name = name.upper()
    redix = re.compile(r'(\d+)-?([A-Z]{3,4})')
    if (match := redix.match(name)) is None:
        raise ValueError(f'Invalid modulation name "{name}"')

    points = int(match.groups()[0])
    name = match.groups()[1]
    if name == 'APSK':
        c = make_apsk(points)
    elif name == 'PSK':
        c = make_psk(points)
    elif name == 'ASK':
        c = make_ask(points) + 0j
    elif name == 'PAM':
        c = make_pam(points) + 0j
    elif name == 'QAM':
        c = make_qam(points)
    elif name == 'RAND':
        c = np.random.uniform(-1., 1., points) + 1j * np.random.uniform(-1., 1., points)
        c = relabel(np.stack([c.real, c.imag]))
        c = c[0, :] + 1j * c[1, :]
    else:
        raise ValueError(f"Unknown modulation name '{name}'")
    # c += np.random.uniform(-1., 1., points) * 1e-9 + 1j * np.random.uniform(-1., 1., points) * 1e-9  # break symmetry
    return Modulation(c)


def load_from_file(file_name) -> Union[Modulation, None]:
    if file_name is None or not os.path.exists(file_name):
        return None
    if file_name.lower().endswith('.npz'):
        data = np.load(file_name)
    elif file_name.lower().endswith('.mat'):
        data = loadmat(file_name)
    else:
        return None
    for key in data.keys():
        array = data[key]
        if not isinstance(array, np.ndarray):
            continue
        if array.dtype.kind == 'f' and array.ndim == 2:
            if array.shape[0] != 2:
                array = array.T
            if array.shape[0] != 2:
                continue
            array = np.array([array[0] + array[1] * 1j])
        elif array.dtype.kind == 'c':
            pass
        else:
            continue
        if array.shape[0] & (array.shape[0] - 1) != 0:  # if not power of 2
            continue
        return Modulation(array)
    return None
