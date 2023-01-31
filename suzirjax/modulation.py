import numpy as np
import re
from jax import numpy as jnp
import utils

__MOD = {}

MODULATIONS = ["ASK", "PAM", "PSK", "QAM", "APSK"]


def graycode(o):
    j = jnp.arange(o, dtype=int)
    return j ^ (j >> 1)


def relabel(x: jnp.ndarray):
    d = x.shape[0]
    n = int(jnp.log2(x.shape[1] * 2 ** d))

    def _maprecursive(_x, _m):
        if _x.shape[0] == 1:
            return jnp.zeros(_x.size).at[jnp.argsort(_x)].set(graycode(2 ** _m[0]))
        else:
            _mmap = jnp.zeros(((2 ** _m.sum()).astype(int),), dtype=int)
            mmap_d = graycode(2 ** _m[0])
            argsort = jnp.lexsort(jnp.flipud(_x))
            for i in jnp.arange(2 ** _m[0]):
                idx = argsort[(jnp.arange(2 ** _m[1:].sum()) + i * 2 ** _m[1:].sum()).astype(int)]
                _mmap = _mmap.at[idx].set(2 ** _m[1:].sum() * mmap_d[i] + _maprecursive(_x[1:, idx], _m[1:]))
            return _mmap

    m = jnp.floor(n / d) * jnp.ones((d,))
    mlow = int(n - m.sum())
    m = m.at[-mlow:].set(m[-mlow:] + 1).astype(int)
    mmap = _maprecursive(x, m)
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


def get_modulation(name: str, unit_power=True) -> np.ndarray:
    """
    Returns modulation alphabet
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
        c = make_ask(points)
    elif name == 'PAM':
        c = make_pam(points)
    elif name == 'QAM':
        c = make_qam(points)
    else:
        raise ValueError(f"Unknown modulation name '{name}'")
    if unit_power:
        c = c / c.real.max()
    return c
