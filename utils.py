import numpy as np
from os import path
import matplotlib
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from jax import numpy as jnp

# Source: https://www.kennethmoreland.com/color-advice/
_cdir = path.dirname(path.realpath(__file__))
_kindlmann_data = np.loadtxt(path.join(_cdir, "resources", "kindlmann-table-float-0128.csv"), delimiter=",", skiprows=1)

kindlmann_cmap = LinearSegmentedColormap(
    'kindlmann',
    {
        'red': _kindlmann_data[:, [0, 1, 1]],
        'green': _kindlmann_data[:, [0, 2, 2]],
        'blue': _kindlmann_data[:, [0, 3, 3]]
    },
    N=_kindlmann_data.shape[0]
)

# Source: uceeesi@ucl.ac.uk
sillekens_cmap = ListedColormap(np.load(path.join(_cdir, "resources", "DrSillekensCMap.npy")), name='sillekens')


def register_cmaps():
    matplotlib.colormaps.register(kindlmann_cmap)
    matplotlib.colormaps.register(sillekens_cmap)


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
