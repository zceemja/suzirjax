import numpy as np
from os import path
import matplotlib
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

_cdir = path.dirname(path.realpath(__file__))


def get_resource(name) -> str:
    return path.join(_cdir, "..", "resources", name)


# Source: https://www.kennethmoreland.com/color-advice/
_kindlmann_data = np.loadtxt(get_resource("kindlmann-table-float-0128.csv"), delimiter=",", skiprows=1)
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
sillekens_cmap = ListedColormap(np.load(get_resource("DrSillekensCMap.npy")), name='sillekens')


def register_cmaps():
    matplotlib.colormaps.register(kindlmann_cmap)
    matplotlib.colormaps.register(sillekens_cmap)
