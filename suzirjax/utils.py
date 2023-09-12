import numpy as np
import matplotlib
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from suzirjax.resources import get_resource

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
    if 'kindlmann' not in matplotlib.colormaps:
        matplotlib.colormaps.register(kindlmann_cmap)
    if 'sillekens' not in matplotlib.colormaps:
        matplotlib.colormaps.register(sillekens_cmap)
