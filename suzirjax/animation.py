import os.path
import time

import jax.random
from jax import numpy as jnp

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from rich.progress import Progress
from rich.prompt import Confirm
from rich import print

from channels import Channel, AWGNChannel
from gui_helpers import Connector
from modulation import get_modulation
from optimiser import Optimiser, AdamOpt, GradientDescentOpt
from utils import register_cmaps
import numpy as np

register_cmaps()


class Animator:
    LIM = 1 + 2 ** -.5
    BINS = 256

    def __init__(self, data: Connector, const, channel: Channel, optimiser: Optimiser, seq_len=2**12):
        self.data = data
        self.figure, self.ax = plt.subplots(figsize=(5, 5), dpi=1080//5)
        self.ax.set_ylim(ymin=-self.LIM, ymax=self.LIM)
        self.ax.set_xlim(xmin=-self.LIM, xmax=self.LIM)
        self.ax.set_aspect('equal', 'box')

        self.channel = channel
        self.optimiser = optimiser

        self.const = const
        self.seq_len = seq_len
        self.mod_points = const.shape[0]
        self.key = jax.random.PRNGKey(time.time_ns())
        self.key, key2 = jax.random.split(self.key, 2)
        self.task = None
        self.progress = None
        channel.data['snr'] = 22
        optimiser.data['learning_rate'] = -.8

        rx, snr = channel.propagate(self.const, key2, self.seq_len)
        tx_seq, _ = self.channel.get_tx(self.const, key2, self.seq_len)
        self.optimiser.update(jnp.array([self.const.real, self.const.imag]).T, rx, snr, tx_seq[0])
        self.ax.set_title(f'SNR={snr:.3f}dB GMI={self.optimiser.data["gmi"]:.3f}')

        self.im = self.ax.imshow(
            self._make_hist(rx), vmin=0, vmax=255,
            interpolation='gaussian', extent=[-self.LIM, self.LIM, -self.LIM, self.LIM],
            origin='lower', cmap='sillekens'
        )
        self.const_plt, = self.ax.plot(self.const.real, self.const.imag, '.', c='magenta')

    def _make_hist(self, rx):
        d, _, _ = jnp.histogram2d(rx[:, 0], rx[:, 1], bins=self.BINS, density=True,
                                  range=np.array([[-self.LIM, self.LIM], [-self.LIM, self.LIM]]))
        # d[d < 1] = np.nan
        # d /= jnp.log2(rx.shape[0]) * self.mod_points
        # d /= d.max() * 0.85
        d *= self.BINS ** 2 / jnp.sum(d) * 25
        # d /= d.max() / 255
        # d *= jnp.log2(rx.shape[0]) / 1.9
        return d.T

    def _update(self, frame):
        self.key, key2 = jax.random.split(self.key, 2)
        rx, snr = self.channel.propagate(self.const, key2, self.seq_len)
        tx_seq, _ = self.channel.get_tx(self.const, key2, self.seq_len)
        c = self.optimiser.update(jnp.array([self.const.real, self.const.imag]).T, rx, snr, tx_seq[0])
        self.const = c[:, 0] + 1j * c[:, 1]

        self.im.set_data(self._make_hist(rx))
        self.const_plt.set_data(self.const.real, self.const.imag)
        self.ax.title.set_text(f'SNR={snr:.3f}dB GMI={self.optimiser.data["gmi"]:.3f}')
        if self.task is not None:
            self.progress.update(self.task, advance=1)
        return self.const_plt, self.im

    def _init(self):
        return self.const_plt, self.im

    def animate(self, frames=300):
        fname = f"{self.data['mod_name']}_adam_awgn_{self.channel.data['snr']}.mp4"
        if os.path.exists(fname):
            if Confirm(f'File [red]{fname}[/red] exists, overwrite?'):
                os.remove(fname)
            else:
                exit(0)
        with Progress() as progress:
            self.progress = progress
            self.task = progress.add_task("Simulating", total=frames)
            anim = FuncAnimation(self.figure, self._update, init_func=self._init, frames=frames, interval=0, blit=True)
            anim.save(fname, fps=30)
        print(f"Saving to [red]{fname}[/red]..")

    def animate_now(self):
        matplotlib.use('Qt5Agg')
        anim = FuncAnimation(self.figure, self._update, init_func=self._init, frames=0, interval=0, blit=True)
        plt.show()


if __name__ == '__main__':
    print("Starting setup..")
    data = Connector()
    data['mod_name'] = '64APSK'
    const = get_modulation(data['mod_name'])
    data['mod_points'] = const.shape[0]
    channel = AWGNChannel(None)
    optimiser = AdamOpt(data, learning_rate=1e-1)
    Animator(data, const, channel, optimiser, seq_len=2**17).animate(600)
