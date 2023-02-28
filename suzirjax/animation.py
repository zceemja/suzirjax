import os.path
import time

import jax.random
from jax import numpy as jnp

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from rich.progress import Progress
from rich import print

from channels import Channel, AWGNChannel, PCAWGNChannel, ChalmersQAMpy
from gui_helpers import Connector
from modulation import get_modulation
from optimiser import Optimiser, AdamOpt, GradientDescentOpt, RMSPropOpt, YogiOpt
from utils import register_cmaps
import numpy as np

register_cmaps()


class Animator:
    # LIM = 1 + 2 ** -.5
    LIM = 2.39
    BINS = 128

    def __init__(self, data: Connector, const, channel: Channel, optimiser: Optimiser, seq_len=2 ** 12, width=1920):
        self.data = data
        self.figure, (self.ax, self.ax2) = plt.subplots(1, 2, figsize=(16, 9), dpi=width // 16,
                                                        gridspec_kw={'width_ratios': [3, 2]})

        self.ax.set_ylim(ymin=-self.LIM, ymax=self.LIM)
        self.ax.set_xlim(xmin=-self.LIM, xmax=self.LIM)
        self.ax.set_aspect('equal', 'box')

        self.figure.patch.set_facecolor('black')
        self.ax2.set_facecolor('black')
        self.ax2.tick_params(axis='x', labelcolor='white')
        self.ax2.tick_params(axis='y', labelcolor='white')
        self.ax2.spines['left'].set_color('white')
        self.ax2.spines['bottom'].set_color('white')
        self.ax2.set_title('GMI over time', color='white')
        self.ax2.grid(axis='y')

        self.channel = channel
        self.optimiser = optimiser

        self.const = const
        self.seq_len = seq_len
        self.mod_points = const.shape[0]
        self.key = jax.random.PRNGKey(time.time_ns())
        self.task = None
        self.progress = None
        self.results = {'snr': [], 'gmi': []}
        self.text = None
        self.plot_snr = False

        channel.data['linewidth'] = 10000
        channel.data['fs'] = 1

        self.im = None
        self.im2 = None
        self.const_plt = None

    def _make_hist(self, rx):
        d, _, _ = jnp.histogram2d(rx[:, 0], rx[:, 1], bins=self.BINS, density=True,
                                  range=np.array([[-self.LIM, self.LIM], [-self.LIM, self.LIM]]))
        d *= self.BINS ** 2 / jnp.sum(d) * 25
        return d.T

    def _update(self, frame):
        # if 400 <= frame <= 430:
        #     self.channel.data['ase'] -= 0.25
        # if 700 <= frame <= 730:
        #     self.channel.data['ase'] += 0.25

        """ SIMULATION """
        self.key, key2 = jax.random.split(self.key, 2)
        rx, snr = self.channel.propagate(self.const, key2, self.seq_len)
        tx_seq, _ = self.channel.get_tx(self.const, key2, self.seq_len)

        if frame > 0:
            print(f'X={self.const.shape} RX={rx.shape} TX={tx_seq[0].shape} SNR={snr}')
            c = self.optimiser.update(jnp.array([self.const.real, self.const.imag]).T, rx, snr, tx_seq[0])
            self.const = c[:, 0] + 1j * c[:, 1]

        """ ANIMATION """
        self.results['snr'].append(snr)
        self.results['gmi'].append(self.optimiser.data["gmi"] if self.optimiser.data["gmi"] > 0 else np.nan)
        gmi = np.array(self.results['gmi'])
        gmi = gmi[np.isfinite(gmi)]
        snr = np.array(self.results['snr'])
        snr = snr[np.isfinite(snr)]

        if self.im is None:
            self.ax.set_title('-', color='white')
            self.im = self.ax.imshow(
                self._make_hist(rx), vmin=0, vmax=255,
                interpolation='gaussian', extent=[-self.LIM, self.LIM, -self.LIM, self.LIM],
                origin='lower', cmap='sillekens'
            )
            self.im2, = self.ax2.plot(np.arange(len(gmi)), gmi, c='tab:blue', lw=1.6)
            if self.plot_snr:
                self.im3, = self.ax2.plot(np.arange(len(snr)), snr, c='tab:orange', lw=1.6)
            self.ax2.set_xlabel('Iterations', color='white')
            self.ax2.set_ylabel('GMI (bit/2D symbol)', color='white')
            self.const_plt, = self.ax.plot(self.const.real, self.const.imag, '.', c='magenta')
            if self.text is not None:
                self.ax.text(0, 0.1, self.text, color='white', horizontalalignment='left',
                             verticalalignment='top', transform=self.ax.transAxes)
            self.figure.tight_layout()

        self.im.set_data(self._make_hist(rx))
        if len(gmi) > 0:
            self.im2.set_data(np.arange(len(gmi)), gmi)
            self.ax2.set_ylim([np.round(np.min(gmi) - 0.1, 1), np.round(np.max(gmi) + 0.1, 1)])
        if len(snr) > 0 and self.plot_snr:
            self.im3.set_data(np.arange(len(snr)), snr)

        self.const_plt.set_data(self.const.real, self.const.imag)
        self.ax.title.set_text(f'GMI={self.optimiser.data["gmi"]:0.3f} bit/2D symbol')  # SNR={snr:.3f}dB
        if self.task is not None:
            self.progress.update(self.task, advance=1)
        return self.const_plt, self.im, self.im2

    def _init(self):
        return self._update(-1)

    def plot_snr_gmi(self):
        snr = np.array(self.results['snr'])
        gmi = np.array(self.results['gmi'])
        x = np.arange(snr.size)

        fig, ax1 = plt.subplots()
        ax1.set_xlabel('Iterations')
        ax1.set_ylabel('GMI (bit/2Dsym)')
        ax1.plot(x, gmi, color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        # ax2 = ax1.twinx()
        # ax2.set_ylabel('SNR (dB)')
        # ax2.plot(x, snr, color='tab:orange')
        # ax2.tick_params(axis='y', labelcolor='tab:orange')
        ax1.grid()

        return plt

    def animate(self, dirname, frames=300, name=''):
        dirname = os.path.abspath(os.path.expanduser(dirname))
        fname = f"{self.data['mod_name']}-SNR_{self.channel.data['ase']}dB-{name}"
        i = 0
        while os.path.exists(os.path.join(dirname, f'{fname}-{i:02d}.mp4')):
            i += 1
        full_fname = os.path.join(dirname, f'{fname}-{i:02d}')
        print(f"Processing [red]{fname}-{i:02d}[/red]")

        self.ax2.set_xlim([0, frames])
        with Progress() as progress:
            self.progress = progress
            self.task = progress.add_task("Simulating", total=frames)
            anim = FuncAnimation(self.figure, self._update, init_func=self._init, frames=frames, interval=0, blit=True)
            anim.save(full_fname + '.mp4', fps=30)
            progress.remove_task(self.task)
            print(f"Saving [red]{full_fname}.mp4[/red]")
        np.savez_compressed(full_fname + '.npz', snr=np.array(self.results['snr']), gmi=np.array(self.results['gmi']))
        self.plot_snr_gmi().savefig(full_fname + '.png')

    def animate_now(self, frames):
        anim = FuncAnimation(self.figure, self._update, init_func=self._init, frames=frames, interval=20, repeat=True,
                             blit=False)
        plt.show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        prog='Suzirjax Animator',
        description='Create animations for constellation shaping'
    )
    parser.add_argument('constellation', help='Constellation formats')
    parser.add_argument('-o', '--output', default='.', help='Output directory')
    parser.add_argument('-s', '--seq_size', default=19, type=int, choices=range(4, 23),
                        help='Simulation sequence size, value of 2 ** x')
    parser.add_argument('-w', '--window', action='store_true',
                        help='Show GUI window instead of just saving to file')
    parser.add_argument('-r', '--resolution', type=int, default=480,
                        help='Graphic resolution in pixels')
    parser.add_argument('-f', '--frames', type=int, default=100, help='Frames to simulate')
    parser.add_argument('-n', '--snr', type=float, default=12, help='Set target SNR')
    parser.add_argument('-O', '--optimiser', choices=['none', 'adam', 'gd', 'rmsprop', 'yogi'], default='adam')
    parser.add_argument('-l', '--lr', type=float, default=1e-3, help='Optimiser Learning Rate')
    parser.add_argument('-D', '--decrease', action='store_true',
                        help='Ignore results if GMI drops from previous interation')
    parser.add_argument('-C', '--channel', default='awgn', help='Set channel')

    args = parser.parse_args()
    print("Starting setup..")
    data = Connector()
    data['mod_name'] = args.constellation
    const = get_modulation(data['mod_name'])
    data['mod_points'] = const.shape[0]

    if args.channel == 'awgn':
        channel = AWGNChannel(None)
        channel.data['ase'] = args.snr
    elif args.channel == 'pcawgn':
        channel = PCAWGNChannel(None)
        channel.data['ase'] = args.snr
    elif args.channel == 'qampy':
        channel = ChalmersQAMpy(None)
        channel.data['ase'] = args.snr
        channel.data['linewidth'] = 10  # kHz
        channel.data['fb'] = 15  # GHz
        channel.data['ntaps'] = 31
    else:
        raise ValueError("No channel: " + args.channel)

    if args.optimiser == 'none':
        optimiser = Optimiser(data)
    elif args.optimiser == 'adam':
        optimiser = AdamOpt(data, learning_rate=args.lr)
    elif args.optimiser == 'rmsprop':
        optimiser = RMSPropOpt(data, learning_rate=args.lr)
    elif args.optimiser == 'yogi':
        optimiser = YogiOpt(data, learning_rate=args.lr)
    elif args.optimiser == 'gd':
        optimiser = GradientDescentOpt(data)
        optimiser.data['learning_rate'] = float(np.log10(args.lr))
    else:
        raise ValueError("No optimiser: " + args.optimiser)
    optimiser.data['allow_decrease'] = not args.decrease

    if args.window:
        # Load before frame instance
        matplotlib.use('Qt5Agg')

    anim = Animator(data, const, channel, optimiser, seq_len=2 ** args.seq_size, width=args.resolution)
    anim.text = (
        f"Constellation points: {data['mod_points']}\n"
        f"Simulation points: 2^{args.seq_size}\n"
        f"Channel model: {channel.NAME}\n"
        f"Channel SNR: {channel.data['ase']}dB\n"
        f"Optimiser: {optimiser.NAME}\n"
        f"Optimiser learning rate: {args.lr}\n"
        f"UCL ONG Suzirjax (github.com/zceemja/suzirjax)"
    )
    if args.window:
        anim.animate_now(args.frames)
    else:
        anim.animate(args.output, args.frames, f'{args.channel}-{args.optimiser}')
