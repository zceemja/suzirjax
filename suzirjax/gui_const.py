import itertools

import numpy as np
import matplotlib.axes
from scipy.io import savemat

from suzirjax.gui_helpers import *
from suzirjax.simulation import Simulation

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *


class ConstellationCanvas(FigureCanvas):
    # LIM = 1 + 2 ** -.5  # Should match simulation.py HIST_LIM
    LIM = Simulation.HIST_LIM
    TEXT_OFFSET = 0.03
    POLS = 2
    POL_NAMES = 'X', 'Y'

    def __init__(self, data: Connector):
        super().__init__()
        self.data = data
        self.ax: List[matplotlib.axes.Axes] = self.figure.subplots(ncols=self.POLS)

        for p in range(self.POLS):
            self.ax[p].set_title(self.POL_NAMES[p], color='white')
            self.ax[p].set_ylim(ymin=-self.LIM, ymax=self.LIM)
            self.ax[p].set_xlim(xmin=-self.LIM, xmax=self.LIM)
            self.ax[p].set_aspect('equal', 'box')

            self.ax[p].set_facecolor('black')
            self.ax[p].tick_params(axis='x', labelcolor='white')
            self.ax[p].tick_params(axis='y', labelcolor='white')

            for spine in self.ax[p].spines.values():
                spine.set_color('white')

        self.figure.patch.set_facecolor('black')

        self.h_plt = None
        self.c_plt = None
        self.v_plt = None

        self.bg = None
        self.last_const = None

        self.const_text = [], []

        add_right_clk_menu(
            self,
            # make_checkable_action("Show received", self.data.bind("show_rx", True), self),
            make_checkable_action("Show constellation", self.data.bind("show_c", True), self),
            make_checkable_action("Show bitmap", self.data.bind("show_bmap", False), self),
            make_checkable_action("Show vectors", self.data.bind("show_vec", True), self),
            # ("Show received", lambda: self.data.set("show_rx", not self.data["show_rx"])),
            # ("Show constellation", lambda: self.data.set("show_c", not self.data["show_c"])),
            # ("Show bitmap", lambda: self.data.set("show_bmap", not self.data["show_bmap"])),
            None,
            ("Save figure", self.save),
            ("Copy figure", self.copy_image),
            ("Save data", lambda: self.save_data(const=self.last_const)),
            parent=self,
        )

    def _redraw(self):
        if self.h_plt is not None and self.data['show_rx']:
            for p in range(self.POLS):
                self.ax[p].draw_artist(self.h_plt[p])
        if self.c_plt is not None and self.data['show_c']:
            for p in range(self.POLS):
                self.ax[p].draw_artist(self.c_plt[p])
        if self.data['show_bmap']:
            for p in range(self.POLS):
                for text in self.const_text[p]:
                    self.ax[p].draw_artist(text)
        if self.v_plt is not None and self.data['show_vec']:
            for p in range(self.POLS):
                self.ax[p].draw_artist(self.v_plt[p])
        self.figure.canvas.blit(self.figure.bbox)
        self.figure.canvas.flush_events()

    def _init_quiver(self, const, gmi_grad):
        if gmi_grad is None or gmi_grad.shape[0] != const.shape[1]:
            gmi_grad = np.zeros((const.shape[1], 2))
        # gmi_grad /= gmi_grad.max()
        self.v_plt = [
            self.ax[p].quiver(const[p].real, const[p].imag, gmi_grad[:, 0], gmi_grad[:, 1],
                              color='aqua', width=0.003, animated=True)
            for p in range(self.POLS)
        ]

    @property
    def _const_texts(self):
        return itertools.chain(*self.const_text)

    def _update_points(self):
        m = self.last_const.shape[1]
        format_str = f'{{0:0{int(np.log2(m))}b}}'
        self.bit_text = [format_str.format(i) for i in range(m)]
        for text in self._const_texts:
            text.remove()
        self.const_text = [], []
        for i, text in enumerate(self.bit_text):
            for p in range(self.POLS):
                self.const_text[p].append(self.ax[p].text(
                    self.last_const[p, i].real + self.TEXT_OFFSET,
                    self.last_const[p, i].imag + self.TEXT_OFFSET,
                    text, c='magenta', fontsize=8
                ))
        if self.v_plt is not None:
            self._init_quiver(self.last_const, self.data['gmi_grad'])

    def update_data(self, const, hist=None):
        # https://matplotlib.org/stable/tutorials/advanced/blitting.html
        if not np.iscomplexobj(const):
            const = const[:, 0] + 1j * const[:, 1]
        if const.ndim == 1:
            const = np.vstack([const, const])

        if self.last_const is None or const.shape[1] != self.last_const.shape[1]:
            self.last_const = const
            self._update_points()

        # Initialising
        if self.h_plt is None and hist is not None:
            self.h_plt = [
                self.ax[p].imshow(
                hist[p], interpolation='bicubic', extent=[-self.LIM, self.LIM, -self.LIM, self.LIM],
                origin='lower', cmap='sillekens', animated=True)  # kindlmann, sillekens
                for p in range(self.POLS)
            ]
            for p in range(self.POLS):
                self.h_plt[p].set_visible(self.data['show_rx'])
                self.data.on('show_rx', lambda x: (self.h_plt[p].set_visible(x), self._redraw()), now=False)

            # self.h_plt0 = self.ax[0].imshow(
            #     hist[0], interpolation='bicubic', extent=[-self.LIM, self.LIM, -self.LIM, self.LIM],
            #     origin='lower', cmap='sillekens', animated=True)  # kindlmann, sillekens
            # self.h_plt0.set_visible(self.data['show_rx'])
            #
            # self.h_plt1 = self.ax[1].imshow(
            #     hist[1], interpolation='bicubic', extent=[-self.LIM, self.LIM, -self.LIM, self.LIM],
            #     origin='lower', cmap='sillekens', animated=True)  # kindlmann, sillekens
            # self.h_plt1.set_visible(self.data['show_rx'])
            #
            # self.data.on('show_rx', lambda x: (self.h_plt0.set_visible(x), self._redraw()), now=False)
            # self.data.on('show_rx', lambda x: (self.h_plt1.set_visible(x), self._redraw()), now=False)

        if self.c_plt is None and const is not None:
            self.c_plt = [
                self.ax[p].plot(const[p].real, const[p].imag, '.', c='magenta', animated=True)[0]
                for p in range(self.POLS)
            ]
            for p in range(self.POLS):
                self.c_plt[p].set_visible(self.data['show_c'])
                self.data.on('show_c', lambda x: (self.c_plt[p].set_visible(x), self._redraw()), now=False)

            for text in self._const_texts:
                text.set_visible(self.data['show_bmap'])
            self.data.on('show_bmap', lambda x: ([t.set_visible(x) for t in self._const_texts], self._redraw()), now=False)

        if self.v_plt is None and self.data['gmi_grad'] is not None:
            self._init_quiver(const, self.data['gmi_grad'])
            for p in range(self.POLS):
                self.data.on('show_vec', lambda x: (self.v_plt[p].set_visible(x), self._redraw()), now=False)

        # Drawing
        if self.h_plt is not None and self.data['show_rx']:
            for p in range(self.POLS):
                self.h_plt[p].set_data(np.zeros((128, 128)) if hist is None else hist[p])
        if self.c_plt is not None and self.data['show_c']:
            for p in range(self.POLS):
                self.c_plt[p].set_data(const[p].real, const[p].imag)
        if self.v_plt is not None and self.data['show_vec']:
            for p in range(self.POLS):
                self.v_plt[p].set_offsets(np.array([const[p].real, const[p].imag]).T)
            if hist is None or self.data['gmi_grad'] is None or np.any(np.isnan(self.data['gmi_grad'])):
                m = np.zeros(const.shape[1])
                for p in range(self.POLS):
                    self.v_plt[p].set_UVC(m, m)
            else:
                for p in range(self.POLS):
                    self.v_plt[p].set_UVC(self.data['gmi_grad'][:, 0], self.data['gmi_grad'][:, 1])
        if self.c_plt is not None and self.data['show_bmap']:
            for p in range(self.POLS):
                for i, text in enumerate(self.const_text[p]):
                    if np.all(np.isfinite(const[p, i])):
                        text.set_x(const[p, i].real + self.TEXT_OFFSET)
                        text.set_y(const[p, i].imag + self.TEXT_OFFSET)
        self._redraw()
