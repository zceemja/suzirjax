from PyQt5 import QtTest
from PyQt5.QtCore import QEventLoop, QTimer

from suzirjax.gui_helpers import *
from suzirjax.simulation import Simulation

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

import numpy as np

class ConstellationCanvas(FigureCanvas):
    # LIM = 1 + 2 ** -.5  # Should match simulation.py HIST_LIM
    LIM = Simulation.HIST_LIM
    TEXT_OFFSET = 0.03

    def __init__(self, data: Connector):
        super().__init__()
        self.data = data

        self.ax = self.figure.subplots()
        self.ax.set_ylim(ymin=-self.LIM, ymax=self.LIM)
        self.ax.set_xlim(xmin=-self.LIM, xmax=self.LIM)
        self.ax.set_aspect('equal', 'box')

        self.figure.patch.set_facecolor('black')
        self.ax.set_facecolor('black')
        self.ax.tick_params(axis='x', labelcolor='white')
        self.ax.tick_params(axis='y', labelcolor='white')
        for spine in self.ax.spines.values():
            spine.set_color('white')

        self.h_plt = None
        self.c_plt = None
        self.v_plt = None
        self.bg = None
        self.last_const = None

        self.const_text = []

        add_right_clk_menu(
            self,
            # make_checkable_action("Show received", self.data.bind("show_rx", True), self),
            make_checkable_action("Show constellation", self.data.bind("show_c", True), self),
            make_checkable_action("Show bitmap", self.data.bind("show_bmap", True), self),
            make_checkable_action("Show vectors", self.data.bind("show_vec", False), self),
            # ("Show received", lambda: self.data.set("show_rx", not self.data["show_rx"])),
            # ("Show constellation", lambda: self.data.set("show_c", not self.data["show_c"])),
            # ("Show bitmap", lambda: self.data.set("show_bmap", not self.data["show_bmap"])),
            None,
            ("Save as", self.save),
            parent=self,
        )

    def _redraw(self):
        if self.h_plt is not None and self.data['show_rx']:
            self.ax.draw_artist(self.h_plt)
        if self.c_plt is not None and self.data['show_c']:
            self.ax.draw_artist(self.c_plt)
        if self.data['show_bmap']:
            for text in self.const_text:
                self.ax.draw_artist(text)
        if self.v_plt is not None and self.data['show_vec']:
            self.ax.draw_artist(self.v_plt)
        self.figure.canvas.blit(self.figure.bbox)
        self.figure.canvas.flush_events()


    def _update_points(self):
        m = self.last_const.shape[0]
        format_str = f'{{0:0{int(np.log2(m))}b}}'
        self.bit_text = [format_str.format(i) for i in range(m)]
        if len(self.const_text) > 0:
            for text in self.const_text:
                text.remove()
        self.const_text = []
        for i, text in enumerate(self.bit_text):
            self.const_text.append(self.ax.text(
                self.last_const[i, 0] + self.TEXT_OFFSET,
                self.last_const[i, 1] + self.TEXT_OFFSET,
                text, c='magenta', fontsize=8
            ))

    def update_data(self, const, hist=None):
        if self.last_const is None or const.shape[0] != self.last_const.shape[0]:
            self.last_const = const
            self._update_points()

        # Initialising
        initial = False
        if self.h_plt is None and hist is not None:
            self.h_plt = self.ax.imshow(
                hist, interpolation='bicubic', extent=[-self.LIM, self.LIM, -self.LIM, self.LIM],
                origin='lower', cmap='sillekens', animated=True)  # kindlmann, sillekens
            self.h_plt.set_visible(self.data['show_rx'])
            self.data.on('show_rx', lambda x: (self.h_plt.set_visible(x), self._redraw()), now=False)

        if self.c_plt is None and const is not None:
            self.c_plt, = self.ax.plot(const[:, 0], const[:, 1], '.', c='magenta', animated=True)
            self.c_plt.set_visible(self.data['show_c'])
            for text in self.const_text:
                text.set_visible(self.data['show_bmap'])
            self.data.on('show_c', lambda x: (self.c_plt.set_visible(x), self._redraw()), now=False)
            self.data.on('show_bmap', lambda x: ([t.set_visible(x) for t in self.const_text], self._redraw()), now=False)

        if self.v_plt is None and self.data['gmi_grad'] is not None:
            self.v_plt = self.ax.quiver(
                const[:, 0], const[:, 1], self.data['gmi_grad'][:, 0], self.data['gmi_grad'][:, 1],
                color='aqua', width=0.003, animated=True)
            self.v_plt.set_visible(self.data['show_vec'])
            self.data.on('show_vec', lambda x: (self.v_plt.set_visible(x), self._redraw()), now=False)

        # Drawing
        if self.h_plt is not None and self.data['show_rx']:
            if hist is None:
                hist = np.zeros((128, 128))
            self.h_plt.set_data(hist)
        if self.c_plt is not None and self.data['show_c']:
            self.c_plt.set_data(const[:, 0], const[:, 1])
        if self.v_plt is not None and self.data['show_vec']:
            self.v_plt.set_offsets(const)
            self.v_plt.set_UVC(self.data['gmi_grad'][:, 0], self.data['gmi_grad'][:, 1])
        if self.c_plt is not None and self.data['show_bmap']:
            for i, text in enumerate(self.const_text):
                if np.all(np.isfinite(const[i])):
                    text.set_x(const[i, 0] + self.TEXT_OFFSET)
                    text.set_y(const[i, 1] + self.TEXT_OFFSET)
        self._redraw()
