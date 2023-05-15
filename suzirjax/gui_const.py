from suzirjax.gui_helpers import *
from suzirjax.simulation import Simulation

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

import numpy as np


class ConstellationCanvas(FigureCanvas):
    # LIM = 1 + 2 ** -.5  # Should match simulation.py HIST_LIM
    LIM = Simulation.HIST_LIM
    TEXT_OFFSET = 0.05

    def __init__(self, data: Connector):
        super().__init__()
        self.data = data
        self.ax = self.figure.subplots()
        self.ax.set_ylim(ymin=-self.LIM, ymax=self.LIM)
        self.ax.set_xlim(xmin=-self.LIM, xmax=self.LIM)
        self.ax.set_aspect('equal', 'box')

        self.h_plt = None
        self.c_plt = None
        self.bg = None
        self.last_const = None

        self.const_text = []

        self.data.on('show_rx', lambda v: self.set_comp_visible(self.h_plt, v), default=True)
        self.data.on('show_c', lambda v: self.set_comp_visible(self.c_plt, v), default=True)
        self.data.on('show_bmap', lambda v: ([t.set_visible(v) for t in self.const_text], self.figure.canvas.draw()), default=True)

        # ("Show received", make_checkbox(bind=self.data.bind("show_rx", True))),
        # ("Show constellation", make_checkbox(bind=self.data.bind("show_c", True))),
        # ("Show bitmap", make_checkbox(bind=self.data.bind("show_bmap", True))),

        add_right_clk_menu(
            self,
            make_checkable_action("Show received", self.data.bind("show_rx"), self),
            make_checkable_action("Show constellation", self.data.bind("show_c"), self),
            make_checkable_action("Show bitmap", self.data.bind("show_bmap"), self),
            # ("Show received", lambda: self.data.set("show_rx", not self.data["show_rx"])),
            # ("Show constellation", lambda: self.data.set("show_c", not self.data["show_c"])),
            # ("Show bitmap", lambda: self.data.set("show_bmap", not self.data["show_bmap"])),
            None,
            ("Save as", self.save),
            parent=self,
        )

    def set_comp_visible(self, comp, v):
        if comp is not None:
            comp.set_visible(v)
            self.ax.draw_artist(comp)
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
                self.last_const[i, 0] + self.TEXT_OFFSET, self.last_const[i, 1] + self.TEXT_OFFSET, text, c='magenta'))

    def update_data(self, const, hist):
        # Not initialised yet
        if self.h_plt is None:
            self.bg = self.figure.canvas.copy_from_bbox(self.figure.bbox)

            self.h_plt = self.ax.imshow(
                hist, interpolation='bicubic', extent=[-self.LIM, self.LIM, -self.LIM, self.LIM],
                origin='lower', cmap='sillekens', animated=True)  # kindlmann, sillekens
            self.c_plt, = self.ax.plot(const[:, 0], const[:, 1], '.', c='magenta', animated=True)

            self.h_plt.set_visible(self.data['show_rx'])
            self.c_plt.set_visible(self.data['show_c'])

            self.ax.draw_artist(self.h_plt)
            self.ax.draw_artist(self.c_plt)

            self.figure.canvas.blit(self.figure.bbox)
        else:
            if self.data['show_rx']:
                self.h_plt.set_data(hist)
                self.ax.draw_artist(self.h_plt)
            if self.data['show_c']:
                self.c_plt.set_data(const[:, 0], const[:, 1])
                self.ax.draw_artist(self.c_plt)

            # self.h_plt.set_visible(self.data['show_rx'])
            # self.c_plt.set_visible(self.data['show_c'])

            self.figure.canvas.blit(self.figure.bbox)
            self.figure.canvas.flush_events()

        if self.last_const is None or const.shape[0] != self.last_const.shape[0]:
            self.last_const = const
            self._update_points()

        # for i, text in enumerate(self.const_text):
        #     if np.all(np.isfinite(const[i])):
        #         text.set_x(const[i, 0] + self.TEXT_OFFSET)
        #         text.set_y(const[i, 1] + self.TEXT_OFFSET)
        # self.figure.canvas.draw()

