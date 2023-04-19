from __future__ import annotations

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from typing import List, Tuple, TypeVar, Union, Any, Dict
from inspect import signature

__all__ = []


def export(fn):
    __all__.append(fn.__name__)
    return fn


@export
class ConnectorValue:
    def __init__(self, connector, name):
        self.connector = connector
        self.name = name

    def set(self, value):
        self.connector[self.name] = value

    def get(self):
        return self.connector[self.name]

    def on(self, callback, now=True):
        self.connector.on(self.name, callback, now=now)

    def __delitem__(self, key):
        if self.name not in self.connector.callbacks:
            return
        cbs = self.connector.callbacks[self.name]
        if key in cbs:
            del key[cbs]


@export
class Connector:
    def __init__(self):
        self.data = {}
        self.callbacks = {}

    def _call_cb(self, callback_params, current_val, previous_val):
        callback, call_on_none = callback_params
        sig = signature(callback)
        if len(sig.parameters) == 0:
            callback()
        elif len(sig.parameters) == 1:
            if call_on_none or current_val is not None:
                callback(current_val)
        elif len(sig.parameters) == 2:
            if call_on_none or not (previous_val is None or current_val is None):
                callback(current_val, previous_val)
        else:
            raise ValueError(f"Callback {callback} needs more than two arguments")

    def set(self, name, value) -> 'Connector':
        if self.data.get(name) == value:
            return self
        previous = self.data.get(name)
        self.data[name] = value
        if name in self.callbacks:
            for callback in self.callbacks[name]:
                self._call_cb(callback, value, previous)
        return self

    def get(self, name, default=None):
        if name in self.data:
            return self.data[name]
        return default

    def bind(self, name, default=None) -> ConnectorValue:
        """ return single connector value """
        if default is not None:
            self.set(name, default)
        return ConnectorValue(self, name)

    def on(self, name, callback, default=None, now=True, call_on_none=True) -> 'Connector':
        """ callback on value change, now to execute immediately once """
        if name not in self.data and default is not None:
            self.set(name, default)
        if name not in self.callbacks:
            self.callbacks[name] = set()
        self.callbacks[name].add((callback, call_on_none))
        if now:
            self._call_cb((callback, call_on_none), self.data.get(name), None)
        return self

    def remove_callbacks(self, name):
        if name in self.callbacks:
            del self.callbacks[name]

    def __getitem__(self, item):
        return self.get(item)

    def __setitem__(self, key, value):
        return self.set(key, value)

    def copy(self):
        ret = Connector()
        ret.data = self.data.copy()
        return ret


def make_widget_layout(layout, widgets, parent=None, widget_class=QWidget):
    for w in widgets:
        layout.addWidget(w)

    if widget_class is not None:
        widget = QWidget(parent)
        widget.setLayout(layout)
        return widget
    return layout


@export
def HLayout(*widgets: QWidget, parent=None, widget_class=QWidget) -> QWidget:
    return make_widget_layout(QHBoxLayout(), widgets, parent=parent, widget_class=widget_class)


@export
def VLayout(*widgets: QWidget, parent=None, widget_class=QWidget) -> QWidget:
    return make_widget_layout(QVBoxLayout(), widgets, parent=parent, widget_class=widget_class)


@export
def FLayout(*widgets: Tuple[str | QWidget, QWidget], parent=None) -> QWidget:
    layout = QFormLayout()
    root_wget = QWidget(parent)
    for text, wget in widgets:
        if isinstance(text, str):
            text = QLabel(text)
        layout.addRow(text, wget)
    root_wget.setLayout(layout)
    return root_wget


@export
def make_dialog(title, *widgets, parent=None, buttons=QDialogButtonBox.Ok | QDialogButtonBox.Cancel):
    dlg = QDialog(parent)
    dlg.setWindowTitle(title)

    layout = QVBoxLayout()
    for w in widgets:
        layout.addWidget(w)

    if buttons is not None:
        dlg.bbox = QDialogButtonBox(buttons)
        dlg.bbox.accepted.connect(dlg.accept)
        dlg.bbox.rejected.connect(dlg.reject)
        layout.addWidget(dlg.bbox)
    dlg.setLayout(layout)
    return dlg


@export
def make_combo(*items, bind: ConnectorValue = None, parent=None):
    combo = QComboBox(parent)
    combo.addItems(items)
    if bind:
        bind.on(lambda s: combo.setCurrentText(s))
        combo.currentTextChanged.connect(lambda _: bind.set(combo.currentText()))
    return combo


@export
def make_combo_dict(items: Dict[str, Any], bind: ConnectorValue = None, parent=None):
    combo = QComboBox(parent)
    combo.addItems(items.keys())
    if bind:
        bind.on(lambda s: combo.setCurrentText(list(items.keys())[list(items.values()).index(s)]))
        combo.currentTextChanged.connect(lambda _: bind.set(items.get(combo.currentText())))
    return combo


@export
def make_input(bind: ConnectorValue = None, parent=None, placeholder=None, validator=None, cast=str):
    wget = QLineEdit(parent)
    if placeholder:
        wget.setPlaceholderText(placeholder)
    if validator:
        wget.setValidator(validator)
    if bind:
        def _cb(s):
            try:
                bind.set(cast(wget.text()))
            except ValueError as e:
                print(e)

        wget.textChanged.connect(_cb)
    return wget


@export
def make_int_input(min: int, max: int, step: int = 1, bind: ConnectorValue = None, parent=None, placeholder=None):
    # return make_input(bind, QSpinBox, parent, placeholder, QIntValidator(min, max), cast=int)
    widget = QSpinBox(parent)
    widget.setMinimum(min)
    widget.setMaximum(max)
    widget.setSingleStep(step)
    bind.on(lambda v: widget.setValue(v))
    widget.valueChanged.connect(lambda _: bind.set(widget.value()))
    return widget
    # return make_input(bind, QSpinBox, parent, placeholder, cast=int)


@export
def make_float_input(min: float, max: float, step: float, bind: ConnectorValue = None, parent=None, placeholder=None):
    # return make_input(bind, QDoubleSpinBox, parent, placeholder, QDoubleValidator(min, max, num), cast=float)
    widget = QDoubleSpinBox(parent)
    widget.setMinimum(min)
    widget.setMaximum(max)
    widget.setSingleStep(step)
    bind.on(lambda v: widget.setValue(v))
    widget.valueChanged.connect(lambda _: bind.set(widget.value()))
    return widget
    # return make_input(bind, QDoubleSpinBox, parent, placeholder, cast=float)


@export
def make_button(text, func, parent=None):
    button = QPushButton(text, parent)
    if callable(func):
        button.clicked.connect(lambda _: func(button))
    else:
        raise TypeError('func must be callable')
    return button


@export
def make_hidden_group(*widgets: QWidget, bind: ConnectorValue, bind_value=True) -> Tuple[QWidget]:
    """ Hide widgets in the list, unless bind connector value is equal to bind_value """

    def _on_change(new_val):
        hide = new_val != bind_value
        for widget in widgets:
            widget.setHidden(hide)

    bind.on(_on_change)
    return widgets


@export
def make_radio_buttons(*options: Tuple[str, Any], parent=None, bind: ConnectorValue) -> QWidget:
    """ Make a group of radio buttons, options are tuples of string and value """
    main = QWidget(parent)
    layout = QVBoxLayout()
    main.setLayout(layout)

    val_map = {}

    def __reg(name, val):
        nonlocal val_map
        wget = QRadioButton(main)
        wget.setText(name)
        wget.toggled.connect(lambda _: bind.set(val))
        layout.addWidget(wget)
        val_map[val] = wget
    [__reg(n, v) for n, v in options]

    def _toggle(val):
        if val in val_map:
            val_map[bind.get()].toggle()

    bind.on(_toggle)
    return main


@export
def make_checkbox(bind: ConnectorValue, name=None, on_value=True, off_value=False) -> QWidget:
    cbox = QCheckBox(name)
    bind.on(lambda x: cbox.setChecked(x == on_value))
    cbox.stateChanged.connect(lambda _: bind.set(on_value if cbox.isChecked() else off_value))
    return cbox


@export
def make_label(bind: ConnectorValue, formatting='{}') -> QWidget:
    label = QLabel()
    bind.on(lambda x: label.setText(formatting.format(x)))
    return label


@export
def make_latex(tex: str, fontsize=18) -> QWidget:
    """ Source: https://stackoverflow.com/questions/32035251 """
    fig = matplotlib.figure.Figure()
    fig.patch.set_facecolor('none')
    fig.set_canvas(FigureCanvasAgg(fig))
    renderer = fig.canvas.get_renderer()

    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.patch.set_facecolor('none')
    t = ax.text(0, 0, tex, ha='left', va='bottom', fontsize=fontsize)

    fwidth, fheight = fig.get_size_inches()
    fig_bbox = fig.get_window_extent(renderer)
    text_bbox = t.get_window_extent(renderer)
    tight_fwidth = text_bbox.width * fwidth / fig_bbox.width
    tight_fheight = text_bbox.height * fheight / fig_bbox.height
    fig.set_size_inches(tight_fwidth, tight_fheight)

    buf, size = fig.canvas.print_to_buffer()
    qimage = QImage.rgbSwapped(QImage(buf, size[0], size[1], QImage.Format.Format_ARGB32))

    label = QLabel()
    label.setPixmap(QPixmap(qimage))

    return label
