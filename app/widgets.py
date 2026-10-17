"""Small reusable Qt widgets."""

import numpy as np
from PySide6 import QtCore, QtWidgets


class ParamSlider(QtWidgets.QWidget):
    """Labeled slider over a float range (linear or log), emitting valueChanged(float)."""

    valueChanged = QtCore.Signal(float)

    def __init__(self, label, lo, hi, value, unit="", log=False, decimals=1, steps=400, tooltip=""):
        super().__init__()
        self.lo, self.hi, self.log, self.decimals, self.unit, self.steps = lo, hi, log, decimals, unit, steps
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 2)
        lay.setSpacing(0)
        top = QtWidgets.QHBoxLayout()
        self.name = QtWidgets.QLabel(label)
        self.value_label = QtWidgets.QLabel()
        self.value_label.setAlignment(QtCore.Qt.AlignRight)
        self.value_label.setStyleSheet("font-weight: 600;")
        top.addWidget(self.name)
        top.addWidget(self.value_label, 1)
        lay.addLayout(top)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, steps)
        lay.addWidget(self.slider)
        if tooltip:
            self.setToolTip(tooltip)
        self.slider.valueChanged.connect(self._changed)
        self.set_value(value)

    def _to_pos(self, v):
        if self.log:
            f = (np.log(v) - np.log(self.lo)) / (np.log(self.hi) - np.log(self.lo))
        else:
            f = (v - self.lo) / (self.hi - self.lo)
        return int(round(np.clip(f, 0, 1) * self.steps))

    def _from_pos(self, k):
        f = k / self.steps
        if self.log:
            return float(np.exp(np.log(self.lo) + f * (np.log(self.hi) - np.log(self.lo))))
        return float(self.lo + f * (self.hi - self.lo))

    def value(self):
        v = self._from_pos(self.slider.value())
        if self.log:  # round log values to 3 significant figures for readable labels
            return float(f"{v:.3g}")
        return round(v, self.decimals)

    def set_value(self, v):
        self.slider.blockSignals(True)
        self.slider.setValue(self._to_pos(v))
        self.slider.blockSignals(False)
        self._update_label()

    def _update_label(self):
        v = self.value()
        text = f"{v:,.{self.decimals}f}" if not self.log else (f"{v:,.0f}" if v >= 100 else f"{v:.3g}")
        self.value_label.setText(f"{text} {self.unit}".strip())

    def _changed(self, _):
        self._update_label()
        self.valueChanged.emit(self.value())


def section(title):
    box = QtWidgets.QGroupBox(title)
    lay = QtWidgets.QVBoxLayout(box)
    lay.setSpacing(2)
    return box, lay
