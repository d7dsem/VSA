# vsa_runkit.py
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg

from data_layer import ParamsDSP, ParamsStreamUDP, do_hard_work
from fft_core import swap_freq, to_dbfs

from PySide6.QtGui import QColor
import pyqtgraph as pg
# RAW_PEN = pg.mkPen(QColor("gold"), width=1)
# EMA_PEN = pg.mkPen(QColor("crimson"), width=1)


LogFn = Callable[[List[str]], None]
GetStreamFn = Callable[[], ParamsStreamUDP]
GetDSPFn = Callable[[], ParamsDSP]


@dataclass
class SharedSpectr:
    lock: threading.Lock
    last_raw: Optional[np.ndarray]
    last_ema: Optional[np.ndarray]
    last_meta: Optional[Dict[str, Any]]
    seq: int

    @staticmethod
    def create() -> "SharedSpectr":
        return SharedSpectr(threading.Lock(), None, None, None, 0)


class Worker(QtCore.QObject):
    finished = QtCore.Signal()

    def __init__(self, shared: SharedSpectr, stream_prm: ParamsStreamUDP, dsp_prm: ParamsDSP, out_fps: float):
        super().__init__()
        self._shared = shared
        self._stream_prm = stream_prm
        self._dsp_prm = dsp_prm
        self._out_fps = float(out_fps)
        self._stop_evt = threading.Event()

    def request_stop(self) -> None:
        self._stop_evt.set()

    def _stop_requested(self) -> bool:
        return self._stop_evt.is_set()

    def _on_spectrum(self, raw: np.ndarray, ema: Optional[np.ndarray], meta: Dict) -> None:
        with self._shared.lock:
            spec = raw.copy()
            to_dbfs(spec)
            swap_freq(spec)
            if ema is not None:
                ema_spec = ema.copy()
                to_dbfs(ema_spec)
                swap_freq(ema_spec)
            else:
                ema_spec = None
            self._shared.last_raw = spec
            self._shared.last_ema = ema_spec
            self._shared.last_meta = dict(meta)
            self._shared.seq += 1
            

    @QtCore.Slot()
    def run(self) -> None:
        try:
            do_hard_work(
                self._stream_prm,
                self._dsp_prm,
                stop_requested=self._stop_requested,
                on_spectrum_cb=self._on_spectrum,
                out_fps=self._out_fps,
            )
        finally:
            self.finished.emit()


class SpectrumDialog(QtWidgets.QDialog):
    def __init__(self, shared: SharedSpectr, dsp_prm: ParamsDSP, *, target_fps: float, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Spectrum")
        self.setWindowModality(QtCore.Qt.ApplicationModal)

        # auto y-range control (dB-domain)
        self._y_guard_frac = 0.10
        self._y_hyst_up = 3.0
        self._y_hyst_dn = 3.0
        self._y_shrink_hold = 3
        self._y_shrink_steps = 5

        self._ylim_inited = False
        self._ymin = 0.0
        self._ymax = 0.0
        self._quiet_updates = 0
        self._shrink_active = False
        self._shrink_step_ymin = 0.0
        self._shrink_step_ymax = 0.0

        self._ylim_inited = False
        self._ymin = 0.0
        self._ymax = 0.0

        self._quiet_updates = 0
        self._shrink_active = False
        self._shrink_step_ymin = 0.0
        self._shrink_step_ymax = 0.0

        self._shared = shared
        self._seen_seq = -1

        lay = QtWidgets.QVBoxLayout(self)

        row = QtWidgets.QHBoxLayout()
        self.cb_raw = QtWidgets.QCheckBox("RAW")
        self.cb_raw.setChecked(True)
        self.cb_ema = QtWidgets.QCheckBox("EMA")
        self.cb_ema.setChecked(True)
        row.addWidget(self.cb_raw)
        row.addWidget(self.cb_ema)
        row.addStretch(1)
        lay.addLayout(row)

        pg.setConfigOptions(antialias=False)
        self.plot = pg.PlotWidget()
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.setLabel("bottom", "frequency", units="Hz")
        self.plot.setLabel("left", "power")
        lay.addWidget(self.plot, 1)

        x = dsp_prm.gen_freq_bins().astype(np.float32, copy=False)
        self._x = x
        self.curve_raw = self.plot.plot(x, np.zeros_like(x),
                                        pen=pg.mkPen(QColor("gold"), width=1))
        self.curve_ema = self.plot.plot(x, np.zeros_like(x),
                                        pen=pg.mkPen(QColor("crimson"),width=1, style=QtCore.Qt.DashLine))

        self.lbl = QtWidgets.QLabel("")
        lay.addWidget(self.lbl)

        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(int(1000 / max(float(target_fps), 0.1)))
        self._timer.timeout.connect(self._on_tick)
        self._timer.start()

    def _update_ylim(self, y: np.ndarray) -> None:
        y_min = float(np.min(y))
        y_max = float(np.max(y))

        span = max(y_max - y_min, 1e-6)
        guard = span * self._y_guard_frac

        tgt_ymin = y_min - guard
        tgt_ymax = y_max + guard

        if not self._ylim_inited:
            self._ymin = tgt_ymin
            self._ymax = tgt_ymax
            self._ylim_inited = True
            self._quiet_updates = 0
            self._shrink_active = False
            self.plot.setYRange(self._ymin, self._ymax, padding=0)
            return

        expanded = False

        # expand fast (hysteresis)
        if y_max > self._ymax + self._y_hyst_up:
            self._ymax = y_max + guard
            expanded = True
        if y_min < self._ymin - self._y_hyst_dn:
            self._ymin = y_min - guard
            expanded = True

        if expanded:
            self._quiet_updates = 0
            self._shrink_active = False
            self.plot.setYRange(self._ymin, self._ymax, padding=0)
            return

        # detect quiet
        if (tgt_ymax < self._ymax - self._y_hyst_dn) and (tgt_ymin > self._ymin + self._y_hyst_dn):
            self._quiet_updates += 1
        else:
            self._quiet_updates = 0
            self._shrink_active = False
            return

        # arm shrink
        if (not self._shrink_active) and (self._quiet_updates >= self._y_shrink_hold):
            self._shrink_active = True
            self._shrink_step_ymax = max((self._ymax - tgt_ymax) / self._y_shrink_steps, 0.0)
            self._shrink_step_ymin = max((tgt_ymin - self._ymin) / self._y_shrink_steps, 0.0)

        # shrink slow
        if self._shrink_active:
            if self._ymax > tgt_ymax:
                self._ymax = max(tgt_ymax, self._ymax - self._shrink_step_ymax)
            if self._ymin < tgt_ymin:
                self._ymin = min(tgt_ymin, self._ymin + self._shrink_step_ymin)

            if (self._ymax <= tgt_ymax) and (self._ymin >= tgt_ymin):
                self._shrink_active = False
                self._quiet_updates = 0

            self.plot.setYRange(self._ymin, self._ymax, padding=0)


    @QtCore.Slot()
    def _on_tick(self) -> None:
        with self._shared.lock:
            seq = self._shared.seq
            if seq == self._seen_seq:
                return
            raw = self._shared.last_raw
            ema = self._shared.last_ema
            meta = self._shared.last_meta
            self._seen_seq = seq

        self.curve_raw.setVisible(self.cb_raw.isChecked())
        self.curve_ema.setVisible(self.cb_ema.isChecked() and (ema is not None))

        if raw is not None and self.cb_raw.isChecked():
            self.curve_raw.setData(self._x, raw)
            self._update_ylim(raw)
        if ema is not None and self.cb_ema.isChecked():
            self.curve_ema.setData(self._x, ema)

        if meta is not None:
            self.lbl.setText(
                f"seq={seq} pkt={meta.get('pkt_total')} mbps={meta.get('mbps'):.2f} fft_ms={meta.get('avg_fft_ms'):.3f}"
            )

    def closeEvent(self, ev) -> None:
        self._timer.stop()
        super().closeEvent(ev)


class VsaRunKit(QtCore.QObject):
    """
    Переносимий атомарний RUN/IDLE:
    - start(): створює Shared, відкриває модальний графік, стартує worker thread
    - stop(): просить зупинитись, закриває діалог
    """
    running_changed = QtCore.Signal(bool)

    def __init__(
        self,
        *,
        parent: QtWidgets.QWidget,
        get_stream: GetStreamFn,
        get_dsp: GetDSPFn,
        log: LogFn,
        target_fps: float = 24.0,
        out_fps: float = 24.0,
    ):
        super().__init__(parent)
        self._parent = parent
        self._get_stream = get_stream
        self._get_dsp = get_dsp
        self._log = log
        self._target_fps = float(target_fps)
        self._out_fps = float(out_fps)

        self.shared = SharedSpectr.create()

        self._running = False
        self._dlg: Optional[SpectrumDialog] = None
        self._th: Optional[QtCore.QThread] = None
        self._worker: Optional[Worker] = None

    def is_running(self) -> bool:
        return self._running

    @QtCore.Slot(bool)
    def set_running(self, on: bool) -> None:
        if on:
            self.start()
        else:
            self.stop()

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self.running_changed.emit(True)

        stream_prm = self._get_stream()
        dsp_prm = self._get_dsp()

        # dialog
        self._dlg = SpectrumDialog(self.shared, dsp_prm, target_fps=self._target_fps, parent=self._parent)
        self._dlg.finished.connect(self._on_dialog_closed)
        self._dlg.resize(1000, 600)
        self._dlg.show()

        # worker thread
        self._th = QtCore.QThread(self._parent)
        self._worker = Worker(self.shared, stream_prm=stream_prm, dsp_prm=dsp_prm, out_fps=self._out_fps)
        self._worker.moveToThread(self._th)

        self._th.started.connect(self._worker.run)
        self._worker.finished.connect(self._th.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._th.finished.connect(self._th.deleteLater)
        self._th.finished.connect(self._on_worker_finished)

        self._th.start()
        self._log(["[RUN] started"])

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        self.running_changed.emit(False)

        if self._worker is not None:
            self._worker.request_stop()

        if self._dlg is not None:
            self._dlg.blockSignals(True)
            try:
                self._dlg.close()
            finally:
                self._dlg.blockSignals(False)
            self._dlg = None

        self._log(["[RUN] stop requested"])

    @QtCore.Slot()
    def _on_dialog_closed(self) -> None:
        # закрили графік -> вся система в IDLE
        if self._running:
            self._log(["[RUN] dialog closed -> stopping"])
            self.stop()

    @QtCore.Slot()
    def _on_worker_finished(self) -> None:
        self._log(["[RUN] worker finished"])
        self._th = None
        self._worker = None
