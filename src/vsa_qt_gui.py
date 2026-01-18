# vsa_gui.py
from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, List, Optional

from PySide6 import QtCore, QtWidgets

from data_layer import ParamsDSP, ParamsStreamUDP
from vsa_runkit import VsaRunKit


LogFn = Callable[[List[str]], None]


class LogFrame(QtWidgets.QGroupBox):
    def __init__(self, parent=None):
        super().__init__("Log", parent)
        lay = QtWidgets.QVBoxLayout(self)

        self._text = QtWidgets.QPlainTextEdit()
        self._text.setReadOnly(True)
        self._text.setMaximumBlockCount(5000)  # щоб не ріс безмежно
        lay.addWidget(self._text)

    def append_lines(self, lines: List[str]) -> None:
        if not lines:
            return
        self._text.appendPlainText("\n".join(lines))

    def clear(self) -> None:
        self._text.clear()


class StreamParamsFrame(QtWidgets.QGroupBox):
    """
    UI для ParamsStreamUDP.
    Видає "поточні параметри" через get_params().
    """
    def __init__(self, prm: ParamsStreamUDP, log: LogFn, parent=None):
        super().__init__("Source Params (ParamsStreamUDP)", parent)
        self._log = log

        lay = QtWidgets.QFormLayout(self)

        self._port = QtWidgets.QSpinBox()
        self._port.setRange(1, 65535)
        self._port.setValue(int(prm.port))

        self._hdr_sz = QtWidgets.QSpinBox()
        self._hdr_sz.setRange(0, 4096)
        self._hdr_sz.setValue(int(prm.hdr_sz))

        self._payload_sz = QtWidgets.QSpinBox()
        self._payload_sz.setRange(16, 1_000_000)
        self._payload_sz.setValue(int(prm.payload_sz))

        self._rd_timeout = QtWidgets.QSpinBox()
        self._rd_timeout.setRange(1, 60_000)
        self._rd_timeout.setValue(int(prm.rd_timeout_ms))

        lay.addRow("port", self._port)
        lay.addRow("hdr_sz", self._hdr_sz)
        lay.addRow("payload_sz", self._payload_sz)
        lay.addRow("rd_timeout_ms", self._rd_timeout)

        btn = QtWidgets.QPushButton("Log current")
        btn.clicked.connect(self._on_log_current)
        lay.addRow(btn)

    def get_params(self) -> ParamsStreamUDP:
        return ParamsStreamUDP(
            port=int(self._port.value()),
            hdr_sz=int(self._hdr_sz.value()),
            payload_sz=int(self._payload_sz.value()),
            rd_timeout_ms=int(self._rd_timeout.value()),
        )

    @QtCore.Slot()
    def _on_log_current(self) -> None:
        prm = self.get_params()
        self._log([f"[StreamParams] {prm.to_str()}"])


class DSPParamsFrame(QtWidgets.QGroupBox):
    """
    UI для ParamsDSP.
    """
    def __init__(self, prm: ParamsDSP, log: LogFn, parent=None):
        super().__init__("DSP Params (ParamsDSP)", parent)
        self._log = log

        lay = QtWidgets.QFormLayout(self)

        self._Fs = QtWidgets.QDoubleSpinBox()
        self._Fs.setRange(1.0, 1000e6)
        self._Fs.setDecimals(3)
        self._Fs.setValue(float(prm.Fs))

        self._Fc = QtWidgets.QDoubleSpinBox()
        self._Fc.setRange(-1000e6, 1000e6)
        self._Fc.setDecimals(3)
        self._Fc.setValue(float(prm.Fc))

        self._fft_n = QtWidgets.QSpinBox()
        self._fft_n.setRange(16, 1 << 20)
        self._fft_n.setValue(int(prm.fft_n))

        self._batch_depth = QtWidgets.QSpinBox()
        self._batch_depth.setRange(1, 1 << 20)
        self._batch_depth.setValue(int(prm.fft_batch_depth))

        self._sigma = QtWidgets.QDoubleSpinBox()
        self._sigma.setRange(0.0, 1000.0)
        self._sigma.setDecimals(3)
        self._sigma.setValue(float(prm.sigma))

        self._ema_alpha = QtWidgets.QDoubleSpinBox()
        self._ema_alpha.setRange(0.0, 1.0)
        self._ema_alpha.setDecimals(6)
        # якщо в prm None — показуємо 0 і вимикаємо чекбоксом
        self._ema_enabled = QtWidgets.QCheckBox("EMA enabled")
        self._ema_enabled.setChecked(prm.ema_alpha is not None)
        self._ema_alpha.setValue(0.0 if prm.ema_alpha is None else float(prm.ema_alpha))
        self._ema_alpha.setEnabled(self._ema_enabled.isChecked())
        self._ema_enabled.toggled.connect(self._ema_alpha.setEnabled)

        lay.addRow("Fs", self._Fs)
        lay.addRow("Fc", self._Fc)
        lay.addRow("fft_n", self._fft_n)
        lay.addRow("fft_batch_depth", self._batch_depth)
        lay.addRow("sigma", self._sigma)
        lay.addRow(self._ema_enabled, self._ema_alpha)

        btn = QtWidgets.QPushButton("Log current")
        btn.clicked.connect(self._on_log_current)
        lay.addRow(btn)

    def get_params(self) -> ParamsDSP:
        ema = float(self._ema_alpha.value()) if self._ema_enabled.isChecked() else None
        return ParamsDSP(
            Fs=float(self._Fs.value()),
            Fc=float(self._Fc.value()),
            fft_n=int(self._fft_n.value()),
            fft_batch_depth=int(self._batch_depth.value()),
            sigma=float(self._sigma.value()),
            ema_alpha=ema,
        )

    @QtCore.Slot()
    def _on_log_current(self) -> None:
        prm = self.get_params()
        self._log([f"[DSPParams] Fs={prm.Fs} Fc={prm.Fc} fft_n={prm.fft_n} depth={prm.fft_batch_depth} sigma={prm.sigma} ema={prm.ema_alpha}"])


class ControlsFrame(QtWidgets.QGroupBox):
    """
    Кнопки: VSA on/off, clear log, exit.
    Сигнали залишаємо простими, каркас під майбутнє.
    """
    vsa_toggled = QtCore.Signal(bool)
    clear_log = QtCore.Signal()
    exit_requested = QtCore.Signal()

    def __init__(self, log: LogFn, parent=None):
        super().__init__("Controls", parent)
        self._log = log

        lay = QtWidgets.QHBoxLayout(self)

        self._btn_vsa = QtWidgets.QPushButton("VSA")
        self._btn_vsa.setCheckable(True)
        self._btn_vsa.setChecked(False)
        self._btn_vsa.toggled.connect(self._on_vsa_toggled)

        self._btn_clear = QtWidgets.QPushButton("Clear log")
        self._btn_clear.clicked.connect(self._on_clear)

        self._btn_exit = QtWidgets.QPushButton("Exit")
        self._btn_exit.clicked.connect(self._on_exit)

        lay.addWidget(self._btn_vsa)
        lay.addWidget(self._btn_clear)
        lay.addStretch(1)
        lay.addWidget(self._btn_exit)

    def set_vsa_on(self, on: bool) -> None:
        """
        Встановити стан кнопки VSA програмно БЕЗ емісу vsa_toggled.
        Використовується для синхронізації UI зі станом runkit.
        """
        self._btn_vsa.blockSignals(True)
        try:
            self._btn_vsa.setChecked(on)
        finally:
            self._btn_vsa.blockSignals(False)

    @QtCore.Slot(bool)
    def _on_vsa_toggled(self, on: bool) -> None:
        self.vsa_toggled.emit(on)

    @QtCore.Slot()
    def _on_clear(self) -> None:
        self.clear_log.emit()

    @QtCore.Slot()
    def _on_exit(self) -> None:
        self.exit_requested.emit()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, prm_stream: ParamsStreamUDP, prm_dsp: ParamsDSP):
        super().__init__()
        self.setWindowTitle("VSA Qt")

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        self._runkit: VsaRunKit = None  
        self.log_frame = LogFrame()
        self.stream_frame = StreamParamsFrame(prm_stream, self.log_gui)
        self.dsp_frame = DSPParamsFrame(prm_dsp, self.log_gui)
        self.ctrl_frame = ControlsFrame(self.log_gui)

        self.ctrl_frame.clear_log.connect(self.log_frame.clear)
        self.ctrl_frame.exit_requested.connect(self.close)

        root.addWidget(self.stream_frame)
        root.addWidget(self.dsp_frame)
        root.addWidget(self.log_frame, 1)
        root.addWidget(self.ctrl_frame)

    def attach_runkit(self, runkit: VsaRunKit):
        self._runkit = runkit
        self.ctrl_frame.vsa_toggled.connect(runkit.set_running)
        runkit.running_changed.connect(self.ctrl_frame.set_vsa_on)
        
    def log_gui(self, msg: List[str]) -> None:
        # ЄДИНА точка логування GUI
        self.log_frame.append_lines(msg)

    def get_params_stream(self) -> ParamsStreamUDP:
        return self.stream_frame.get_params()

    def get_params_dsp(self) -> ParamsDSP:
        return self.dsp_frame.get_params()
