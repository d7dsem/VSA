# app_main.py
import sys
from dataclasses import dataclass
from typing import Any, Dict, Final

from PySide6 import QtCore, QtWidgets


_HDR_SZ: Final = 8
def do_sock_work(
    port: int,
    rd_timeout_ms: int = 750,
    hdr_sz: int = _HDR_SZ,
    pack_sz: int = 8192 + _HDR_SZ,
    Fs: float = 480e3,
    # Fc: float = 430.1e6,
    Fc: float = 429.85e6,
    fft_n: int = 256,
    fft_batch_depth : int = 128, # fft windows in batch
    sigma: float = 1.75,
    ema_alpha: float = 0.1,
    target_fps: float = 24.0,
    draw_every: int = 100,
    vsa_on: bool = True
) -> None:
    ...


@dataclass
class ParamsSocket:
    port: int = 9999
    rd_timeout_ms: int = 750
    hdr_sz: int = 8
    pack_sz: int = 8192 + 8


@dataclass
class ParamsStream:
    Fs: float = 480e3
    Fc: float = 430e6
    fft_n: int = 256
    fft_batch_depth: int = 128
    sigma: float = 1.75
    ema_alpha: float = 0.1


@dataclass
class ParamsTech:
    target_fps: float = 24.0
    draw_every: int = 100
    vsa_on: bool = True


class Worker(QtCore.QObject):
    log_line = QtCore.Signal(str)
    finished = QtCore.Signal()

    def __init__(self, params: Dict[str, Any]):
        super().__init__()
        self._params = params
        self._stop = False

    @QtCore.Slot()
    def run(self) -> None:
        # do_sock_work is blocking; stop is handled by closing socket / internal checks.
        # For now: rely on app stop by terminating thread (next step: add stop flag to do_sock_work).
        try:
            def log_cb(s: str) -> None:
                self.log_line.emit(s)

            do_sock_work(**self._params, log=log_cb)
        finally:
            self.finished.emit()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("UDP FFT Tool (prototype)")
        self._thread: QtCore.QThread | None = None
        self._worker: Worker | None = None

        self.sock = ParamsSocket()
        self.stream = ParamsStream()
        self.tech = ParamsTech()

        # -------- UI --------
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        # Socket group
        g_sock = QtWidgets.QGroupBox("Socket")
        f_sock = QtWidgets.QFormLayout(g_sock)
        self.ed_port = QtWidgets.QSpinBox(); self.ed_port.setRange(1, 65535); self.ed_port.setValue(self.sock.port)
        self.ed_to = QtWidgets.QSpinBox(); self.ed_to.setRange(1, 60_000); self.ed_to.setValue(self.sock.rd_timeout_ms)
        self.ed_hdr = QtWidgets.QSpinBox(); self.ed_hdr.setRange(0, 4096); self.ed_hdr.setValue(self.sock.hdr_sz)
        self.ed_pack = QtWidgets.QSpinBox(); self.ed_pack.setRange(1, 1_000_000); self.ed_pack.setValue(self.sock.pack_sz)
        f_sock.addRow("port", self.ed_port)
        f_sock.addRow("rd_timeout_ms", self.ed_to)
        f_sock.addRow("hdr_sz", self.ed_hdr)
        f_sock.addRow("pack_sz", self.ed_pack)

        # Stream group
        g_stream = QtWidgets.QGroupBox("Stream")
        f_stream = QtWidgets.QFormLayout(g_stream)
        self.ed_Fs = QtWidgets.QDoubleSpinBox(); self.ed_Fs.setRange(1.0, 1e9); self.ed_Fs.setDecimals(1); self.ed_Fs.setValue(self.stream.Fs)
        self.ed_Fc = QtWidgets.QDoubleSpinBox(); self.ed_Fc.setRange(-1e12, 1e12); self.ed_Fc.setDecimals(1); self.ed_Fc.setValue(self.stream.Fc)
        self.ed_fft_n = QtWidgets.QSpinBox(); self.ed_fft_n.setRange(8, 1_048_576); self.ed_fft_n.setValue(self.stream.fft_n)
        self.ed_fft_bd = QtWidgets.QSpinBox(); self.ed_fft_bd.setRange(1, 1_000_000); self.ed_fft_bd.setValue(self.stream.fft_batch_depth)
        self.ed_sigma = QtWidgets.QDoubleSpinBox(); self.ed_sigma.setRange(0.0, 100.0); self.ed_sigma.setDecimals(3); self.ed_sigma.setValue(self.stream.sigma)
        self.ed_alpha = QtWidgets.QDoubleSpinBox(); self.ed_alpha.setRange(0.0, 1.0); self.ed_alpha.setDecimals(6); self.ed_alpha.setValue(self.stream.ema_alpha)
        f_stream.addRow("Fs", self.ed_Fs)
        f_stream.addRow("Fc", self.ed_Fc)
        f_stream.addRow("fft_n", self.ed_fft_n)
        f_stream.addRow("fft_batch_depth", self.ed_fft_bd)
        f_stream.addRow("sigma", self.ed_sigma)
        f_stream.addRow("ema_alpha", self.ed_alpha)

        # Technical group
        g_tech = QtWidgets.QGroupBox("Technical")
        f_tech = QtWidgets.QFormLayout(g_tech)
        self.ed_fps = QtWidgets.QDoubleSpinBox(); self.ed_fps.setRange(0.1, 1000.0); self.ed_fps.setDecimals(2); self.ed_fps.setValue(self.tech.target_fps)
        self.ed_draw_every = QtWidgets.QSpinBox(); self.ed_draw_every.setRange(1, 1_000_000); self.ed_draw_every.setValue(self.tech.draw_every)
        self.cb_vsa = QtWidgets.QCheckBox("vsa_on"); self.cb_vsa.setChecked(self.tech.vsa_on)
        f_tech.addRow("target_fps", self.ed_fps)
        f_tech.addRow("draw_every", self.ed_draw_every)
        f_tech.addRow(self.cb_vsa)

        # Controls
        row = QtWidgets.QHBoxLayout()
        self.btn_start = QtWidgets.QPushButton("Start")
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        row.addWidget(self.btn_start)
        row.addWidget(self.btn_stop)
        row.addStretch(1)

        # Terminal
        self.term = QtWidgets.QPlainTextEdit()
        self.term.setReadOnly(True)

        layout.addWidget(g_sock)
        layout.addWidget(g_stream)
        layout.addWidget(g_tech)
        layout.addLayout(row)
        layout.addWidget(self.term, 1)

        self.btn_start.clicked.connect(self.on_start)
        self.btn_stop.clicked.connect(self.on_stop)

    def _collect_params(self) -> Dict[str, Any]:
        return dict(
            port=int(self.ed_port.value()),
            rd_timeout_ms=int(self.ed_to.value()),
            hdr_sz=int(self.ed_hdr.value()),
            pack_sz=int(self.ed_pack.value()),
            Fs=float(self.ed_Fs.value()),
            Fc=float(self.ed_Fc.value()),
            fft_n=int(self.ed_fft_n.value()),
            fft_batch_depth=int(self.ed_fft_bd.value()),
            sigma=float(self.ed_sigma.value()),
            ema_alpha=float(self.ed_alpha.value()),
            target_fps=float(self.ed_fps.value()),
            draw_every=int(self.ed_draw_every.value()),
            vsa_on=bool(self.cb_vsa.isChecked()),
        )

    @QtCore.Slot()
    def on_start(self) -> None:
        if self._thread is not None:
            return

        params = self._collect_params()
        self.term.appendPlainText(f"[UI] start with params: {params}")

        self._thread = QtCore.QThread(self)
        self._worker = Worker(params)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.log_line.connect(self.term.appendPlainText)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(self._on_thread_finished)

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._thread.start()

    @QtCore.Slot()
    def on_stop(self) -> None:
        # MVP: force-stop by stopping thread.
        # Next step: add a stop flag into do_sock_work and exit cleanly.
        if self._thread is None:
            return
        self.term.appendPlainText("[UI] stop requested (hard)")
        self._thread.requestInterruption()
        self._thread.quit()

    @QtCore.Slot()
    def _on_thread_finished(self) -> None:
        self.term.appendPlainText("[UI] worker finished")
        self._thread = None
        self._worker = None
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.resize(900, 800)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
