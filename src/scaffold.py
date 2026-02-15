
import multiprocessing
import sys, os
import threading
import time
from typing import Dict, Optional
from PySide6 import QtCore, QtWidgets
import numpy as np

from data_layer import HDR_SZ, ParamsDSP, ParamsStreamUDP, do_hard_work
from vsa_runkit import SharedSpectr

_dev_default_stream= ParamsStreamUDP(
    port=9999,
    hdr_sz= HDR_SZ,
    payload_sz= 8192,
    rd_timeout_ms=750
)
_dev_default_dsp = ParamsDSP(
    Fs=480e3,
    Fc=0.0,
    fft_n=256,
    fft_batch_depth=128, 
    sigma=1.75, 
    ema_alpha=0.1
)

class Worker(QtCore.QObject):
    """
    Qt wrapper for thread start of do_hard_work
    """
    finished = QtCore.Signal()

    def __init__(self, shared: SharedSpectr, stream_prm: ParamsStreamUDP, dsp_prm: ParamsDSP, out_fps: float):
        super().__init__()
        self._shared = shared
        self._stream_prm = stream_prm
        self._dsp_prm = dsp_prm
        self._out_fps = out_fps

    def _on_spectrum(self, raw: np.ndarray, ema: Optional[np.ndarray], meta: Dict) -> None:
        with self._shared.lock:
            self._shared.last_raw = raw.copy()
            self._shared.last_ema = None if ema is None else ema.copy()
            self._shared.last_meta = dict(meta)
            self._shared.seq += 1

    @QtCore.Slot()
    def run(self) -> None:
        try:
            do_hard_work(
                self._stream_prm,
                self._dsp_prm,
                on_spectrum_cb=self._on_spectrum,
                out_fps=self._out_fps,
            )
        finally:
            self.finished.emit()

def _main_qt() -> None:

    stream_prm = _dev_default_stream
    dsp_prm = _dev_default_dsp
    out_fps = 24.0
    
    app = QtWidgets.QApplication(sys.argv)
    th = QtCore.QThread()
    
    shared = SharedSpectr.create()
    worker = Worker(shared, stream_prm=stream_prm, dsp_prm=dsp_prm, out_fps=out_fps)
    worker.moveToThread(th)
    th.started.connect(worker.run)
    worker.finished.connect(th.quit)
    worker.finished.connect(worker.deleteLater)
    th.finished.connect(th.deleteLater)
    th.start()

    sys.exit(app.exec())


def start_watchdog_hard(proc: multiprocessing.Process, delay_sec: float) -> None:
    def _wd() -> None:
        time.sleep(delay_sec)

        pid = proc.pid
        if pid is None:
            return  # процес ще не стартував або вже знищений об'єкт

        # POSIX (Linux/macOS): SIGKILL
        if os.name == "posix":
            try:
                os.kill(pid, signal.SIGKILL)
                print(f"[WATCHDOG] SIGKILL pid={pid}", flush=True)
            except ProcessLookupError:
                pass
            return

        # Windows: terminate() == TerminateProcess (жорстко)
        if os.name == "nt":
            if proc.is_alive():
                proc.terminate()
                print(f"\n[WATCHDOG] TerminateProcess pid={pid}\n", flush=True)
            return

        # Невідома платформа: fallback (спроба terminate)
        if proc.is_alive():
            proc.terminate()
            print(f"\n[WATCHDOG] terminate() pid={pid} (fallback os.name={os.name})\n", flush=True)

    threading.Thread(target=_wd, daemon=True).start()


if __name__ == "__main__":
    _DEBUG = False
    # scaffold block
    if _DEBUG:
        # === ALT 1: cooperative stop (thread) ===
        if 1:
            import threading
            stop_evt = threading.Event()

            def stop_requested() -> bool:
                return stop_evt.is_set()

            t = threading.Thread(
                target=do_hard_work,
                args=(_dev_default_stream, _dev_default_dsp),
                kwargs={
                    "stop_requested": stop_requested,
                    "on_spectrum_cb": None,
                    "out_fps": 10.0,
                },
                daemon=False,
            )
            t.start()
            print("[MAIN][DBG] do_hard_work(thread) started", flush=True)

            time.sleep(15.0)
            print("\n[MAIN][DBG] requesting cooperative stop", flush=True)
            stop_evt.set()
            t.join(timeout=5.0)
            print("\n[MAIN][DBG] do_hard_work(thread) joined", flush=True)
            # === ALT 2: hard kill (process + watchdog) ===
        else:
            p = multiprocessing.Process(
                target=do_hard_work,
                daemon=False,
                args=(_dev_default_stream, _dev_default_dsp),
                kwargs={"on_spectrum_cb": None, "out_fps": 10.0},
            )
            p.start()
            print(f"[MAIN][DBG] do_hard_work(process) started pid={p.pid}", flush=True)

            start_watchdog_hard(p, delay_sec=10.0)
            time.sleep(12.0)  # після цього процес вже має бути вбитий watchdog’ом
            print("[MAIN][DBG] watchdog path done", flush=True)
        time.sleep(1.0)
        sys.exit(0)
