# app_main.py


import sys
from PySide6 import QtWidgets


from data_layer import ParamsDSP, ParamsStreamUDP, HDR_SZ
from vsa_qt_gui import MainWindow
from vsa_runkit import VsaRunKit

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


def main_qt() -> None:
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow(_dev_default_stream, _dev_default_dsp)
    w.resize(900, 700)
    w.show()
    
    runkit = VsaRunKit(
        parent=w,
        get_stream=w.get_params_stream,
        get_dsp=w.get_params_dsp,
        log=w.log_gui,
        target_fps=24.0,
        out_fps=24.0,
    )
    w.attach_runkit(runkit)


    
    sys.exit(app.exec())


if __name__ == "__main__":
    print(f"[MAIN] Start Qt app", flush=True)
    main_qt()
    
