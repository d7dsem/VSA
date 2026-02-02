
# import faulthandler
# faulthandler.enable()
# faulthandler.dump_traceback_later(10, repeat=False)
# faulthandler.cancel_dump_traceback_later()

import sys, os
# print("=== DEBUG ENV ===")
# print("sys.executable:", sys.executable)
# print("sys.prefix:", sys.prefix)
# print("VIRTUAL_ENV:", os.environ.get("VIRTUAL_ENV"))
# print("CONDA_PREFIX:", os.environ.get("CONDA_PREFIX"))
# print("PYTHONPATH:", os.environ.get("PYTHONPATH"))
# try:
#     import matplotlib
#     print("matplotlib:", matplotlib.__version__, "backend:", matplotlib.get_backend())
# except Exception as e:
#     print("matplotlib import failed:", repr(e))
# try:
#     import PySide6
#     from PySide6 import QtGui, QtWidgets
#     print("PySide6:", PySide6.__version__)
#     print("QtGui has QApplication:", hasattr(QtGui, "QApplication"))
#     print("QtWidgets has QApplication:", hasattr(QtWidgets, "QApplication"))
# except Exception as e:
#     print("PySide6 import failed:", repr(e))
# print("=================")

import argparse
from pathlib import Path
import signal
import traceback
from typing import Optional, Tuple
from matplotlib import pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from colorizer import colorize, inject_colors_into
from fft_core import P_FS_RAW, IQInterleavedF32, IQInterleavedI16, i16_to_f32
from io_stuff import FReader
from vsa_gsv import VSA
# --- color names for IDE/static analysis suppress warnings --------------------
GREEN: str; BRIGHT_GREEN: str; RED: str; BRIGHT_RED: str
YELLOW: str; BRIGHT_YELLOW: str; BLACK: str; BRIGHT_BLACK: str
GRAY: str; BRIGHT_GRAY: str; CYAN: str; BRIGHT_CYAN: str
BLUE: str; BRIGHT_BLUE: str; MAGENTA: str; BRIGHT_MAGENTA: str
WHITE: str; BRIGHT_WHITE: str; RESET: str; WARN: str; ERR: str; INFO: str; DBG: str
inject_colors_into(globals())

# =====================================================
# Identity
_MODULE_MARKER = Path(__file__).stem

# _fpath = r"e:\home\d7\Public\signals\OFDM\manual_gen\test_rand_BPSK_ofdm_128-16-90.wav"
# _fpath = r"e:\home\d7\Public\signals\OFDM\manual_gen\test_rand_BPSK_ofdm_64-16-44.wav"
# _fpath = r"e:\home\d7\Public\signals\OFDM\manual_gen\test_rand_BPSK_ofdm_64-32-44.wav"
# _fpath = r"e:\home\d7\Public\signals\OFDM\manual_gen\test_rand_BPSK_ofdm_64-4-44.wav"
samp_rate = 2e6

_fpath = r"e:\home\d7\Public\signals\OFDM\5MHz-11-14-25.bin"
samp_rate = 10e6

# _fpath = r"e:\home\d7\Public\signals\OFDM\5MHz-11-14-25.bin"
# samp_rate = 10e6

# _fpath = r"e:\home\d7\Public\signals\OFDM\baseband_1340000000Hz_13-23-48_22-01-2026.wav"
# _fpath = r"e:\home\d7\Public\signals\OFDM\baseband_1340000000Hz_13-50-16_22-01-2026.wav"
# _fpath = r"e:\home\d7\Public\signals\OFDM\baseband_1340000000Hz_13-51-21_22-01-2026.wav"


_Fc = 0
_Fwdt = None
_fft_n = 1024

def do_vsa_file(
    fr: FReader,

    Fc: float = _Fc,
    freq_span: Tuple[float, float] | None = (-_Fwdt, +_Fwdt) if _Fwdt else None,

    fft_n: int = _fft_n,
    freq_grid_count: int | None = None,

    # sigma: float|None = None,
    # alpha: float|None = None, 
    
    sigma: float|None = .5,
    alpha: float|None = 0.01,
    
    start_pos: int = 0,
    step_samples: Optional[int] = 1,
    spec_y_lim: Optional[Tuple[float, float]] = None, #(-40, 40),
    iq_y_lim: Optional[Tuple[float, float]] = None,
    render_dt: float = 0.001,
    use_dbfs: bool = True,
) -> None:
    """
    Process WAV file with tri-panel visualization.
    
    spec_y_lim: hard y-limits for spectrum (optional)
    iq_y_lim: hard y-limits for I/Q signals (optional)
    """

    Fs: float = fr.Fs
    dF = Fs / fft_n
    Fl, Fr = freq_span if freq_span else (Fc - Fs/2, Fc + Fs/2)

    # Індекси бінів для меж діапазону
    idx_l = int(np.round((Fl - Fc) / dF + fft_n // 2))
    idx_r = int(np.round((Fr - Fc) / dF + fft_n // 2))

    # Уточнені частоти на основі дійсних індексів
    Fl_adj = (idx_l - fft_n // 2) * dF + Fc
    Fr_adj = (idx_r - fft_n // 2) * dF + Fc

    print(f"{INFO} Fs={Fs/1e6:.1f} MHz  {fft_n=:_}  dF={int(dF):_} Hz"\
        f"\n    Freq_span req: [{Fl/1e6:^6.1f} {Fc/1e6:^6.1f} {Fr/1e6:^6.1f}] MHz"\
        f"\n    Freq_span adj: [{Fl_adj/1e6:^6.1f} {Fc/1e6:^6.1f} {Fr_adj/1e6:^6.1f}] MHz"\
        f"\n    Bin indices:   [{idx_l:_}..{idx_r:_}] ({idx_r-idx_l:_})")
    
    freq_bins = (np.arange(fft_n, dtype=np.float32) - (fft_n // 2)) * dF + Fc
        
    i16_buf: IQInterleavedI16 = np.empty(fft_n * 2, dtype=np.int16)
    f32_buf: IQInterleavedF32 = np.empty(fft_n * 2, dtype=np.float32)
    x_c: np.ndarray = np.empty(fft_n, dtype=np.complex64)
    spec_ema: np.ndarray = np.zeros(fft_n, dtype=np.float32)
    freq_bins = (np.arange(fft_n, dtype=np.float32) - (fft_n // 2)) * dF + Fc

    vsa = VSA(
        freq_bins,
        freq_show_idx=(idx_l, idx_r),
        freq_grid_step= fft_n // freq_grid_count if freq_grid_count else None,

        render_dt=render_dt,
        spec_y_lim=spec_y_lim,
        iq_y_lim=iq_y_lim,
        use_dbfs=use_dbfs
    )

    fr.jump_to_samp_pos(start_pos)
    n_iter = 0

    def _process_frame(update=None):
        """Process one frame: FFT for spectrum, extract I/Q for time-domain."""
        i16_to_f32(i16_buf, f32_buf, fft_n, normalize=False)
        if update is None:
            update = lambda *args, **kwargs: None
        # Extract I and Q components
        y_i = f32_buf[0::2].copy()
        y_q = f32_buf[1::2].copy()

        # Compute spectrum 
        x_c.real[:] = f32_buf[0::2]
        x_c.imag[:] = f32_buf[1::2]

        X = np.fft.fft(x_c, n=fft_n)
        X = np.fft.fftshift(X)
        P = (X.real * X.real + X.imag * X.imag)
        p_spec_ema = None
        if alpha is not None:
            np.add(alpha * P, (1-alpha)*spec_ema, out=spec_ema)
        if use_dbfs:
            y_spec = 10.0 * np.log10(P / P_FS_RAW)
            if alpha is not None: 
                p_spec_ema = 10.0 * np.log10(spec_ema / P_FS_RAW)
        else:
            y_spec = np.sqrt(P)
            if alpha is not None:
                p_spec_ema = np.sqrt(spec_ema)

        y_spec_smooth = None if sigma is None else gaussian_filter1d(y_spec, sigma=sigma)
        # analize spectre curve for find peaks with prominience - find idx of them
        # map finded idx to freq bins
        v_lines_hz = None
        h_lines_db = None
        title_add = ""
        if 1:
            _spec_smooth = y_spec_smooth
            # _spec_smooth = p_spec_ema
            
            # 1. Шукаємо індекси піків на згладженому спектрі
            # prominence=10 означає, що пік має підніматися над "підошвою" мінімум на 10 dB
            peaks_idx, properties = find_peaks(_spec_smooth, prominence=10, distance=5)

            # 2. Мапимо індекси на фізичні частоти (Гц)
            # ВАЖЛИВО: використовуємо повний масив частот, що відповідає y_spec
            v_lines_hz = freq_bins[peaks_idx]

            # 3. Визначаємо горизонтальну лінію (наприклад, середній рівень шуму + запас)
            noise_floor = np.median(y_spec) 
            threshold_line = noise_floor + 15  # Лінія відсічки
            h_lines_db = [threshold_line]
            cn_peaks = len(v_lines_hz)
            df_mean = 0.0
            title_add = f" peaks found: {cn_peaks} |"
            if cn_peaks > 1:
                # np.diff рахує різницю між сусідніми елементами: [f2-f1, f3-f2, ...]
                df_mean = np.mean(np.diff(v_lines_hz))
                title_add += f" df_mean: {df_mean/1e3:.3f} KHz |"
            # print(title_add)

        # _freqs = freq_bins[::freq_bins.size//4]
        update( 
            y_spec, smoothed=y_spec_smooth, ema=p_spec_ema, 
            I=y_i, Q=y_q,
            v_lines=v_lines_hz,
            h_lines=h_lines_db,
            title=f" {fr.progress_str()} {title_add}"
        )

    while True:
        if vsa.stop_requested:
            print(f"[MAIN] stop_requested=True: terminating...", flush=True)
            break

        if not vsa.figure_alive():
            print(f"[MAIN] figure closed: terminating...", flush=True)
            break

        if vsa.paused:
            delta = vsa.get_and_clear_step_delta()
            if delta != 0:
                new_pos = fr.curr_sampl_pos + delta * step_samples
                new_pos = max(0, min(new_pos, fr.samples_total - fft_n))
                fr.jump_to_samp_pos(new_pos)

                n_read = fr.read_raw_into(i16_buf, fft_n*2)
                if n_read == fft_n*2:
                    _process_frame(update=vsa.update)

            plt.pause(vsa.render_dt)
            continue

        n_read = fr.read_raw_into(i16_buf, fft_n*2)
        if n_read != fft_n*2:
            print(f"read {n_read} not match chunk {fft_n}x2 : terminate...")
            break
        _process_frame(update=vsa.update)
        n_iter += 1
        if step_samples is not None:
            fr.jump_to_samp_pos(start_pos + step_samples * n_iter)

    print(f"[MAIN] do_vsa_file() exited cleanly", flush=True)


# CLI
def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vsa-file.py",
        description="VSA-file: spectral analysis & visualization for IQ recordings (.bin/.wav).",
    )

    p.add_argument(
        "-i", "--input",
        dest="file",
        type=Path,
        required=True,
        help="Path to IQ file: .bin or .wav",
    )

    p.add_argument(
        "--samp-rate",
        dest="samp_rate",
        type=float,
        default=None,
        help="Sample rate (Hz). Required for .bin; for .wav taken from header (if provided: must match).",
    )

    p.add_argument(
        "--samp-offs",
        dest="samp_offs",
        type=int,
        default=0,
        help="Sample offset in IQ samples (I/Q pairs). 1 sample = 4 bytes for int16 IQ.",
    )

    p.add_argument(
        "--dtype",
        dest="dtype",
        type=str,
        default="int16",
        help="Expected sample dtype. For .wav mismatch -> exception. (Current reader: int16 only)",
    )

    p.add_argument(
        "--center-freq",
        dest="center_freq",
        type=float,
        default=0.0,
        help="Center frequency (Hz). Default: 0",
    )

    p.add_argument(
        "--fft-n",
        dest="fft_n",
        type=int,
        default=4096,
        help="FFT size. Default: 4096",
    )

    p.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        help="Verbose output.",
    )

    return p


def _apply_vsa_file_contract(args: argparse.Namespace) -> argparse.Namespace:
    """
    Contract enforcement (no guessing):
    - file must be .bin or .wav
    - .bin requires samp_rate
    - .wav uses Fs from header; if args.samp_rate provided must match
    - dtype is currently restricted to int16 (per current reader contract)
    - samp_offs is in IQ samples (pairs): byte offset = samp_offs * 4
    """
    f: Path = args.file
    if f.suffix.lower() not in (".bin", ".wav"):
        raise RuntimeError(f"unsupported file type '{f.absolute()}'")

    if args.dtype != "int16":
        raise RuntimeError(f"unsupported dtype '{args.dtype}' (current contract: int16 only)")

    if f.suffix.lower() == ".bin":
        if args.samp_rate is None:
            raise RuntimeError("--samp-rate is required for .bin input")
    else:
        # wav: Fs from header; validate if user also provided samp_rate
        from wav import read_wav_header  # uses your existing module
        props = read_wav_header(f)
        wav_fs = float(props["sample_rate"])
        if args.samp_rate is not None and float(args.samp_rate) != wav_fs:
            raise RuntimeError(f"--samp-rate mismatch: cli={args.samp_rate} wav={wav_fs}")
        args.samp_rate = wav_fs

    return args
 
def show_cli():
    print(f"\n{DBG} CLI: {GRAY}" + " ".join(map(str, sys.argv)) + RESET)
    
def handle_sigint(signum, frame):
    print("Ctrl+C pressed. Graceful shutdown...")
    raise KeyboardInterrupt
# ==============================================================================
# === CLI Entry Point
# ==============================================================================
if __name__ == "__main__":
    os.system("")  # Colorizing 'on'
    # Стандартна ініціалізація
    signal.signal(signal.SIGINT, handle_sigint)
    
    args: argparse.Namespace = None
    if len(sys.argv) > 1:
        # show_cli()
        args = _build_cli().parse_args()
    else:
        print(f"{WARN} USe dev defaults!")
        args = argparse.Namespace(
            file=Path(_fpath),
            samp_offset=0,
            samp_rate=samp_rate,
            center_freq=5e6,
            fft_n=1024,
            dtype="int16",
            verbose=True
        )
    try:
        args = _apply_vsa_file_contract(args)
        print(f"{INFO} File: {YELLOW}{args.file.stem}{RESET}")
        with FReader(args) as fr:
            do_vsa_file(fr)
    except KeyboardInterrupt:
        pass
    except FileNotFoundError as e:
        print(f"{ERR} {str(e)}")
    except Exception as e:
        print(f"{ERR} Unhandled exception: {e}")
        traceback.print_exc()
        show_cli()
        