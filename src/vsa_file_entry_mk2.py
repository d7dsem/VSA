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
from typing import Annotated, Final, List, Literal, Optional, Tuple, TypeAlias
from matplotlib import pyplot as plt
import numpy as np

import scipy.signal.windows as windows
from scipy.ndimage import gaussian_filter1d
from scipy.fft import fft as sfft
from scipy.signal import find_peaks


from helpers import BandwidtBurst, analyze_and_export_bands, find_bands
from io_stuff import FReader
from vsa import VSA, CMapType, ControledVidget, deploy_layout
from colorizer import colorize, inject_colors_into
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
# ============

INT16_FULL_SCALE: Final = 32768.0
IQInterleavedF32: TypeAlias = Annotated[np.typing.NDArray[np.float32], "float32 1D: i0 q0 i1 q1 ... (interleaved IQ)"]


_fpath = r"e:\home\d7\Public\signals\OFDM\5MHz-11-14-25.bin"
samp_rate = 1/1.00e-7

_fpath = r"D:\C\Repo\signals_data\Fc_421000000Hz_ch_4_v1.bin"
samp_rate = 480e3

_fpath = r"D:\C\Repo\signals_data\Fc_450000000Hz.wav"
_fpath = "e:\\home\\d7\\Public\\signals\\OFDM\\5MHz-11-14-25.bin"
samp_rate = 1/1.00e-7
# _fpath = r"D:\C\Repo\signals_data\OFDM\"
# samp_rate = None


Fc = 0
freq_wnd: Tuple[float, float] = None # (449.5e6, 449.9e6)
alpha = 0.01
sigma = 4.0
start_pos = 498_000 # 800_000
start_pos = 0
fft_n = 1024*2
batch_n = 4
p_val = 12
step = fft_n * batch_n // 8
def do_vsa_file(
    fr: FReader,
    Fs: float,
    Fc: float = Fc,
    fft_n: int = fft_n,
    freq_wnd: Optional[Tuple[float, float]] = freq_wnd,
    spec_windowing:  Literal['hann', 'hamming', 'blackmanharris', 'rect'] = 'hann',
    
    start_pos: int = start_pos,
    batch_n: int = batch_n,
    step_samples: Optional[int] = step,

    sigma: Optional[float] = sigma,
    # p_val: Optional[int] = p_val,  # noise floor percentile
    threshold_iqr: float = 6.0, # Interquartile Range singal presence thr, dB
    min_bw: float = 1.25e6, # Minimal bw
    spec_y_lim: Optional[Tuple[float, float]] = None, #(-40, 40),
    render_dt: float = 0.001,
    no_plot: bool = False
) -> None:
    """
    Process IQ file.

    spec_y_lim: hard y-limits for spectrum (optional)
    """
    if not step_samples:
        step_samples = fft_n
    
    dF = Fs / fft_n
    chunk_len = fft_n * batch_n
    f32_buf: IQInterleavedF32 = np.empty(fft_n * batch_n * 2, dtype=np.float32)

    freq_bins_full = (np.arange(fft_n, dtype=np.float32) - (fft_n // 2)) * dF + Fc
    if freq_wnd:
        f_min, f_max = freq_wnd
        mask = (freq_bins_full >= f_min) & (freq_bins_full <= f_max)
    else:
        mask = slice(None) 
    freq_bins_view = freq_bins_full[mask]
    dur_ms = 1e3 * batch_n*fft_n / Fs
    title_base = f"File: {fr.f_path.stem} | SR: {Fs/1e6} MHz | {batch_n=}, {fft_n=:_} {dur_ms=:.3f} | dF: {dF/1e3} KHz |"
    
    if not no_plot:
        d7fg, ax_spec, ax_wfall, ax_side = deploy_layout()
        
        cmap_name: CMapType = 'inferno' # type: ignore
        vsa: ControledVidget = VSA(
            freq_bins_view, d7fg, ax_spec, ax_wfall, ax_side,
            render_dt=render_dt,
            spec_y_lim=spec_y_lim,
            cmap_name=cmap_name,
            title_base = title_base,
        ) 
    else:
        vsa = None

    fr.jump_to_samp_pos(start_pos)
    n_iter = 0

    _P_FS = float(fft_n)**2  # Константа для 0 dBFS (максимальна потужність комплексного тону)
    # INT16_FULL_SCALE == 32768
    if spec_windowing == 'rect':
        wnd_coeffs = np.ones(fft_n, dtype=np.float32) / INT16_FULL_SCALE
    else:
        raw_wnd = windows.get_window(spec_windowing, fft_n, fftbins=True)
        # Поєднуємо нормалізацію вікна (mean) та амплітуди (INT16)
        wnd_coeffs = (raw_wnd / (np.mean(raw_wnd) * INT16_FULL_SCALE)).astype(np.float32)
    p10: float = 0
    p75: float = 0
    def _process_frame(_update=None)->Tuple[bool, float, List[BandwidtBurst]]:
        # Returns has_signal and occupance
        # 1. Вікнування (Broadcasting)
        obs = (f32_buf[0::2].reshape(batch_n, fft_n) + 
            1j * f32_buf[1::2].reshape(batch_n, fft_n)) * wnd_coeffs

        # 2. Scipy FFT (Пакетне)
        
        X = sfft(obs, axis=1)

        # 3. Потужність БЕЗ кореня (X * conj(X))
        # Метод .real тут безпечний, бо результат множення комплексного на спряжене 
        # завжди суто дійсний, але NumPy може залишити тип complex.
        P_linear = np.mean((X * X.conj()).real, axis=0)
        
        # 4. Центрування та логарифм
        P_shifted = np.fft.fftshift(P_linear)
        y_spec = 10.0 * np.log10(np.maximum(P_shifted, 1e-15) / _P_FS)
        y_spec_smooth = gaussian_filter1d(y_spec, sigma=sigma) if sigma else y_spec
        
        # Depends on signal record BW and need msnual adjuct
        p10 = np.percentile(y_spec, 10)
        pHigh = np.percentile(y_spec, 40) # 75
        iqr = pHigh - p10
        has_sign = iqr > threshold_iqr
        v_lines = []
        h_coords =[p10, pHigh]
        if has_sign:
            # --- РОЗРАХУНОК OCCUPANCE ---
            # Визначаємо адаптивний поріг детектування сигналу. 
            # p75 + 0.5*iqr — це досить чутливий поріг, який адаптується під рівень шуму.
            
            # thr = pHigh + (0.25 * iqr)    # freq ocupance < 0.75*Fs
            thr = pHigh - (0.25 * iqr)      # freq ocupance > 0.75*Fs
            h_coords.append(thr)
            # Рахуємо кількість бінів, що перевищують цей поріг
            active_bins = np.sum(y_spec > thr)
            # Occupance — це відношення активних бінів до загальної кількості (від 0.0 до 1.0)
            occupance = active_bins / len(y_spec)
            bnd_lst:List[BandwidtBurst] = find_bands(y_spec_smooth, freq_bins_full, thr)
            for bnd in bnd_lst:
                v_lines.extend([bnd.f_start, bnd.center, bnd.f_end])
                v_lines.append(bnd.center)
        else:
            noise_floor = np.percentile(y_spec, p_val)
            occupance = 0.0
            h_coords =[noise_floor]
            thr = None
            bnd_lst:List[BandwidtBurst] = []
            
        # ----------------------------
        h_coords = np.array(h_coords)
 
        y_masked = y_spec[mask]
        y_smooth_masked = y_spec_smooth[mask]
        
        # 6. Виразний пошук піків
        # distance: мінімальна відстань між піками в бінах (наприклад, 10 бінів)
        # prominence: наскільки пік має виділятися над оточенням (наприклад, 5 dB)
        # peaks_idx, _ = find_peaks(
        #     y_smooth_masked, 
        #     height=-70,       # Поріг (як і раніше)
        #     distance=5,       # Не ліпити лінії занадто близько
        #     prominence=3.0    # Відсікаємо дрібний шум
        # )
        # # Отримуємо частоти для v_lines
        # peak_freqs = freq_bins_view[peaks_idx]
        # v_lines.extend(peak_freqs)

        if _update:
            _update(y_masked, y_smooth_masked, f" {fr.progress_str()} | {has_sign=} occ={occupance*100:.2f}%",
                    v_coords=np.array(v_lines),
                    h_coords=np.array(h_coords)
            )
        return has_sign, occupance, bnd_lst

    bands_lst:List[BandwidtBurst] = []
    while True:
        if vsa and vsa.stop_requested:
            print(f"[MAIN] stop_requested=True: terminating...", flush=True)
            break

        if vsa and not vsa.fig_alive:
            print(f"[MAIN] figure closed: terminating...", flush=True)
            break

        if vsa and vsa.paused:
            delta = vsa.delta
            if delta != 0:
                new_pos = fr.curr_sampl_pos + delta * step_samples
                new_pos = max(0, min(new_pos, fr.samples_total - chunk_len))
                fr.jump_to_samp_pos(new_pos)
                n_read = fr.read_samples_into(f32_buf, chunk_len)
                if n_read == chunk_len:
                    _process_frame(_update=vsa.update)
            plt.pause(vsa.render_dt)
            continue

        n_read = fr.read_samples_into(f32_buf, chunk_len)
        if n_read != chunk_len:
            print(f"read {n_read} not match chunk {fft_n}x{batch_n}x2 : terminate...")
            break
        # _process_frame(_update=vsa.update)
        has_sign, occ, bnd_lst = _process_frame(_update=vsa.update if vsa else None )
        for bnd in bnd_lst: bnd.start_samp_pos = fr.curr_sampl_pos
        if has_sign and occ*Fs > min_bw: bands_lst.extend(bnd_lst)
        n_iter += 1
        if no_plot and n_iter%1_000==0:
            prcnt = 100.0 * fr.curr_sampl_pos / fr.samples_total
            print(f"{prcnt:.2f}% {n_iter=:_}  {fr.curr_sampl_pos:_} {fr.samples_total:_}", end="\r")
        if step_samples is not None:
            fr.jump_to_samp_pos(start_pos + step_samples * n_iter)
    if no_plot: print()
    print(f"[MAIN] do_vsa_file() exited cleanly", flush=True)

    print(f"\n[FINAL] Analysis Complete.")
    print(f"Total iterations: {n_iter}")
    print(f"Signals detected in {len(bands_lst)} frames.")

    if bands_lst:
       analyze_and_export_bands(bands_lst, Fs, base_filename=fr.f_path.stem, folder=fr.f_path.parent)
    else:
        print("No signals matched your criteria (min_bw / threshold_iqr).")
    
    # plt.pause(30)

# CLI
def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vsa-file.py",
        description="VSA-file: spectral analysis & visualization for IQ recordings (.bin/.wav).",
    )

    p.add_argument(
        "-i", "--input", dest="file", type=Path, required=True,
        help="Path to IQ file: .bin or .wav",
    )

    p.add_argument(
        "--samp-rate", dest="samp_rate", type=float, default=None,
        help="Sample rate (Hz). Required for .bin; for .wav taken from header (if provided: must match).",
    )

    p.add_argument(
        "--samp-offs", dest="samp_offs", type=int, default=0,
        help="Sample offset in IQ samples (I/Q pairs). 1 sample = 4 bytes for int16 IQ.",
    )

    p.add_argument(
        "--dtype", dest="dtype", type=str, default="int16",
        help="Expected sample dtype. For .wav mismatch -> exception. (Current reader: int16 only)",
    )

    p.add_argument(
        "--center-freq", dest="center_freq", type=float, default=0.0,
        help="Center frequency (Hz). Default: 0",
    )

    p.add_argument(
        "--fft-n", dest="fft_n", type=int, default=4096,
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
    # signal.signal(signal.SIGINT, handle_sigint)
    
    args: argparse.Namespace = None
    if len(sys.argv) > 1:
        # show_cli()
        args = _build_cli().parse_args()
    else:
        print(f"{WARN} Use dev defaults!")

        args = argparse.Namespace(
            file=Path(_fpath),
            samp_offset=0,
            samp_rate=samp_rate,
            center_freq=0,
            fft_n=1024,
            dtype="int16",
            verbose=True
        )
        
    try:
        
        args = _apply_vsa_file_contract(args)
        
        with FReader(args) as fr:
            Fs=args.samp_rate
            print(f"Input file: {YELLOW}{args.file}{RESET}. Fs={Fs/1e3} KHz")
            do_vsa_file(fr, Fs=Fs)

    except KeyboardInterrupt:
        pass
    except FileNotFoundError as e:
        print(f"{ERR} {str(e)}")
    except Exception as e:
        print("-"*32)
        traceback.print_exc()
        print("-"*32)
        print(f"{ERR} Unhandled exception: {GRAY}{e}{RESET}")
        show_cli()
