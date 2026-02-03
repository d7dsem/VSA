import sys, os
import argparse
from pathlib import Path
import signal
import traceback
from typing import Annotated, Final, List, Literal, Optional, Tuple, TypeAlias
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np

import scipy.signal.windows as windows
from scipy.ndimage import gaussian_filter1d
from scipy.fft import fft as sfft
from scipy.signal import find_peaks


from fft_core import INT16_FULL_SCALE, IQInterleavedF32
from helpers import Bandwidt, analyze_and_export_bands, find_bands
from io_stuff import FReader
from vsa import CMapType

from colorizer import colorize, inject_colors_into
from vsa_file_entry import _apply_vsa_file_contract, show_cli
# --- color names for IDE/static analysis suppress warnings --------------------
GREEN: str; BRIGHT_GREEN: str; RED: str; BRIGHT_RED: str
YELLOW: str; BRIGHT_YELLOW: str; BLACK: str; BRIGHT_BLACK: str
GRAY: str; BRIGHT_GRAY: str; CYAN: str; BRIGHT_CYAN: str
BLUE: str; BRIGHT_BLUE: str; MAGENTA: str; BRIGHT_MAGENTA: str
WHITE: str; BRIGHT_WHITE: str; RESET: str; WARN: str; ERR: str; INFO: str; DBG: str
inject_colors_into(globals())

_break_req = False

_fpath="D:/C/Repo/signals_data/OFDM/baseband_1330000000Hz_15-47-10_02-02-2026_2ant.wav"
Fs=1e10

Fc=0
fft_n = 1024*2
batch_n = 4
min_bw: float = 1.25e6
sigma: float = 10.0
def do_file_stage_I(
    fr: FReader,
    Fs: float,
    Fc: float = Fc,
    fft_n: int = fft_n,
    batch_n: int = batch_n,
    sigma: float = sigma,
    min_bw: float = min_bw,
    threshold_iqr: float = 10.0,
    spec_windowing:  Literal['hann', 'hamming', 'blackmanharris', 'rect'] = 'rect'
):
    dF = Fs / fft_n
    chunk_len = fft_n * batch_n
    f32_buf: IQInterleavedF32 = np.empty(fft_n * batch_n * 2, dtype=np.float32)
    freq_bins_full = (np.arange(fft_n, dtype=np.float32) - (fft_n // 2)) * dF + Fc
    
    _P_FS = float(fft_n)**2  # Константа для 0 dBFS (максимальна потужність комплексного тону)
    # INT16_FULL_SCALE == 32768
    if spec_windowing == 'rect':
        wnd_coeffs = np.ones(fft_n, dtype=np.float32) / INT16_FULL_SCALE
    else:
        raw_wnd = windows.get_window(spec_windowing, fft_n, fftbins=True)
        # Поєднуємо нормалізацію вікна (mean) та амплітуди (INT16)
        wnd_coeffs = (raw_wnd / (np.mean(raw_wnd) * INT16_FULL_SCALE)).astype(np.float32)
    bands_lst:List[Bandwidt] = []
    # TODO: prealloc to wfall memory with zapas
    n_read = 0
    pbar = tqdm(total=fr.samples_total)
    while True:
        n_read = fr.read_samples_into(f32_buf, chunk_len)
        if n_read != chunk_len or _break_req:
            break
        
        # 1. Вікнування (Broadcasting)
        obs = (f32_buf[0::2].reshape(batch_n, fft_n) + 
            1j * f32_buf[1::2].reshape(batch_n, fft_n)) * wnd_coeffs
        # 2. Scipy FFT (Пакетне)
        X = sfft(obs, axis=1)
        # 3. Потужність
        P_linear = np.mean((X * X.conj()).real, axis=0)
                # 4. Центрування та логарифм
        P_shifted = np.fft.fftshift(P_linear)
        y_spec = 10.0 * np.log10(np.maximum(P_shifted, 1e-15) / _P_FS)
        y_spec_smooth = gaussian_filter1d(y_spec, sigma=sigma) if sigma else y_spec
        
        p10 = np.percentile(y_spec, 10)
        p75 = np.percentile(y_spec, 75)
        iqr = p75 - p10
        has_sign = iqr > threshold_iqr
        if has_sign:
            # --- РОЗРАХУНОК OCCUPANCE ---
            # Визначаємо адаптивний поріг детектування сигналу. 
            # p75 + 0.5*iqr — це досить чутливий поріг, який адаптується під рівень шуму.
            thr_occupance = p75 + (0.5 * iqr) 
            h_coords =[p10, p75, thr_occupance]
            # Рахуємо кількість бінів, що перевищують цей поріг
            active_bins = np.sum(y_spec > thr_occupance)
            # Occupance — це відношення активних бінів до загальної кількості (від 0.0 до 1.0)
            occupance = active_bins / len(y_spec)
            _bnd_lst:List[Bandwidt] = find_bands(y_spec_smooth, freq_bins_full, thr_occupance)
            bands_lst.extend(_bnd_lst)
            # TODO: push to wfall
        pbar.update(chunk_len)
    
    pbar.close()
    if _break_req:
        print(f"\n  Ctrl+C breaking")
    elif n_read != chunk_len:
        print(f"\n  {n_read=:_} not match chunk {fft_n}x{batch_n}x2. EOF, skip chunk.")
    else:
        print(f"\n  EOF")
        
    # TODO: bands to separate files

# CLI
def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vsa-file.py",
        description="Spec Analize pgase I: spectral analysis, build wfall finherprint, detect and spread bursts\bandwidth, for IQ recordings (.bin/.wav).",
    )

    p.add_argument(
        "-i", "--input", dest="file", type=Path, default=_fpath,
        help="Path to IQ file: .bin or .wav",
    )
    
    p.add_argument(
        "--Fs", dest="Fs", type=float, default=None,
        help="Sample rate (Hz). Required for .bin; for .wav taken from header (if provided: must match).",
    )
    
    p.add_argument(
        "--dtype", dest="dtype", type=str, default="int16",
        help="Expected sample dtype. For .wav mismatch -> exception. (Current reader: int16 only)",
    )

    p.add_argument(
        "--Fc", dest="Fc", type=float, default=0.0,
        help="Center frequency (Hz). Default: 0",
    )

    p.add_argument(
        "--fft-n", dest="fft_n", type=int, default=fft_n,
        help=f"FFT size. Default: {fft_n}",
    )

    p.add_argument(
        "--batch-n", dest="batch_n", type=int, default=batch_n,
        help=f"FFT batch size. Default: {batch_n}",
    )
    
    p.add_argument(
        "--bw", dest="bw", type=float, default=min_bw,
        help=f"Min Target  BW (Hz). {min_bw}",
    )

def handle_sigint(signum, frame):
    global _break_req
    _break_req = True
# ==============================================================================
# === CLI Entry Point
# ==============================================================================
if __name__ == "__main__":
    os.system("")  # Colorizing 'on'
    # Стандартна ініціалізація
    signal.signal(signal.SIGINT, handle_sigint)

    # show_cli()
    args: argparse.Namespace = _build_cli().parse_args()
    try:
        args = _apply_vsa_file_contract(args)
        with FReader(args) as fr:
            dur_sec = fr.samples_total / args.Fs
            print(f"Input file: {YELLOW}{args.file}{RESET}. Fs={args.Fs/1e3} KHz. df={1e-3*args.Fs/fft_n:.2} KHz. Dur={dur_sec:.2f} s")
            do_file_stage_I(fr, Fs=args.Fs, Fc=args.Fc, fft_n=args.fft_n, batch_n=args.batch_n, min_bw=args.bw)

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
