# src/analize_stage_I.py

# PS hint
# ls e:\home\d7\Public\signals\OFDM\*wav | % {py analize_stage_I.py -i $_}
import sys, os
import argparse
from pathlib import Path
import signal
import traceback
from typing import Dict, List, Literal, Optional
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np

import scipy.signal.windows as windows
from scipy.ndimage import gaussian_filter1d
from scipy.fft import fft as sfft
from scipy.signal import find_peaks


from fft_core import INT16_FULL_SCALE, IQInterleavedF32
from helpers import BandwidtBurst, PackBwBurst, aggregate_to_packs, analyze_pack_intervals, find_bands, load_bursts, save_bursts
from io_stuff import FReader


from io_stuff import _apply_vsa_file_contract, show_cli
from colorizer import colorize, inject_colors_into
# --- color names for IDE/static analysis suppress warnings --------------------
GREEN: str; BRIGHT_GREEN: str; RED: str; BRIGHT_RED: str
YELLOW: str; BRIGHT_YELLOW: str; BLACK: str; BRIGHT_BLACK: str
GRAY: str; BRIGHT_GRAY: str; CYAN: str; BRIGHT_CYAN: str
BLUE: str; BRIGHT_BLUE: str; MAGENTA: str; BRIGHT_MAGENTA: str
WHITE: str; BRIGHT_WHITE: str; RESET: str; WARN: str; ERR: str; INFO: str; DBG: str
inject_colors_into(globals())

_break_req = False

def save_session_artifacts_wfall(out_dir:Path, idx:int, wfall, base_name, fs:float, fc:float, start_samp:int, verbose=False)->Path:
    t_start = start_samp / fs
    prefix = f"{base_name}_part{idx:03d}_T{t_start:08.2f}s"

    # --- Збереження PNG (SDR++ Style) ---
    plt.figure(figsize=(10, 20)) # Вертикальний "рулон"
    f_range = (fs / 2) / 1e6
    t_dur = (len(wfall) * (fft_n * batch_n)) / fs
    
    img = plt.imshow(wfall, aspect='auto', interpolation='nearest', cmap='magma',
                     extent=[fc/1e6 - f_range, fc/1e6 + f_range, t_start + t_dur, t_start],
                     vmin=-100, vmax=-40) # Підлаштуй під свій шум
    
    plt.colorbar(img, label='dBFS', pad=0.02)
    plt.title(f"Part {idx} | Offset: {t_start:.2f}s")
    out=out_dir / f"{prefix}.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    if verbose:
        print(f"{INFO} W-fall saved to {YELLOW}{out}{RESET}")
    return out


def _form_artefact_folder_path(sign_fpath:Path)->Path:
    return fr.f_path.parent / ("." + fr.f_path.stem)

def _form_bwbusrt_file_path(sign_fpath:Path, fft_n:int, batch_n:int)->Path:
    return _form_artefact_folder_path(sign_fpath) / f"bw_busrts_{fft_n}x{batch_n}.json"

def _save_burst_pack(fr: FReader, f_key: int, folder: Path, burst_packs: Dict[int, List[PackBwBurst]]):
    if f_key not in burst_packs:
        return

    initial_sample_pos = fr.curr_sampl_pos
    
    packs = sorted(burst_packs[f_key], key=lambda x: x.start_samp_pos)
    out_path = folder / f"bw_{f_key/1e6:.2f}MHz_Fs_{fr.Fs/1e6}MHz.bin"
    if len(packs) == 0:
        print(f"  {f_key/1e6:8.2f} MHz -> {RED} no data{RESET}")
        return
    try:
        with open(out_path, 'wb') as f_out:
            for p in packs:
                fr.jump_to_samp_pos(p.start_samp_pos)
                
                byte_size = p.duration_samples * 4
                chunk = fr._file.read(byte_size)
                
                if chunk:
                    f_out.write(chunk)
    finally:
        # Повертаємось у вихідну точку через твій метод
        fr.jump_to_samp_pos(initial_sample_pos)

    print(f"  [Save] {f_key/1e6:8.2f} MHz -> {out_path.name}")
    
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
    spec_windowing:  Literal['hann', 'hamming', 'blackmanharris', 'rect'] = 'rect',
    sz_lim_mb: int = 500 # for wfall files
):
    dF = Fs / fft_n
    chunk_len = fft_n * batch_n
    f32_buf: IQInterleavedF32 = np.empty(fft_n * batch_n * 2, dtype=np.float32)
    freq_bins_full = (np.arange(fft_n, dtype=np.float32) - (fft_n // 2)) * dF + Fc
    print(f"Chunk dur: {CYAN}{1e3*chunk_len/Fs}{RESET} ms ()")
    _P_FS = float(fft_n)**2  # Константа для 0 dBFS (максимальна потужність комплексного тону)
    # INT16_FULL_SCALE == 32768
    if spec_windowing == 'rect':
        wnd_coeffs = np.ones(fft_n, dtype=np.float32) / INT16_FULL_SCALE
    else:
        raw_wnd = windows.get_window(spec_windowing, fft_n, fftbins=True)
        # Поєднуємо нормалізацію вікна (mean) та амплітуди (INT16)
        wnd_coeffs = (raw_wnd / (np.mean(raw_wnd) * INT16_FULL_SCALE)).astype(np.float32)
    band_lst:List[BandwidtBurst] = []
    
    report_dir: Path = _form_artefact_folder_path(fr.f_path)
    report_dir.mkdir(parents=True, exist_ok=True)
    cashed_bw = _form_bwbusrt_file_path(fr.f_path, fft_n, batch_n)
    
    if cashed_bw.exists():
        print(f"{INFO} Artifacts store: {YELLOW}{report_dir}{RESET}")
        band_lst = load_bursts(cashed_bw)
        if band_lst:
            # reanalyze_csv(csv, fs=Fs)
            # sort by start idx and make dict where centre is key and val is list of bands
            burst_packs = aggregate_to_packs(band_lst, chunk_len, fr.samples_total, Fs, verbose=True)
            # analyze_pack_intervals(burst_packs, Fs, verbose=True)
            for f_key in burst_packs.keys():
                _save_burst_pack(fr, f_key=f_key, folder=report_dir, burst_packs=burst_packs)
        else:
            print(f"{WARN} No data finded on firs run try with manualy adjust detection parameters.")
            cashed_bw.unlink()
        print()
        return

            
    #  w-fall prework
    # Вирівнюємо до кратного chunk_len, щоб сесія закінчувалась цілим батчем
    # 1. Точний розрахунок розміру сесії
    samples_per_session = ((sz_lim_mb * 1024 * 1024 // 4) // chunk_len) * chunk_len
    rows_per_session = samples_per_session // chunk_len
    
    # 2. Pre-allocation: рівно стільки, скільки батчів пройде через цикл
    wfall_memory = np.full((rows_per_session, fft_n), -120.0, dtype=np.float32)
    wfall_ptr = 0

    session_idx = 0
    start_samp_pos = fr.curr_sampl_pos

    n_read = 0
    pbar = tqdm(total=fr.samples_total, desc="Processing IQ")
    while True:
        n_read = fr.read_samples_into(f32_buf, chunk_len)
        if n_read != chunk_len or _break_req:
            break
        curr_block_start = fr.curr_sampl_pos
        
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
        
        wfall_memory[wfall_ptr, :] = y_spec
        wfall_ptr += 1
        if wfall_ptr >= rows_per_session:
            save_session_artifacts_wfall(report_dir, session_idx, wfall_memory, fr.f_path.stem, Fs, Fc, start_samp_pos, verbose=True)
            # Скидання для нової сесії
            session_idx += 1
            wfall_ptr = 0
            start_samp_pos = fr.curr_sampl_pos
            wfall_memory.fill(-120.0)
        # Depends on signal record BW and need msnual adjuct
        p10 = np.percentile(y_spec, 10)
        pHigh = np.percentile(y_spec, 50) # 75
        iqr = pHigh - p10
        has_sign = iqr > threshold_iqr
        if has_sign:
            # --- РОЗРАХУНОК OCCUPANCE ---
            # Визначаємо адаптивний поріг детектування сигналу. 
            # p75 + 0.5*iqr — це досить чутливий поріг, який адаптується під рівень шуму.
            # thr = pHigh + (0.25 * iqr)    # freq ocupance < 0.75*Fs
            thr = pHigh - (0.25 * iqr)      # freq ocupance > 0.75*Fs
            # Рахуємо кількість бінів, що перевищують цей поріг
            active_bins = np.sum(y_spec > thr)
            # Occupance — це відношення активних бінів до загальної кількості (від 0.0 до 1.0)
            occupance = active_bins / len(y_spec)
            _bnd_lst:List[BandwidtBurst] = find_bands(y_spec_smooth, freq_bins_full, thr)
            if occupance*Fs > min_bw:
                for bnd in _bnd_lst: bnd.start_samp_pos = fr.curr_sampl_pos
                band_lst.extend(_bnd_lst)

        pbar.update(chunk_len)
    
    pbar.close()
    if _break_req:
        print(f"\n  Ctrl+C breaking")
    elif n_read != chunk_len:
        print(f"\n  {n_read=:_} not match chunk {fft_n}x{batch_n}x2. EOF, skip chunk.")
    else:
        print(f"\n  EOF")
    if wfall_ptr > 0:
        print(f"{INFO} Saving final chunk ({wfall_ptr} rows)...")
        # Передаємо тільки фактично заповнену частину через слайсинг [:wfall_ptr]
        save_session_artifacts_wfall(
            report_dir, 
            session_idx, 
            wfall_memory[:wfall_ptr], 
            fr.f_path.stem, 
            Fs, Fc, 
            start_samp_pos,
            verbose=True
        )
    
    save_bursts(band_lst, cashed_bw)
    print(f"{INFO} {GRAY}Run again with same params for spread sig by hop freqs.{RESET}\n\n")


# CLI
def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="Analize Stage 1",
        description="Spec Analize pgase I: spectral analysis, build wfall finherprint, detect and spread bursts\bandwidth, for IQ recordings (.bin/.wav).",
    )

    p.add_argument(
        "-i", "--input", dest="file", type=Path, default=_fpath,
        help="Path to IQ file: .bin or .wav",
    )
    
    p.add_argument(
        "--Fs", dest="samp_rate", type=float, default=None,
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
    
    return p

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
            dur_sec = fr.samples_total / args.samp_rate
            print(f"{INFO} Input file: {YELLOW}{args.file}{RESET}. Fs={args.samp_rate/1e6} MHz. df={1e-3*args.samp_rate/fft_n:.2} KHz. Dur={dur_sec:.2f} s")
            do_file_stage_I(fr, Fs=args.samp_rate, Fc=args.Fc, fft_n=args.fft_n, batch_n=args.batch_n, min_bw=args.bw)

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
