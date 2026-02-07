# src/analize_stage_I.py
from dataclasses import dataclass
import json
import sys, os
import argparse
from pathlib import Path
import signal
from time import monotonic
import traceback
from typing import Dict, List, Literal, Optional
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np

import scipy.signal.windows as windows
from scipy.ndimage import gaussian_filter1d
from scipy.fft import fft as sfft
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
from scipy.ndimage import binary_opening, binary_closing
    
from analize_stage_I import _form_artefact_folder_path
from analize_stage_II import Burst
from fft_core import INT16_FULL_SCALE, IQInterleavedF32

from helpers import print_ascii_hist
from io_stuff import FReader


from vsa_file_entry import _apply_vsa_file_contract, show_cli
from colorizer import colorize, inject_colors_into
from vsa_pro import deploy_vsa_pro
# --- color names for IDE/static analysis suppress warnings --------------------
GREEN: str; BRIGHT_GREEN: str; RED: str; BRIGHT_RED: str
YELLOW: str; BRIGHT_YELLOW: str; BLACK: str; BRIGHT_BLACK: str
GRAY: str; BRIGHT_GRAY: str; CYAN: str; BRIGHT_CYAN: str
BLUE: str; BRIGHT_BLUE: str; MAGENTA: str; BRIGHT_MAGENTA: str
WHITE: str; BRIGHT_WHITE: str; RESET: str; WARN: str; ERR: str; INFO: str; DBG: str
inject_colors_into(globals())



    
def do_burst_spread(
    fr: FReader,
    Fs: float,
    Fc: float = 0.0,
    s: int | None = None,
    e: int | None = None,

    wnd_analize_dur: float = 0.2e-5,     # seconds
    freq_tol: float = 2e3 # Hz
):
    start = s if s is not None else 0
    end = e if e is not None else fr.samples_total
    total_samp = end - start
     # INFO про зчитуваний діапазон
    dur_total = fr.samples_total / Fs
    dur_read = total_samp / Fs
    pct = 100 * total_samp / fr.samples_total
    if start > 0:
        fr.jump_to_samp_pos(start)
    mem_bytes = total_samp * 2 * 4
    if mem_bytes * 8 > 512*1024**2:
        print(f"{WARN} Requested read IQ span {RED}{total_samp:_}{RESET} ({mem_bytes:_}) is bigger 512 MB. Perfrmance problems may appear. It is your own risk.")
    f32_buf: IQInterleavedF32 = np.empty(total_samp * 2, dtype=np.float32)
    res_folder = fr.f_path.parent / f".{fr.f_path.stem}"
    res_folder.mkdir(parents=True, exist_ok=True)
    # print(f"  Search SEQ range: {CYAN}{seq_dur_min*1e6:.1f} - {seq_dur_max*1e6:.1f}{RESET} us")
    # print(f"  Samples range: {YELLOW}{n_min} - {n_max}{RESET} samples")
    t0 = monotonic()
    n_read = 0
    # pbar = tqdm(total=fr.samples_total, desc="Processing IQ")
    # read int16 as samples to memory as f32
    n_read = fr.read_samples_into(f32_buf, total_samp)
    if n_read != total_samp:
        print(f"{ERR} Error read file {YELLOW}{fr.f_path}{RESET}")
        return
    
    # 1. F32 C-Cont arr to np.complex
    iq_samples = f32_buf.view(np.complex64)
    print(f"{INFO} File read span:"
          f"\n      Total: {fr.samples_total:_} smp ({dur_total:.2f} s)"
          f"\n      Range: [{start:_}, {end:_}) = {total_samp:_} smp ({dur_read:.2f} s, {pct:.1f}%)"
          f"\n      Start offset: {start/Fs*1e3:.2f} ms")
    
    # ==============================================================================
    # BURST DETECTION PARAMETERS
    # ==============================================================================
    # === OFDM Geometry (виміряно з реального сигналу) ===
    BURST_DUR = 720.0e-6      # Тривалість одного бурсту (секунди)
    BURST_PAUSE = 300.0e-6    # Пауза між бурстами (секунди)
    _burst_wnd = int(BURST_DUR * Fs)        # Очікувана довжина бурсту (семпли)
    _pause_wnd = int(BURST_PAUSE * Fs)      # Довжина паузи (семпли)

    # === Fine-tune параметри детекції ===
    THR = 20.0                  # Поріг (dB) для відділення сигналу від шуму
                            # Має бути між noise floor (~10 dB) і signal level (~40 dB)
                            # Нижче → більше false positives, вище → втрата слабких бурстів

    burst_guard = 0.15        # Допустиме відхилення розміру бурсту (±15%)
                            # Діапазон: [_burst_wnd*(1-guard), _burst_wnd*(1+guard)]
                            # Більше → толерантність до edge effects, менше → строгість

    W = int(BURST_DUR * Fs / 20)  # Вікно усереднення для згладжування power (семпли)
                                # Менше → точніші краї але більше шуму в масці
                                # Більше → чистіша маска але розмиті краї бурстів
                                # Рекомендовано: 5-10% від BURST_DUR

    # === Морфологічні ядра (автоматично) ===
    K_closing = min(W, _pause_wnd // 2)  # Для закриття дірок ВСЕРЕДИНІ бурсту
    K_opening = W                         # Для видалення шумових сплесків
    # ==============================================================================

    print(f"{INFO} Energy-based Burst Detection with Sliding Window Averaging and Morphological Cleaning"
         f"\n      Window {GREEN}{W}{RESET} samples <=> {CYAN}{wnd_analize_dur*1e3}{RESET} ms."
         f"\n      Start search...")
    power = (iq_samples.real**2 + iq_samples.imag**2)
    # 1. Переводимо в dB з захистом від нулів
    p_db = 10 * np.log10(power + 1e-12)
    tst_arr = uniform_filter1d(p_db, size=W, mode='constant', cval=0, origin=-(W//2))
    # thr = tst_arr.mean()
    # thr = np.percentile(tst_arr, 70) 
    # Використати:
    # noise_floor = np.percentile(tst_arr, 10)  # 10-й перцентиль = шум
    # signal_level = np.percentile(tst_arr, 90)  # 90-й = сигнал
    # thr = (noise_floor + signal_level) / 2  # Посередині

    # 1. Створюємо базову логічну маску
    mask = (tst_arr > THR)

    # 2. "дивитись ліворуч/праворуч":
    # Використовуємо морфологічне відкриття (opening), щоб прибрати "пчихи" на шумі,
    # та закриття (closing), щоб прибрати "дірки" в бурсті.



    # Спочатку зшиваємо дрібні провали всередині бурсту
    mask_closed = binary_closing(mask, structure=np.ones(K_closing))
    # Потім видаляємо короткі сплески на шумі
    final_mask = binary_opening(mask_closed, structure=np.ones(K_opening))
    dur1 = monotonic() - t0
    print(f"Stage 1 dur {dur1:.3f} s")
    if 0:
        K = 50_000
        plt.ion()
        plt.figure(figsize=(15, 8))
        plt.subplot(4, 1, 1)
        plt.plot(power[:K])
        plt.title('Raw power')

        plt.subplot(4, 1, 2)
        plt.plot(tst_arr[:K])
        plt.axhline(THR, color='r', label='threshold')
        plt.title('Smoothed power (dB)')

        plt.subplot(4, 1, 3)
        plt.plot(mask[:K])
        plt.title('Initial mask')

        plt.subplot(4, 1, 4)
        plt.plot(final_mask[:K])
        plt.title('After morphology')
        plt.show(block=False)
    # 3. Тепер знаходимо старти і стопи по ОЧИЩЕНІЙ масці
    diff_sign = np.diff(final_mask.astype(np.int8))
    starts = np.where(diff_sign > 0)[0]
    ends = np.where(diff_sign < 0)[0]
    
    bursts: Dict[int, List[Burst]] = {}
    burst_discarded: List[Burst] = []
    n_bursts = min(len(starts), len(ends))
    _burst_wnd = int(BURST_DUR * Fs)

    n_guard = int( burst_guard*BURST_DUR * Fs)
    _burst_min = int( (1-burst_guard)*BURST_DUR * Fs)
    _burst_max = int( (1+burst_guard)*BURST_DUR * Fs)
    freqs_cache = {}
    for i in range(n_bursts):
        s = int(starts[i])
        e = int(ends[i])
        l = e - s
        d = l / Fs  # Розрахунок dur
        seg_iq = iq_samples[s:e]
        burst = Burst(id=i, start=s, end=e, length=l, duration=d)
        if not (_burst_min < seg_iq.size < _burst_max):
            print(f"{WARN} Discard burst {i:_}/{n_bursts:_}:"
                  f"  Got {seg_iq.size} smp, expected {_burst_wnd}. Offset: {burst.start}")
            burst_discarded.append(burst)
            continue
        # FFT and find burst center in freq domain (rel to 0.0)
        fft_result = np.fft.fft(seg_iq)
        power_spectrum = fft_result.real**2 + fft_result.imag**2
        n = len(seg_iq)
        if n not in freqs_cache:
            freqs_cache[n] = np.fft.fftfreq(n, 1/Fs)
        freqs = freqs_cache[n]
        burst.center_freq = np.sum(freqs * power_spectrum) / np.sum(power_spectrum)
        f_key = int(burst.center_freq)
        if f_key not in bursts:
            bursts[f_key] = []
        bursts[f_key].append(burst)

    # bursts_ln = np.array([b.length for b in burst_list])
    # ln_mean = int(bursts_ln.mean())
    dur_search = monotonic() - t0
    

    all_bursts = [b for burst_list in bursts.values() for b in burst_list]
    n_valid = len(all_bursts)
    n_discarded = len(burst_discarded)
    n_total = n_valid + n_discarded

    if n_valid > 0:
        bursts_ln = np.array([b.length for b in all_bursts])
        ln_mean = bursts_ln.mean()
        dur_search = monotonic() - t0
        
        print(f"{INFO} Burst count: {GREEN}{n_valid}{RESET}/{n_total} (discarded: {n_discarded})."
            f"\n      Length: min={bursts_ln.min()}, max={bursts_ln.max()}, mean={ln_mean:.0f} ({1e3*ln_mean/Fs:.3f} ms)."
            f"\n      Frequency groups: {len(bursts)} distinct centers."
            f"\n      [Time: {dur_search*1e3:.2f} ms]")
    else:
        print(f"{WARN} No valid bursts found. Total detected: {n_total}, all discarded.")
    
    # print_ascii_hist(bursts_ln, Fs)
    # bursts overlay
    prefix_visual = int(0.025e-3*Fs)
    K = n_bursts - 1
    # K //= 50


    normalize = 32768.0
    

            
    for i, b in enumerate(all_bursts[:K]):
        # --- DISCARD LOGIC (Happy Path Protection) ---

        # Вирізаємо комплексний шматок для фази та I/Q
        slc = slice(b.start - n_guard, b.start + _burst_wnd + n_guard)
        # Отримуємо зріз
        seg_iq = iq_samples[slc]

    print(f"{BRIGHT_MAGENTA}============================{RESET} ")
    return

# CLI
def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=f"{Path(sys.argv[0]).stem}.py",
        description="Analize Stage II: BaseBand OFDM record for investigate autocorr for IQ recordings (.bin/.wav).",
    )

    p.add_argument(
        "-i", "--input", dest="file", type=Path, required=True,
        help="Path to IQ file: .bin or .wav",
    )
    
    p.add_argument(
        "-r", "--samp-rate", dest="samp_rate", type=float, default=None,
        help="Sample rate (Hz). Required for .bin; for .wav taken from header (if provided: must match).",
    )
    
    p.add_argument(
        "--dtype", dest="dtype", type=str, default="int16",
        help="Expected sample dtype. For .wav mismatch -> exception. (Current reader: int16 only)",
    )

    p.add_argument(
        "-Fc", "--center-freq", dest="Fc", type=float, default=0.0,
        help="Center frequency (Hz). Default: 0"
    )

    p.add_argument(
        "-s", "--start", dest="s", type=int, default=None,
        help="Start from sample"
    )
    
    p.add_argument(
        "-e", "--end", dest="e", type=int, default=None,
        help="End at sample"
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

    show_cli()
    try:
        args: argparse.Namespace = _build_cli().parse_args()
        args = _apply_vsa_file_contract(args)

        with FReader(args) as fr:
            dur_sec = fr.samples_total / args.samp_rate
            msg = f"Input file: {YELLOW}{args.file}{RESET}. Fs={args.samp_rate/1e6} MHz. Dur={dur_sec:.2f} s."
            if args.Fc == 0.0:
                try:
                    Fc = float(fr.f_path.name.split("baseband_")[1].split("Hz")[0])
                    msg += f" Record Fc={Fc/1e6:.3f} MHz"
                except (IndexError, ValueError):
                    Fc = 0.0
            s = args.s if args.s else 0
            e = args.e if args.e else 0
            if s > fr.samples_total:
                print(f"{WARN} start pos out of range. Fallback to 0")
            if e > fr.samples_total:
                print(f"{WARN} end pos out of range. Fallback to {fr.samples_total}")
                e = fr.samples_total
            if e-s <= 0:
                print(f"{WARN} end pos out of range. Fallback to {fr.samples_total}")
                s = 0
                e = fr.samples_total
                
            print(msg)
            do_burst_spread(fr, Fs=args.samp_rate, Fc=Fc, s=s, e=e)

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
