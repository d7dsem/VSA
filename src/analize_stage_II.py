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
from constell import ConstellationPlot
from fft_core import INT16_FULL_SCALE, IQInterleavedF32
from helpers import print_ascii_hist
from io_stuff import FReader


from vsa_file_entry import _apply_vsa_file_contract, show_cli
from colorizer import colorize, inject_colors_into
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
Fc = 0

def _analyze_burst_structure(burst: np.ndarray, offset: int):
    # Шукаємо період повторення (Lag)
    # Оскільки ми бачили ~4-5 періодів у 250 семплах, шукаємо в діапазоні 16-128
    max_lag = len(burst) // 2
    lags = np.arange(1, max_lag)
    
    # Обчислюємо ACF (Auto-Correlation Function)
    # Для швидкості на малих відрізках np.correlate достатньо
    acf = np.correlate(burst, burst, mode='full')
    acf = acf[len(acf)//2:] # Беремо праву частину
    
    # Шукаємо піки в ACF
    # Перший великий пік (після нульового) вкаже на період преамбули
    from scipy.signal import find_peaks
    peaks, props = find_peaks(np.abs(acf), height=np.abs(acf[0])*0.3)
    
    if len(peaks) > 0:
        L = peaks[0] # Ймовірний період (наприклад, 64)
        confidence = props['peak_heights'][0] / np.abs(acf[0])
        print(f"  [Burst @ {offset}] Structure found! Period L={L}, Confidence={confidence:.2f}")
    else:
        print(f"  [Burst @ {offset}] No periodic structure detected.")


def _perform_preamble_search(iq_samples: np.ndarray, n_min: int, n_max: int):
    # 1. Обчислюємо миттєву потужність
    mag_sq = np.abs(iq_samples)**2
    
    # 2. Згладжування для стабільного детектора (вікно ~1/4 від n_min)
    win_size = max(1, n_min // 4)
    kernel = np.ones(win_size) / win_size
    smoothed_pwr = np.convolve(mag_sq, kernel, mode='same')
    
    # 3. Поріг (адаптивний або фіксований)
    # Для початку візьмемо 5-кратне перевищення медіанного шуму
    threshold = np.median(smoothed_pwr) * 5.0
    active = smoothed_pwr > threshold
    
    # Знаходимо моменти зміни стану (фронти)
    diff = np.diff(active.astype(np.int8))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    # Синхронізуємо масиви
    if len(ends) > 0 and len(starts) > 0:
        if ends[0] < starts[0]: ends = ends[1:]
        min_len = min(len(starts), len(ends))
        starts, ends = starts[:min_len], ends[:min_len]

    print(f"{INFO} Found {len(starts)} potential energy regions")
    
    valid_bursts = []
    for s, e in zip(starts, ends):
        duration = e - s
        if n_min <= duration <= n_max:
            valid_bursts.append((s, e))
            
    print(f"{INFO} Bursts after temporal filtering ({n_min}-{n_max}): {len(valid_bursts)}")
    
    # Етап 2: Аналіз внутрішньої структури кожного бурсту
    for s, e in valid_bursts:
        _analyze_burst_structure(iq_samples[s:e], s)


@dataclass
class Burst:
    id: int
    start: int      # Індекс початку в семплах
    end: int        # Індекс кінця в семплах
    length: int     # Довжина в семплах (len)
    duration: float # Тривалість в секундах або мс (dur)
    
def do_file_stage_II(
    fr: FReader,
    Fs: float,
    Fc: float = Fc,
    # seconds
    wnd_analize_dur: float = 0.2e-5,
    thr_db: float = 10.0,
    
    seq_dur_min: float = 0.000020,
    seq_dur_max: float = 0.000200,
    
):
    total_samp = fr.samples_total
    f32_buf: IQInterleavedF32 = np.empty(total_samp * 2, dtype=np.float32)

    # 1. Переклад тривалості в кількість семплів (цілі числа)
    # n = duration * Fs
    ln_burst = int(wnd_analize_dur * Fs)
    W = max(16, ln_burst // 4)  # Вікно W
    step = max(4, W // 4)      # Крок
    n_min = int(seq_dur_min * Fs)
    n_max = int(seq_dur_max * Fs)

    print(f"{INFO} Energy-based Burst Detection with Sliding Window Averaging and Morphological Cleaning"
         f"\n      Window {GREEN}{W}{RESET} samples <=> {CYAN}{wnd_analize_dur*1e3}{RESET} ms."
         f"\n      Start search...")
    # print(f"  Search SEQ range: {CYAN}{seq_dur_min*1e6:.1f} - {seq_dur_max*1e6:.1f}{RESET} us")
    # print(f"  Samples range: {YELLOW}{n_min} - {n_max}{RESET} samples")
    t0 = monotonic()
    n_read = 0
    # pbar = tqdm(total=fr.samples_total, desc="Processing IQ")

    # read int16 as samples to memory as f32
    n_read = fr.read_samples_into(f32_buf, total_samp)
    if n_read != total_samp or _break_req:
        print(f"{ERR} Error raed file {YELLOW}{fr.f_path}{RESET}")
        return
    
    # 1. Готуємо сигнал
    iq_samples = f32_buf.view(np.complex64)
    power = (iq_samples.real**2 + iq_samples.imag**2)
    # 1. Переводимо в dB з захистом від нулів
    p_db = 10 * np.log10(power + 1e-12)
    tst_arr = uniform_filter1d(p_db, size=W, mode='constant', cval=0, origin=-(W//2))
    thr = tst_arr.mean()
    # 1. Створюємо базову логічну маску
    mask = (tst_arr > thr)

    # 2. "дивитись ліворуч/праворуч":
    # Використовуємо морфологічне відкриття (opening), щоб прибрати "пчихи" на шумі,
    # та закриття (closing), щоб прибрати "дірки" в бурсті.
    # 'K' - це кількість точок для перевірки (спробуй W чи W//2)
    K = W 
    # Спочатку зшиваємо дрібні провали (прибираємо сині лінії всередині)
    mask_closed = binary_closing(mask, structure=np.ones(K))
    # Потім видаляємо короткі сплески (прибираємо зелені лінії на шумі)
    final_mask = binary_opening(mask_closed, structure=np.ones(K))

    # 3. Тепер знаходимо старти і стопи по ОЧИЩЕНІЙ масці
    diff_sign = np.diff(final_mask.astype(np.int8))
    starts = np.where(diff_sign > 0)[0]
    ends = np.where(diff_sign < 0)[0]
    
    burst_list: List[Burst] = []
    n_bursts = min(len(starts), len(ends))

    for i in range(n_bursts):
        s = int(starts[i])
        e = int(ends[i])
        l = e - s
        d = l / Fs  # Розрахунок dur
        
        # Створюємо екземпляр класу і додаємо в масив
        burst_list.append(Burst(id=i, start=s, end=e, length=l, duration=d))

    bursts_ln = np.array([b.length for b in burst_list])
    ln_mean = int(bursts_ln.mean())
    dur_search = monotonic() - t0
    
    # print_ascii_hist(bursts_ln, Fs)
    print(f"{INFO} Burst count: {GREEN}{len(burst_list)}{RESET}. Lengt: {bursts_ln.min()} {bursts_ln.max()}; mean {ln_mean} ({1e3*ln_mean/Fs:.3f} ms). [TIW: {dur_search*1e3:.2f} ms]")

    # bursts overlay
    prefix_visual = int(0.025e-3*Fs)
    K = n_bursts - 1
    # K //= 50
    BURST_DUR = 0.72e-3 
    _overlay_wnd = ln_mean + ln_mean//10
    _overlay_wnd = int(BURST_DUR * Fs) # look for whole 0.72 ms
    # _overlay_wnd = 600 # slook to start
    _overlay_wnd += 2 * prefix_visual

    
    # 1. Готуємо стек для сегментів
    # Використовуємо лише ті бурсти, які довші за глибину аналізу (deep)
    stack_power = []
    stack_pahse = []
    stack_i = stack_q = None
    stack_i = []
    stack_q = []

    normalize = 32768.0
    # Try constell
    const_plot = None
    # const_plot = ConstellationPlot(fr.f_path)
    if const_plot:
        plt.ion()
        const_plot.show()
        chunk_len = 1000
        
        for i in range(0, len(iq_samples), chunk_len):
            # 1. Беремо зріз (це все ще int16)
            seg_iq = iq_samples[i : i + chunk_len]
            if seg_iq.size == 0: break
            norm_chunk = seg_iq / normalize
            if not np.iscomplexobj(norm_chunk):
                # Пакуємо: парні - Real, непарні - Imag
                norm_chunk = norm_chunk[0::2] + 1j * norm_chunk[1::2]
            const_plot.update(norm_chunk, f"Offset: {i:_}")
            plt.pause(0.001)
            
    for i, b in enumerate(burst_list[:K]):
        # if b.length >= deep:
        if 1:
            # Вирізаємо комплексний шматок для фази та I/Q
            start = b.start
            start_offs = max(start - prefix_visual, 0)
            seg_iq = iq_samples[start_offs : start_offs + _overlay_wnd]
            # phase_data =np.angle(seg_iq)
            diff_iq = seg_iq[1:] * np.conj(seg_iq[:-1])
            phase_data = np.angle(diff_iq)
            
            stack_power.append(p_db[start_offs :start_offs + _overlay_wnd])
            stack_pahse.append(phase_data)
            
            if const_plot: 
                norm_chunk = seg_iq / normalize
                # norm_chunk = seg_iq
                const_plot.update(norm_chunk, f"Burst {i:_} of {K:_}")
                plt.pause(0.001)
            if stack_i: stack_i.append(seg_iq.real)
            if stack_q: stack_q.append(seg_iq.imag)
    # 2. Перетворюємо та обчислюємо (використовуємо медіану для фази — вона стійкіша)
    stack_p_arr = np.array(stack_power)
    stack_ph_arr = np.array(stack_pahse)
    stack_i_arr = np.array(stack_i)
    stack_q_arr = np.array(stack_q)

    median_ph_vec: np.ndarray = np.median(stack_ph_arr, axis=0)
    patterns = {
        "Power (dB)": (stack_p_arr, "red"),
        "Phase (rad)": (stack_ph_arr, "green"),
        "I Component": (stack_i_arr, "blue"),
        "Q Component": (stack_q_arr, "orange")
    }
    # Pattern of dPhase template params /In samples/
    pat_len = 205
    pat_start_idx = prefix_visual + 10
    pat_end_idx = pat_start_idx + pat_len

    # Burst Overlay Analysis: Fine tune patrern len scaffold
    if 1:
        # burst_end_idx = int(0.72e-3 * Fs) + prefix_visual # _baseband_1330000000Hz_15-57-04_02-02-2026_2ant
        # burst_end_idx = int(0.72e-3 * Fs) + prefix_visual # _baseband_1331012500Hz_16-46-58_02-02-2026
        burst_end_idx = int(0.72e-3 * Fs) + prefix_visual # _5MHz-11-14-25
        # fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
        fig, axes = plt.subplots(len(patterns), 1, figsize=(12, 8), sharex=True)
        f_id = f"{fr.f_path.parent.name}/{fr.f_path.name}"
        fig.suptitle(f"Burst Overlay Analysis. File: {f_id}. Burst dur: {BURST_DUR*1e3:.2f} ms."
                     f"\nK={len(stack_power)}, dur {1e3*_overlay_wnd/Fs:.3f} ms, n_samp {_overlay_wnd:_}, {pat_len=} ", fontsize=12)
        def on_key(event):
            if event.key == 'escape':
                plt.close(event.canvas.figure)
                print(f"{INFO} Figure closed by Escape key.")

        fig.canvas.mpl_connect('key_press_event', on_key)
        for ax, (title, (data, clr)) in zip(axes, patterns.items()):
            # Фоновий розкид (перші 30 штук)
            for i in range(min(len(data), 30)):
                ax.plot(data[i], color='gray', alpha=0.05)
            
            # Головний паттерн (Медіана)
            ax.plot(np.median(data, axis=0),color=clr, linewidth=.75, label='Median Pattern',
                    marker="o", markersize=1.5, markevery=4)
            ax.set_title(title)
            ax.grid(True, alpha=0.2)
            # ax.legend(loc='upper right')
            ax.axvline(pat_start_idx, color='magenta', linestyle='--', label=f'START({pat_start_idx})')
            ax.axvline(pat_end_idx, color='magenta', linestyle='--', label=f'END({pat_end_idx})')
            ax.axvline(burst_end_idx, color='cyan', linestyle='--', label=f'END({burst_end_idx})')
        plt.xlabel("Samples from Start")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
    
    gold_pattern: np.ndarray = median_ph_vec[pat_start_idx : pat_end_idx]
    pat_file = fr.f_path.parent / f"pre_pat_{fr.f_path.stem}.json"
    with open(pat_file, "w") as f:
        phase_pattern = gold_pattern.tolist()
        pat_info = {
            "Fs": f"{Fs/1e6:.1f} MHz",
            "len": len(gold_pattern),
            "dur": f"{len(gold_pattern) / Fs: .3f} ms",
            "phase_pattern" : phase_pattern
        }
        json.dump(pat_info, f, indent=4)

    # Corelation of gold_pattern
    if 0:
        # 1. Визначаємо точку відліку - початок K-го бурсту
        K_idx = n_bursts // 50
        target_burst_start = burst_list[K_idx].start
        
        # Беремо весь шматок від 0 до точки детекції K-го бурсту (+ запас для видимості самого бурсту)
        search_end = min(target_burst_start + _overlay_wnd, len(iq_samples))
        search_iq = iq_samples[0 : search_end]
        
        # Обчислюємо диференціальну фазу для всього шматка
        # Використовуємо ту саму логіку, що і для gold_pattern
        search_diff_ph = np.angle(search_iq[1:] * np.conj(search_iq[:-1]))
        
        # 2. Кореляція
        # mode='same' щоб зберегти індексацію часу (центр шаблону)
        corr_res = np.correlate(search_diff_ph, gold_pattern, mode='same')
        corr_mag = np.abs(corr_res)

        # 3. Візуалізація
        fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
        fig.suptitle(f"Long-range Correlation: 0 to Burst #{K_idx}", fontsize=16)

        # Фаза сигналу
        axes[0].plot(search_diff_ph, color='green', linewidth=0.3)
        axes[0].set_title("Differential Phase (0 : K_start)")
        axes[0].grid(True, alpha=0.2)

        # Магнітуда кореляції
        axes[1].plot(corr_mag, color='red', linewidth=0.8)
        axes[1].set_title("Correlation Peaks")
        axes[1].grid(True, alpha=0.2)

        def on_key(event):
            if event.key == 'escape':
                plt.close(event.canvas.figure)
                print(f"{INFO} Figure closed by Escape key.")
        
        fig.canvas.mpl_connect('key_press_event', on_key)

        plt.xlabel("Samples from File Start")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
        
    if 0:

        N = 100_000
        plt.figure(figsize=(12, 6))

        plt.plot(p_db[:N], color='lightgray', alpha=0.7,)
        plt.plot(tst_arr[:N], color='red', alpha=0.7,)
        # # Накладаємо маску (масштабуємо її під графік)
        # plt.fill_between(np.arange(N), 0, power[:N].max(), 
        #                 where=m_pwr[:N] > 0, color='red', alpha=0.3, label='IQR Detection')

        plt.axhline(thr, linestyle="--", color='red', label="mean")
        
        for v in starts[starts < N]:
            plt.axvline(v, linestyle="--", color='green', alpha=0.5)
        for v in ends[ends < N]:
            plt.axvline(v, linestyle="--", color='blue', alpha=0.5)

        plt.title(f"Detector (W={W}, Thr={thr_db}dB)")
        plt.show()


    return
# CLI
def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=f"{Path(sys.argv[0]).stem}.py",
        description="Analize Stage II: BaseBand OFDM record for investigate autocorr for IQ recordings (.bin/.wav).",
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
            print(f"Input file: {YELLOW}{args.file}{RESET}. Fs={args.samp_rate/1e6} MHz. Dur={dur_sec:.2f} s")
            do_file_stage_II(fr, Fs=args.samp_rate, Fc=args.Fc,)

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
