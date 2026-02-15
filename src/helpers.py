from collections import defaultdict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from time import monotonic
import csv
from datetime import datetime
from typing import Dict, List, Literal, Optional, Tuple

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import pandas as pd
import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d
from scipy.ndimage import binary_opening, binary_closing
from scipy.signal import savgol_filter

from colorizer import colorize, inject_colors_into
# --- color names for IDE/static analysis suppress warnings --------------------
GREEN: str; BRIGHT_GREEN: str; RED: str; BRIGHT_RED: str
YELLOW: str; BRIGHT_YELLOW: str; BLACK: str; BRIGHT_BLACK: str
GRAY: str; BRIGHT_GRAY: str; CYAN: str; BRIGHT_CYAN: str
BLUE: str; BRIGHT_BLUE: str; MAGENTA: str; BRIGHT_MAGENTA: str
WHITE: str; BRIGHT_WHITE: str; RESET: str; WARN: str; ERR: str; INFO: str; DBG: str
inject_colors_into(globals())

# === OFDM Geometry  (секунди, "виміряно" з реального сигналу) ===
PRE_DUR = 20.48e-6 # depends on sign BW: on 5 MHz - x2 of 2.5 MHz
PRE_DUR = 19.6e-6 # depends on sign BW: on 5 MHz - x2 of 2.5 MHz
SYM_PAYLOAD_DUR = 327.68e-6 # Виміряно, дає 1024 точки на 3,125 МГц
CP_DUR =  SYM_PAYLOAD_DUR / 16
SYM_DUR = SYM_PAYLOAD_DUR + CP_DUR

BURST_DUR = SYM_DUR*2+PRE_DUR   # Тривалість одного бурсту
BURST_PAUSE = 300.0e-6          # Пауза між бурстами орієнтовно

GUARD_DUR = 20e-6

TRANSMITTER_SAMP_RATE = 3.125e6  # Оцінка на основі тривалості OFDM sym payload and OFDM fft 1024
# =================


def resample(iq: np.ndarray, samp_rate_inp: float, samp_rate_out: float, mode: Literal["fft_dased", "polyphase", "lin_interp"] = "polyphase") -> np.ndarray:
    
    num_samples_out = int(len(iq) * samp_rate_out / samp_rate_inp)
    
    match mode:
        case "polyphase":
            gcd_val = np.gcd(int(samp_rate_out), int(samp_rate_inp))
            up = int(samp_rate_out // gcd_val)
            down = int(samp_rate_inp // gcd_val)
            return signal.resample_poly(iq, up, down)
        
        case "fft_dased":
            return signal.resample(iq, num_samples_out)
        
        case "lin_interp":
            old_idx = np.arange(len(iq))
            new_idx = np.linspace(0, len(iq) - 1, num_samples_out)
            return np.interp(new_idx, old_idx, iq.real) + 1j * np.interp(new_idx, old_idx, iq.imag)
        
        case _:
            raise ValueError(f"Unknown mode: {mode}")


def sec2str(t_sec:float)->str:
    if t_sec < 1e-3: t_str = f"{t_sec * 1e6:.3f} us"
    elif t_sec < 1.0: t_str = f"{t_sec * 1e3:.3f} ms"
    else: t_str = f"{t_sec:.3f} s"
    return t_str


def print_ascii_hist(data: np.ndarray, Fs: float | None = None, label: str="<placeholder>", bins: int = 20, width: int = 50):
    if data.size == 0: 
        return
    
    counts, bin_edges = np.histogram(data, bins=bins)
    max_count = counts.max()
    
    # Формуємо заголовок залежно від наявності Fs
    header = f"\n{label:<20} | {'Count':<6} | {'Distribution':<{width}}"
    if Fs is not None:
        header += " | Duration"
    
    print(header)
    print("-" * len(header))
    
    for i in range(bins):
        # 1. Текстовий діапазон семплів
        range_str = f"{int(bin_edges[i]):>8} - {int(bin_edges[i+1]):<8}"
        
        # 2. ASCII-бар з фіксованою шириною (padding)
        bar_len = int(counts[i] * width / max_count) if max_count > 0 else 0
        bar = "#" * bar_len
        pad = " " * (width - bar_len)
        
        # 3. Базовий рядок
        ts = f"{range_str} | {counts[i]:<6} | {bar}{pad}"
        
        # 4. Додаємо час, якщо є Fs
        if Fs is not None:
            bin_center = (bin_edges[i] + bin_edges[i+1]) / 2
            dur_ms = (bin_center / Fs) * 1000
            ts += f" | ~{dur_ms:8.3f} ms"
            
        print(ts)


def find_cfo(psd: np.ndarray, freqs: np.ndarray) -> float:
    """
    NOTE: psd is in linears scale!
    """
    psd = np.asarray(psd, dtype=float)
    freqs = np.asarray(freqs, dtype=float)

    assert psd.shape == freqs.shape, "psd and freqs must have the same shape"

    total_power = np.sum(psd)
    assert total_power > 0.0, "Total spectral power must be positive"

    return float(np.sum(freqs * psd) / total_power)

@dataclass
class Burst:
    id: int
    start: int      # Індекс початку в семплах
    end: int        # Індекс кінця в семплах
    length: int     # Довжина в семплах (len)
    duration: float # Тривалість в секундах або мс (dur)
    center_freq: float|None = None # Частотний центр 
    
def locate_burst_by_energy(
    iq_samples:np.ndarray[np.complex64], Fs:float, *,
    burst_dur_est: float = BURST_DUR, burst_pause_est: float = BURST_PAUSE,
    thr_db = 25.0,
    burst_dur_tol = 0.2,
    wnd_avr_coef: float =0.05,
    verbose:bool = False
)->Tuple[List[Burst], List[Burst]]:
    """
    Power-based burst detection with morphological filtering and length validation.
    
    OPERATING ENVELOPE:
    - Stable noise floor across capture (THR manually tuned per file)
    - Consistent burst duration (±20% tolerance via burst_dur_tol)
    - SNR sufficient for fixed threshold separation (signal >> THR >> noise)
    - Power-only detection adequate (no phase coherence required)
    
    TUNABLE PARAMETERS:
    thr_db (dB):           Signal/noise separation threshold. 
                        ↑ → fewer false positives, risk missing weak bursts
                        ↓ → catch weak signals, more noise detections
                        Tune: Check noise floor in spectrum, set between noise and signal

    burst_dur_tol (0-1):   Length tolerance (±burst_dur_tol*burst_dur_est).
                        ↑ → accept more variable-length bursts
                        ↓ → stricter filtering, reject edge-degraded bursts
                        Tune: Check discarded histogram for systematic bias

    wnd_avr_coef (0-1):    Smoothing window as fraction of burst (W = wnd_avr_coef*burst_dur_est*Fs).
                        ↑ → cleaner mask, blurred edges, may merge close bursts
                        ↓ → precise edges, noisier mask, more false triggers
                        Tune: Visual check mask vs raw power plot
                        Typical: 0.05-0.10 (5-10% of burst duration)

    K_closing/opening:  Auto-sized from W and _pause_wnd. Touch only if morphology fails.
    
    FAILURE MODES:
    - Variable SNR → fixed THR misses weak or over-triggers on strong sections
    - Overlapping bursts → merged into one long burst → discarded by length check
    - Non-rectangular envelopes → edge timing shifts ±(W/2) samples
    
    Returns:
        (burst_valid, burst_discarded): Lists of Burst objects with timing and CFO
    """
    # ==============================================================================
    # BURST DETECTION PARAMETERS
    # ==============================================================================
    _burst_wnd = int(burst_dur_est * Fs)        # Очікувана довжина бурсту (семпли)
    _pause_wnd = int(burst_pause_est * Fs)      # Довжина паузи (семпли)

    # === Fine-tune параметри детекції ===
    _thr_db = thr_db        # Поріг (dB) для відділення сигналу від шуму
                            # Має бути між noise floor (~10 dB) і signal level (~40 dB)
                            # Нижче → більше false positives, вище → втрата слабких бурстів

    _burst_guard = burst_dur_tol        # Допустиме відхилення розміру бурсту
                            # Діапазон: [_burst_wnd*(1-guard), _burst_wnd*(1+guard)]
                            # Більше → толерантність до edge effects, менше → строгість

    W = int(wnd_avr_coef* burst_dur_est * Fs)  # Вікно усереднення для згладжування power (семпли)
                                # Менше → точніші краї але більше шуму в масці
                                # Більше → чистіша маска але розмиті краї бурстів
                                # Рекомендовано: 5-10% від BURST_DUR
    wnd_analize_dur = W / Fs
    # === Морфологічні ядра (автоматично) ===
    K_closing = min(W, _pause_wnd // 2)  # Для закриття дірок ВСЕРЕДИНІ бурсту
    K_opening = W                         # Для видалення шумових сплесків
    # ==============================================================================
    
    if verbose: print(f"{INFO} Energy-based Burst Detection with Sliding Window Averaging and Morphological Cleaning"
         f"\n      Window {GREEN}{W}{RESET} samples <=> {CYAN}{wnd_analize_dur*1e3:6g}{RESET} ms."
         f"\n      Start search...")
    t0 = monotonic() 
    power = (iq_samples.real**2 + iq_samples.imag**2)
    # 1. Переводимо в dB з захистом від нулів
    p_db = 10 * np.log10(power + 1e-12)
    tst_arr = uniform_filter1d(p_db, size=W, mode='constant', cval=0, origin=-(W//2))

    # 1. Створюємо базову логічну маску
    mask = (tst_arr > _thr_db)

    # 2. "дивитись ліворуч/праворуч":
    # Використовуємо морфологічне відкриття (opening), щоб прибрати "пчихи" на шумі,
    # та закриття (closing), щоб прибрати "дірки" в бурсті.

    # Спочатку зшиваємо дрібні провали всередині бурсту
    mask_closed = binary_closing(mask, structure=np.ones(K_closing))
    # Потім видаляємо короткі сплески на шумі
    final_mask = binary_opening(mask_closed, structure=np.ones(K_opening))
    dur1 = monotonic() - t0
   
    # 3. Тепер знаходимо старти і стопи по ОЧИЩЕНІЙ масці
    diff_sign = np.diff(final_mask.astype(np.int8))
    starts = np.where(diff_sign > 0)[0]
    ends = np.where(diff_sign < 0)[0]
    
    burst_valid: List[Burst] = []
    burst_discarded: List[Burst] = []
    n_bursts = min(len(starts), len(ends))
    _burst_wnd = int(burst_dur_est * Fs)

    _burst_min = int( (1-_burst_guard)*burst_dur_est * Fs)
    _burst_max = int( (1+_burst_guard)*burst_dur_est * Fs)
    freqs_cache = {}
    for i in range(n_bursts):
        s = int(starts[i])
        e = int(ends[i])
        l = e - s
        d = l / Fs  # Розрахунок dur
        seg_iq = iq_samples[s:e]
        burst = Burst(id=i, start=s, end=e, length=l, duration=d)
        # FFT and find burst center in freq domain (rel to 0.0)
        fft_result = np.fft.fft(seg_iq)
        power = fft_result.real**2 + fft_result.imag**2
        n = len(seg_iq)
        if n not in freqs_cache:
            freqs_cache[n] = np.fft.fftfreq(n, 1/Fs)
        freqs = freqs_cache[n]
        burst.center_freq = find_cfo(power, freqs)
        
        if not (_burst_min < seg_iq.size < _burst_max):
            print(f"{WARN} Discard burst {i:>6_}/{n_bursts:<6_}:  Got {seg_iq.size:^6} smp ({1e6*seg_iq.size/Fs:7.3f} us), expected {_burst_wnd:^6} ({BURST_DUR*1e6:7.3f} us). Offset: {burst.start:_}")
            burst_discarded.append(burst)
            # Debug plot: manual comment/uncomment
            if 0:
            # if i in [4079, 2186]:
                from matplotlib import pyplot as plt
                start_idx = max(0, burst.start - 250)
                end_idx = min(len(power), burst.start + 25_000)  # семплів після бурсту
                slc = slice(start_idx, end_idx)
                plt.ion()
                plt.figure(figsize=(15, 8))
                plt.suptitle(f'Burst {i} at offset {burst.start:_} len={seg_iq.size}')
                plt.subplot(4, 1, 1)
                plt.plot(power[slc])
                plt.title('Raw power')

                plt.subplot(4, 1, 2)
                plt.plot(tst_arr[slc])
                plt.axhline(_thr_db, color='r', label='threshold')
                plt.title('Smoothed power (dB)')

                plt.subplot(4, 1, 3)
                plt.plot(mask[slc])
                plt.title('Initial mask')

                plt.subplot(4, 1, 4)
                plt.plot(final_mask[slc])
                plt.title('After morphology')
                plt.show(block=True)
            continue

        burst_valid.append(burst)

    if verbose:
        dur_search = monotonic() - t0
        n_valid = len(burst_valid)
        n_discarded = len(burst_discarded)
        n_total = n_valid + n_discarded
        if n_valid > 0:
            bursts_ln = np.array([b.length for b in burst_valid])
            ln_mean = bursts_ln.mean()
            print(f"{INFO} Burst count: {GREEN}{n_valid}{RESET}/{n_total} (discarded: {n_discarded})."
                f"\n      Length: min={bursts_ln.min()}, max={bursts_ln.max()}, mean={ln_mean:.0f} ({1e3*ln_mean/Fs:.3f} ms)."
                f"\n      [Time: {dur_search*1e3:.2f} ms]")
            # bursts_freqs = np.array([b.center_freq for b in burst_valid])
            # print_ascii_hist(bursts_freqs, Fs=None, label="Freqs")
        else:
            print(f"{WARN} No valid bursts found. Total detected: {n_total}, all discarded.")
        if len(burst_discarded) > 0:
            discarded_lengths = np.array([b.length for b in burst_discarded])
            print(f"\n{WARN} Discarded bursts length distribution:")
            print_ascii_hist(discarded_lengths, Fs=Fs, bins=8, label="Length (samples)")
    return burst_valid, burst_discarded


def save_to_npz(
    path: str|Path,
    *,
    template_pwr: np.ndarray,
    template_phase: np.ndarray,
    template_i: np.ndarray,
    template_q: np.ndarray,
    Fs: float,
) -> None:
    np.savez(
        path,
        template_pwr=template_pwr.astype(np.float32, copy=False),
        template_phase=template_phase.astype(np.float32, copy=False),
        template_i=template_i.astype(np.float32, copy=False),
        template_q=template_q.astype(np.float32, copy=False),
        Fs=np.float64(Fs),
    )


def load_preamb_patterns(path: str|Path):
    try:
        d = np.load(path)
        rv = (
            d["template_pwr"],
            d["template_phase"],
            d["template_i"],
            d["template_q"],
            float(d["Fs"]),
        )
        return rv
    except Exception as e:
        print(f"{ERR} Error read saved OFDM preambula patterns: {e}")
        return None



#  Legasy scaffold
@dataclass
class BandwidtBurst:
    center: float
    wdt: float
    f_end: float
    f_start: float
    # palce holders
    start_samp_pos: int = -1 
    level_db: float = -180.0 


def save_bursts(bursts, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(
            [asdict(b) for b in bursts], 
            f, 
            indent=4, 
            # Конвертуємо будь-який тип numpy у нативний python тип
            default=lambda o: o.item() if hasattr(o, 'item') else str(o)
        )


def load_bursts(filepath):
    """Читання та відновлення об'єктів BandwidtBurst"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        # Розпаковуємо словники назад у конструктор датакласу
        return [BandwidtBurst(**d) for d in data]
    

def find_bands(curve: np.ndarray, freq_bins: np.ndarray, thr: float) -> List[BandwidtBurst]:
    """
    Знаходить неперервні ділянки (смуги), де сигналу перевищує поріг заповненості.
    """
    # 1. Створюємо маску значень вище порогу
    condition = curve > thr
    
    # 2. Знаходимо точки зміни стану (з True на False і навпаки)
    # diff дасть 1 на вході в смугу і -1 на виході
    diff = np.diff(condition.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0]
    
    # Крайові випадки: якщо сигнал почався до або закінчився після видимого спектра
    if condition[0]:
        starts = np.insert(starts, 0, 0)
    if condition[-1]:
        ends = np.append(ends, len(condition) - 1)
        
    bands = []
    for s, e in zip(starts, ends):
        if s >= e: continue # Пропускаємо помилкові сегменти
        
        f_start = freq_bins[s]
        f_end = freq_bins[e]
        
        center = (f_start + f_end) / 2
        wdt = f_end - f_start
        # Варіант А: Швидкий (середнє логарифмів)
        level_db = float(np.mean(curve[s:e]))

        # Варіант Б: Фізично точний (середнє потужностей)
        # Більш релевантний, якщо y_spec — це спектральна щільність
        # level_db = float(10 * np.log10(np.mean(10**(curve[s:e] / 10.0))))
        
        bands.append(BandwidtBurst(center=float(center), wdt=float(wdt), f_start=f_start, f_end=f_end, level_db=level_db))
        
    return bands


@dataclass
class PackBwBurst:
    center: float
    wdt: float
    f_start: float
    f_end: float
    start_samp_pos: int
    end_samp_pos: int
    level_db: float
    count_raw: int = 1      # Скільки сирих бурстів "вклеєно" в цей пак
    
    @property
    def duration_samples(self) -> int:
        return self.end_samp_pos - self.start_samp_pos


def aggregate_to_packs(
    band_lst: List[BandwidtBurst], 
    chunk_len: int,
    total_len: int,
    Fs: float,
    tolerance: int = 250_000,
    verbose=False
) -> Dict[int, List[PackBwBurst]]:
    if not band_lst:
        return {}

    # 1. Попереднє групування за частотою (Bucketing)
    buckets = defaultdict(list)
    for b in band_lst:
        f_key = int(round(b.center / tolerance) * tolerance)
        buckets[f_key].append(b)

    aggregated_results = defaultdict(list)

    for f_key, bands in buckets.items():
        raw_count = len(bands) 
        bands.sort(key=lambda x: x.start_samp_pos)
        
        current_pack: Optional[PackBwBurst] = None
        
        for b in bands:
            if current_pack is None:
                # Ініціалізація першого паку на цій частоті
                current_pack = PackBwBurst(
                    center=b.center,
                    wdt=b.wdt,
                    f_start=b.f_start,
                    f_end=b.f_end,
                    start_samp_pos=b.start_samp_pos,
                    end_samp_pos=b.start_samp_pos + chunk_len,
                    level_db=b.level_db,
                    count_raw=1
                )
                continue

            # Перевірка на неперервність (якщо наступний бурст стикується з поточним паком)
            if b.start_samp_pos <= current_pack.end_samp_pos:
                current_pack.end_samp_pos = b.start_samp_pos + chunk_len
                current_pack.level_db = max(current_pack.level_db, b.level_db)
                current_pack.count_raw += 1
            else:
                # Є розрив (gap) — фіксуємо старий пак і створюємо новий
                aggregated_results[f_key].append(current_pack)
                current_pack = PackBwBurst(
                    center=b.center,
                    wdt=b.wdt,
                    f_start=b.f_start,
                    f_end=b.f_end,
                    start_samp_pos=b.start_samp_pos,
                    end_samp_pos=b.start_samp_pos + chunk_len,
                    level_db=b.level_db,
                    count_raw=1
                )

        if current_pack:
            aggregated_results[f_key].append(current_pack)

        # Вивід статистики обсягу для бакета
        if verbose:
            packs = aggregated_results[f_key]
            # Сумарна кількість семплів, коли частота була активною
            busy_samples = sum(p.duration_samples for p in packs)
            dur_sec = busy_samples / Fs
            load_pct = (busy_samples / total_len) * 100
            
            print(f"  [Freq {f_key/1e6:8.2f} MHz] {raw_count:4d} raw -> {len(packs):3d} packs | "
                  f"Dur: {dur_sec:7.3f}s | Load: {load_pct:5.2f}%")

    if verbose:
        total_p = sum(len(v) for v in aggregated_results.values())
        print(f"--- Total packs created: {total_p} ---")

    return dict(aggregated_results)


def analyze_pack_intervals(
    packs_dict: Dict[int, List[PackBwBurst]], 
    Fs: float,
    verbose=True
):
    print(f"--- Interval Analysis (Timing Patterns) ---")
    
    for f_key, packs in packs_dict.items():
        if len(packs) < 2:
            continue
            
        # Сортуємо на всяк випадок (хоча вони мають бути сортовані)
        sorted_packs = sorted(packs, key=lambda x: x.start_samp_pos)
        
        # Рахуємо інтервали між ПОЧАТКАМИ (Start-to-Start)
        starts = np.array([p.start_samp_pos for p in sorted_packs])
        intervals_samples = np.diff(starts)
        intervals_ms = (intervals_samples / Fs) * 1e3
        
        if verbose:
            print(f"\n{CYAN}[Freq {f_key/1e6:.2f} MHz]{RESET} analyzed {len(intervals_ms)} gaps:")
            
            # Шукаємо найбільш типові інтервали (моду) через округлення
            # Наприклад, з точністю до 0.1 мс
            rounded_intervals = np.round(intervals_ms, 1)
            unique, counts = np.unique(rounded_intervals, return_counts=True)
            
            # Беремо топ-3 найчастіших інтервалів
            top_idx = np.argsort(counts)[-3:][::-1]
            
            for idx in top_idx:
                if counts[idx] > 1: # Виводимо тільки якщо інтервал повторився
                    print(f"  Pattern: {YELLOW}{unique[idx]:8.2f} ms{RESET} | Occurrences: {counts[idx]}")

            # Статистика розкиду
            print(f"  Stats: min={np.min(intervals_ms):.2f}ms, max={np.max(intervals_ms):.2f}ms, avg={np.mean(intervals_ms):.2f}ms")


def analyze_hops_and_timing(bands: List[BandwidtBurst], fs: float):
    if len(bands) < 2:
        return "Not enough data for hop analysis."

    # Сортуємо за часом (на всяк випадок)
    bands.sort(key=lambda x: x.start_samp_pos)

    positions = np.array([b.start_samp_pos for b in bands])
    centers = np.array([b.center for b in bands])
    
    # 1. ЧАСОВІ ПРОМІЖКИ (Dwell Time & Blank Time)
    deltas_samples = np.diff(positions)
    deltas_ms = deltas_samples / fs * 1000
    
    # 2. ЧАСТОТНІ СТРИБКИ (Hop Size)
    freq_diffs = np.diff(centers)
    
    # 3. ВИЗНАЧЕННЯ ПЕРІОДУ (Тут шукаємо найбільш повторюваний інтервал)
    # Округляємо до 0.1 мс, щоб згрупувати джиттер
    rounded_deltas = np.round(deltas_ms, 1)
    unique_deltas, counts = np.unique(rounded_deltas, return_counts=True)
    main_period_ms = unique_deltas[np.argmax(counts)]

    # 4. ВИЗНАЧЕННЯ СІТКИ (Grid Step)
    # Беремо модуль різниці частот (ігноруємо 0, бо це може бути той самий канал)
    active_hops = np.abs(freq_diffs[np.abs(freq_diffs) > 100e3])
    if len(active_hops) > 0:
        # Крок сітки зазвичай є найбільшим спільним дільником, 
        # але для дронів простіше знайти мінімальний крок між каналами
        grid_step_mhz = np.min(active_hops) / 1e6
    else:
        grid_step_mhz = 0

    analysis = [
        "\n--- ADVANCED HOP & TIMING ANALYSIS ---",
        f"Estimated Hop Period (T): {main_period_ms:.2f} ms (Found {np.max(counts)} times)",
        f"Estimated Symbol Rate:    {1000/main_period_ms:.1f} Hz (Hops per second)",
        f"Min Detected Grid Step:   {grid_step_mhz:.3f} MHz",
        f"Max Frequency Jump:       {np.max(np.abs(freq_diffs))/1e6:.3f} MHz",
    ]
    
    return "\n".join(analysis)


def analyze_and_export_bands(bands: List[BandwidtBurst], fs: float, base_filename: str = "vsa_report", folder:Optional[Path]=None):
    if not bands:
        print("No bands to analyze.")
        return

    # Формуємо ім'я файлу з таймстемпом
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if folder is None:
        report = Path(f"{base_filename}_{timestamp}" )
    else:
        report = folder / f"band_report" 
        
        
    csv_file = report.with_suffix(".csv")
    
    
    # 1. Експорт у CSV (Сирі дані для аналізу)
    
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['center_hz', 'width_hz', 'sample_pos', 'level_db','time_sec'])
        for b in bands:
            writer.writerow([b.center, b.wdt, b.start_samp_pos, b.level_db, b.start_samp_pos / fs])

    # 2. Формування звіту (Text Report)
    txt_file = report.with_suffix(".txt")
    
    # Розрахунки
    centers = np.array([b.center for b in bands])
    wdts = np.array([b.wdt for b in bands])
    positions = np.array([b.start_samp_pos for b in bands])
    time_deltas_ms = np.diff(positions) / fs * 1000 if len(positions) > 1 else [0]
    
    unique_c, counts = np.unique(np.round(centers/1e5)*1e5, return_counts=True)
    top_channels = sorted(zip(unique_c, counts), key=lambda x: x[1], reverse=True)[:5]

    report_content = [
        "="*60,
        f"{'OFDM SIGNAL ANALYTICS REPORT':^60}",
        "="*60,
        f"Generated:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total packets:  {len(bands)}",
        f"Freq span:      {centers.min()/1e6:.2f} to {centers.max()/1e6:.2f} MHz",
        f"Avg Bandwidth:  {wdts.mean()/1e3:.2f} kHz",
        f"Max Bandwidth:  {wdts.max()/1e3:.2f} kHz",
        "\n--- TOP ACTIVE CHANNELS ---",
    ]
    
    for f_hz, hits in top_channels:
        report_content.append(f"Freq: {f_hz/1e6:8.3f} MHz | Hits: {hits:4}")
        
    if len(time_deltas_ms) > 1:
        report_content.extend([
            "\n--- TIMING (ms) ---",
            f"Min gap:    {np.min(time_deltas_ms):.2f}",
            f"Median gap: {np.median(time_deltas_ms):.2f}",
            f"Max gap:    {np.max(time_deltas_ms):.2f}"
        ])
    
    report_content.append("="*60)
    
    # Записуємо звіт у файл та дублюємо в консоль
    full_text = "\n".join(report_content)
    with open(txt_file, "w", encoding="utf-8") as f:
        f.write(full_text)
    
    print(full_text)
    print(f"\n[DONE] Data saved to:\n  - {csv_file}\n  - {txt_file}")
    return csv_file, txt_file


def load_gold_pattern(file_path: str|Path)->Tuple[float, np.ndarray]:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Перетворюємо список пар [real, imag] назад у комплексний масив numpy
    phase_pattern = np.array(data["phase_pattern"], dtype=np.float64)
    # 2. Безпечний парсинг Fs
    fs_str = str(data.get("Fs", "0")).upper()
    # Витягуємо тільки числову частину
    import re
    val_match = re.search(r"[-+]?\d*\.\d+|\d+", fs_str)
    val = float(val_match.group()) if val_match else 0.0
    
    # Визначаємо множник
    multiplier = 1.0
    if "MHZ" in fs_str:
        multiplier = 1e6
    elif "KHZ" in fs_str:
        multiplier = 1e3
        
    fs_val = val * multiplier
    return fs_val, phase_pattern

def analyze_dwell_time(df, fs):
    # 1. Визначаємо, коли частота змінилася
    # Округляємо частоту до 100 кГц, щоб ігнорувати дрібний джиттер
    df['freq_group'] = (df['center_hz'] / 1e5).round()
    
    # Створюємо ідентифікатор "сесії" (змінюється, коли змінюється частота)
    df['session_id'] = (df['freq_group'] != df['freq_group'].shift()).cumsum()
    
    # 2. Групуємо за сесіями
    sessions = df.groupby('session_id').agg({
        'center_hz': 'first',
        'sample_pos': ['count', 'min', 'max']
    })
    
    sessions.columns = ['freq_hz', 'packets_count', 'start_samp', 'end_samp']
    
    # 3. Розраховуємо тривалість кожної сесії в мілісекундах
    sessions['duration_ms'] = sessions['packets_count'] * (df['delta_ms'].median() if 'delta_ms' in df else 0)
    # Або більш точно через позиції семплів:
    # sessions['duration_ms'] = (sessions['end_samp'] - sessions['start_samp']) / fs * 1000

    print("\n" + "="*50)
    print(f"{'DWELL TIME ANALYSIS (Time per Frequency)':^50}")
    print("="*50)
    
    # Статистика тривалості
    print(f"Average Dwell Time: {sessions['duration_ms'].mean():.2f} ms")
    print(f"Max Dwell Time:     {sessions['duration_ms'].max():.2f} ms")
    print(f"Total Hops:         {len(sessions) - 1}")
    
    print("\nMost Frequent Session Durations (Top 5):")
    top_durations = sessions['duration_ms'].round(1).value_counts().head(5)
    for dur, count in top_durations.items():
        print(f"  {dur:7.2f} ms | Occurrences: {count}")

    print("="*50)
    return sessions

def reanalyze_csv(csv_path: str|Path, fs):
    # 1. Завантажуємо дані
    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    if df.empty:
        print("File is empty.")
        return

    # 2. Базові розрахунки
    # Часові дельти між пакетами (важливо для Dwell Time)
    df['delta_ms'] = df['sample_pos'].diff() / fs * 1000
    # Стрибки частоти
    df['freq_hop_mhz'] = df['center_hz'].diff() / 1e6
    
    # --- ВИКЛИК АНАЛІЗУ DWELL TIME ---
    # Передаємо df, який вже має delta_ms
    sessions_df = analyze_dwell_time(df, fs)
    # ---------------------------------

    # 3. Вивід основної статистики (як і раніше)
    print("\n" + "="*50)
    print(f"GENERAL STATISTICS")
    print("="*50)
    print(f"Total packets:      {len(df)}")
    print(f"Avg Bandwidth:      {df['width_hz'].mean()/1e3:.2f} kHz")
    print(f"Freq range:         {df['center_hz'].min()/1e6:.2f} to {df['center_hz'].max()/1e6:.2f} MHz")
    
    main_period = df['delta_ms'].round(1).mode()
    if not main_period.empty:
        print(f"Main Packet Period: {main_period[0]:.2f} ms")
    
    active_hops = df['freq_hop_mhz'].abs()
    grid_step = active_hops[active_hops > 0.1].min()
    if not np.isnan(grid_step):
        print(f"Min Grid Step:      {grid_step:.3f} MHz")

    print("\nTop 5 Channels (MHz):")
    top_ch = df['center_hz'].round(-5).value_counts().head(5)
    for freq, count in top_ch.items():
        print(f"  {freq/1e6:8.3f} MHz | Hits: {count}")
    
    print("="*50)
    
    return df, sessions_df # Повертаємо дані, якщо захочеш побудувати графіки


def find_linear_segments(x, y, min_length=20, curvature_threshold=0.0025, verbose: bool = False):
    """
    Знайти сегменти де крива близька до прямої.
    
    Parameters
    ----------
    x : ndarray
        X координати
    y : ndarray
        Y координати
    min_length : int
        Мінімальна довжина сегменту (кількість точок)
    curvature_threshold : float
        Поріг нормалізованої кривизни для класифікації як "пряма"
    verbose : bool
        Показати відладочний графік
        
    Returns
    -------
    segments : list[dict]
        Список знайдених сегментів з полями:
        - start/end: індекси в масивах x,y
        - x_range: діапазон x
        - length: кількість точок
        - slope: нахил прямої (коефіцієнт k)
    is_straight : ndarray[bool]
        Маска: True де кривизна < threshold
    weighted_avg_slope : float
        Зважене середнє slope по довжині сегментів
    median_slope : float
        Медіана slope
    """
    # Друга похідна = кривизна
    dy = np.gradient(y, x)
    d2y = np.gradient(dy, x)
    
    # Нормалізована кривизна (щоб не залежала від масштабу)
    curvature = np.abs(d2y) / (1 + np.abs(dy)**2)**1.5
    
    # Де кривизна низька = пряма
    is_straight = curvature < curvature_threshold
    
    # Segmentify: contiguous regions
    diff = np.diff(is_straight.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    
    if is_straight[0]:
        starts = np.concatenate([[0], starts])
    if is_straight[-1]:
        ends = np.concatenate([ends, [len(x)]])
    
    segments = []
    for start, end in zip(starts, ends):
        if end - start >= min_length:
            segments.append({
                'start': start,
                'end': end,
                'x_range': (x[start], x[end-1]),
                'length': end - start,
                'slope': np.polyfit(x[start:end], y[start:end], 1)[0]
            })
    
    # Обчислити загальний нахил
    if segments:
        slopes = np.array([s['slope'] for s in segments])
        lengths = np.array([s['length'] for s in segments])
        
        weighted_avg_slope = np.sum(slopes * lengths) / np.sum(lengths)
        median_slope = np.median(slopes)
    else:
        weighted_avg_slope = np.nan
        median_slope = np.nan
    
    if verbose:
        ax: Axes
        fig, ax = plt.subplots(1, figsize=(12, 6))
        fig.suptitle("Find linear segments DEBUG")
        
        def close_on_escape(event):
            if event.key == 'escape':
                plt.close(event.canvas.figure)
        fig.canvas.mpl_connect('key_press_event', close_on_escape)

        # Знайти сегмент максимальної довжини
        max_seg = max(segments, key=lambda s: s['length']) if segments else None

        # 1. Вхідна крива + знайдені сегменти
        ax.plot(x, y, alpha=0.4, color='gray', linestyle="-", linewidth=1, label='Input curve')
        
        print("=== Linear Segments ===")
        for i, seg in enumerate(segments):
            is_longest = (seg == max_seg)
            color = 'red' if is_longest else 'green'
            label = f"Longest segment (L={seg['length']})" if is_longest else ('Linear segment' if i == 0 else '')
            
            ax.axvspan(x[seg['start']], x[seg['end']-1], alpha=0.3, color=color, label=label if label else None)
            ax.plot(x[seg['start']:seg['end']], y[seg['start']:seg['end']], color=color, linewidth=2, alpha=0.7)
            
            # Друк інформації про сегмент
            marker = " <- LONGEST" if is_longest else ""
            print(f"Seg #{i}: start={seg['start']}, end={seg['end']}, length={seg['length']}, "
                  f"x_range=[{seg['x_range'][0]:.2f}, {seg['x_range'][1]:.2f}], "
                  f"slope={seg['slope']:.6f}{marker}")
        
        print(f"\nTotal segments found: {len(segments)}")
        print(f"Weighted avg slope: {weighted_avg_slope:.6f}")
        print(f"Median slope: {median_slope:.6f}")
        
        ax.set_ylabel('Y')
        ax.set_xlabel('X')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title(f'Found {len(segments)} segments (min_length={min_length})')

        fig.tight_layout()
        plt.show(block=True)
        
    return segments, is_straight, weighted_avg_slope, median_slope



def fix_ofdm_data(ofdm_subcarriers: np.ndarray[complex], **kwargs):
    """
    Plot single symbol spectrum without classification
    
    Args:
        spectrum: complex array (N,)
    """
    N = len(ofdm_subcarriers)
    verbose = kwargs.get('verbose', False)
    
    if verbose:
        title=kwargs.get("title", "OFDM subcarriers DEBUG")
    Fs = kwargs.get("Fs", None)
    if Fs:
        x = np.fft.fftfreq(N, d=1/Fs)
        x = np.fft.fftshift(x)
        x /= 1e3
        if verbose:
            supxlabel = 'Subcarrier (KHz)'
            title += f" | rbw={1e3*Fs/N:.4f} KHz"
    else:
        # x = np.fft.fftshift(np.fft.fftfreq(N))  # нормалізовані частоти [-0.5, 0.5]
        x = np.arange(N) - N//2
        if verbose:
            supxlabel = 'Subcarrier index'
    
    power = ofdm_subcarriers.real**2 + ofdm_subcarriers.imag**2
    power_db = 10 * np.log10(power + 1e-12)

    thr = kwargs.get("thr", None)
    if thr is None:
        # p_down = np.percentile(power_db, 10)
        # p_up = np.percentile(power_db, 80)
        # thr = (p_up + p_down) / 2
        thr = np.mean(power_db) + 5.0


    mask = (power_db >= thr)
    if verbose:
        ax_power_db: Axes
        fig, (ax_power_db, ax_phase, ax_phase_corr) = plt.subplots(3,figsize=(14, 6))
        fig.supxlabel(supxlabel)
        def close_on_escape(event):
            if event.key == 'escape':
                plt.close(event.canvas.figure)
        fig.canvas.mpl_connect('key_press_event', close_on_escape)

        ax_power_db.plot(
            x[mask], power_db[mask], alpha=0.25, color='blue', 
            linestyle="--", linewidth=0.75,
            marker='o', markersize=1.2 # markers: 'o'=circle, 's'=square, '^'=up, 'v'=down, '*'=star, '+'=plus, 'x'=x, 'D'=diamond, '.'=dot
        )
        ax_power_db.set_ylabel('Power (dB)')
        ax_power_db.grid(True, alpha=0.3)
        ax_power_db.axhline(thr, alpha=0.25, color='red', linestyle="--", linewidth=0.75,)
    if 0:
        # RAW phase БЕЗ фільтру
        fig_raw, ax_raw = plt.subplots(figsize=(14, 4))
        
        pow4 = ofdm_subcarriers[mask]**4
        dph = np.zeros_like(pow4)
        dph[1:] = pow4[1:] * np.conj(pow4[:-1])
        phase = np.angle(dph)
        phase_pow4 = np.angle(pow4)
        ax_raw.plot(
            x[mask], phase_pow4, alpha=0.6, color='orange', 
            linestyle="-", linewidth=0.5,
            marker='o', markersize=0.8, label='pow4 phase'
        )
        ax_raw.plot(
            x[mask], phase, alpha=0.6, color='darkblue', 
            linestyle="-", linewidth=0.5,
            marker='o', markersize=0.8, label='Raw phase'
        )
        ax_raw.set_ylabel('Phase (rad)')
        ax_raw.set_xlabel(supxlabel)
        ax_raw.grid(True, alpha=0.3)
        ax_raw.legend()
        ax_raw.set_title('Raw phase**4 — без фільтру')
        
        def close_on_escape(event):
            if event.key == 'escape':
                plt.close(event.canvas.figure)
        fig_raw.canvas.mpl_connect('key_press_event', close_on_escape)
        plt.tight_layout()
        plt.show(block=True)

    pow4 = ofdm_subcarriers[mask]**4
    phase_uw = np.unwrap(np.angle(pow4))
    phase = np.angle(pow4)
    phase_trend = savgol_filter(phase, window_length=51, polyorder=3)
    phase_uw_trend = savgol_filter(phase_uw, window_length=51, polyorder=3)
    
    CONF_THR = 0.5
    segments, is_straight, weighted_avg_slope, median_slope = find_linear_segments(x[mask], phase_uw_trend, 
                                        min_length=20,  curvature_threshold=0.0025, verbose=False)
    if segments:
        # Total coverage (частка даних покрита сегментами)
        total_length = sum(s['length'] for s in segments)
        total_coverage = total_length / len(x[mask])
        
        # Slope consistency (weighted variance)
        slopes = np.array([s['slope'] for s in segments])
        lengths = np.array([s['length'] for s in segments])
        weighted_variance = np.sum(lengths * (slopes - weighted_avg_slope)**2) / np.sum(lengths)
        slope_consistency = 1 / (1 + weighted_variance)
        
        # Angle agreement (weighted vs median)
        angle_weighted = np.arctan(weighted_avg_slope)
        angle_median = np.arctan(median_slope)
        angle_diff = abs(angle_weighted - angle_median)
        angle_agreement = 1 / (1 + angle_diff)
        
        # Combined confidence
        confidence = total_coverage * slope_consistency * angle_agreement
    else:
        confidence = 0.0
    
    if verbose:
        conf_color = GREEN if confidence > CONF_THR else RED
        print(f"{DBG} Weighted avg slope: {weighted_avg_slope:.6f}  |  Median slope: {median_slope:.6f} | {conf_color}{confidence:.2f}{RESET}")
        print(f"{DBG} SEGMENTS_FOUND: {len(segments)}")
        max_seg = max(segments, key=lambda s: s['length']) if segments else None
        if segments and 0:
            # Знайти найдовший сегмент
            for i, seg in enumerate(segments):
                is_longest = (seg == max_seg)
                marker = " <- LONGEST" if is_longest else ""
                print(f"{DBG}   Seg #{i}: start={seg['start']:4d}, end={seg['end']:4d}, "
                    f"len={seg['length']:3d}, x=[{seg['x_range'][0]:8.2f}, {seg['x_range'][1]:8.2f}], "
                    f"slope={seg['slope']:9.6f}{marker}")
        # ax_phase.plot(
        #     x[mask], phase, alpha=0.25, color='blue', 
        #     linestyle="--", linewidth=0.75,
        #     marker='o', markersize=1.2 # markers: 'o'=circle, 's'=square, '^'=up, 'v'=down, '*'=star, '+'=plus, 'x'=x, 'D'=diamond, '.'=dot
        # )
        ax_phase.plot(
            x[mask], phase_uw, alpha=0.25, color='darkblue', 
            linestyle="--", linewidth=0.75,
            marker='o', markersize=1.2 # markers: 'o'=circle, 's'=square, '^'=up, 'v'=down, '*'=star, '+'=plus, 'x'=x, 'D'=diamond, '.'=dot
        )
        # Відобразити найдовший сегмент на графіку
        if max_seg is not None:
            x_masked = x[mask]
            ax_phase.axvspan(
                x_masked[max_seg['start']], 
                x_masked[max_seg['end']-1], 
                alpha=0.2, color='green', 
                label=f"Longest segment (L={max_seg['length']})"
            )
        # ax_phase.plot(x[mask], phase_trend, color='lime', linewidth=0.75, label='Trend')
        ax_phase.plot(x[mask], phase_uw_trend, color='lime', linewidth=0.85, label='Trend')
        # ax_phase.plot(x[mask], phase_correction, color='magenta', linewidth=2.75, label='Trend')
        ax_phase.set_ylabel('Phase **4')
        ax_phase.grid(True, alpha=0.3)

    if confidence > CONF_THR:
        phase_correction = weighted_avg_slope/4 * x[mask]
        ofdm_subcarriers[mask] *= np.exp(-1j * phase_correction)
    if verbose:
        pow4 = ofdm_subcarriers[mask]**4
        phase = np.unwrap(np.angle(pow4))
        phase_trend = savgol_filter(phase, window_length=51, polyorder=3)
        ax_phase_corr.plot(
            x[mask], phase, alpha=0.25, color='darkblue', 
            linestyle="--", linewidth=0.75,
            marker='o', markersize=1.2 # markers: 'o'=circle, 's'=square, '^'=up, 'v'=down, '*'=star, '+'=plus, 'x'=x, 'D'=diamond, '.'=dot
        )
        ax_phase_corr.plot(x[mask], phase_trend, color='lime', linewidth=0.75, label='Trend')
        ax_phase_corr.set_ylabel('Phase **4 ' + 'corrected' if confidence > 0.7 else "NOT CORECTEDED")
        ax_phase_corr.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show(block=True)