from collections import defaultdict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import csv
from datetime import datetime

from colorizer import colorize, inject_colors_into
# --- color names for IDE/static analysis suppress warnings --------------------
GREEN: str; BRIGHT_GREEN: str; RED: str; BRIGHT_RED: str
YELLOW: str; BRIGHT_YELLOW: str; BLACK: str; BRIGHT_BLACK: str
GRAY: str; BRIGHT_GRAY: str; CYAN: str; BRIGHT_CYAN: str
BLUE: str; BRIGHT_BLUE: str; MAGENTA: str; BRIGHT_MAGENTA: str
WHITE: str; BRIGHT_WHITE: str; RESET: str; WARN: str; ERR: str; INFO: str; DBG: str
inject_colors_into(globals())

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
    Знаходить неперервні ділянки (смуги), де рівень сигналу curve перевищує поріг thr.
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
    print(f"\n{MAGENTA}--- Interval Analysis (Timing Patterns) ---{RESET}")
    
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

    print(f"{MAGENTA}-------------------------------------------{RESET}")
    
    
#  Legasy scaffold
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

import pandas as pd
import numpy as np

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