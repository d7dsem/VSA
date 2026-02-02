from dataclasses import dataclass
from typing import List
import numpy as np
import csv
from datetime import datetime


@dataclass
class Bandwidt:
    center: float
    wdt: float
    f_end: float
    f_start: float
    samp_pos: int = -1 # palce holder


def find_bands(curve: np.ndarray, freq_bins: np.ndarray, thr: float) -> List[Bandwidt]:
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
        
        bands.append(Bandwidt(center=float(center), wdt=float(wdt), f_start=f_start, f_end=f_end))
        
    return bands


def analyze_hops_and_timing(bands: List[Bandwidt], fs: float):
    if len(bands) < 2:
        return "Not enough data for hop analysis."

    # Сортуємо за часом (на всяк випадок)
    bands.sort(key=lambda x: x.samp_pos)

    positions = np.array([b.samp_pos for b in bands])
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

def analyze_and_export_bands(bands: List[Bandwidt], fs: float, base_filename: str = "vsa_report"):
    if not bands:
        print("No bands to analyze.")
        return

    # Формуємо ім'я файлу з таймстемпом
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_name = f"{base_filename}_{timestamp}"
    
    # 1. Експорт у CSV (Сирі дані для аналізу)
    csv_file = f"{report_name}.csv"
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['center_hz', 'width_hz', 'sample_pos', 'time_sec'])
        for b in bands:
            writer.writerow([b.center, b.wdt, b.samp_pos, b.samp_pos / fs])

    # 2. Формування звіту (Text Report)
    txt_file = f"{report_name}.txt"
    
    # Розрахунки
    centers = np.array([b.center for b in bands])
    wdts = np.array([b.wdt for b in bands])
    positions = np.array([b.samp_pos for b in bands])
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


import pandas as pd
import numpy as np

def reanalyze_csv(csv_path, fs):
    # 1. Завантажуємо дані
    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    if df.empty:
        print("File is empty.")
        return

    # 2. Розрахунки (використовуємо векторні операції Pandas/NumPy)
    # Часові дельти між пакетами в мс
    df['delta_ms'] = df['sample_pos'].diff() / fs * 1000
    
    # Стрибки частоти в МГц
    df['freq_hop_mhz'] = df['center_hz'].diff() / 1e6
    
    # 3. Вивід основної статистики
    print("\n" + "="*50)
    print(f"RE-ANALYSIS REPORT for: {csv_path}")
    print("="*50)
    print(f"Total packets:      {len(df)}")
    print(f"Avg Bandwidth:      {df['width_hz'].mean()/1e3:.2f} kHz")
    print(f"Freq range:         {df['center_hz'].min()/1e6:.2f} to {df['center_hz'].max()/1e6:.2f} MHz")
    
    # Шукаємо найбільш імовірний період хопінгу (Mode)
    main_period = df['delta_ms'].round(1).mode()
    if not main_period.empty:
        print(f"Main Hop Period:    {main_period[0]:.2f} ms")
    
    # Шукаємо мінімальний крок сітки
    active_hops = df['freq_hop_mhz'].abs()
    grid_step = active_hops[active_hops > 0.1].min() # ігноруємо дрібний джиттер
    if not np.isnan(grid_step):
        print(f"Min Grid Step:      {grid_step:.3f} MHz")

    # 4. Топ активних каналів
    print("\nTop 5 Channels (MHz):")
    top_ch = df['center_hz'].round(-5).value_counts().head(5) # округлення до 100кГц
    for freq, count in top_ch.items():
        print(f"  {freq/1e6:8.3f} MHz | Hits: {count}")
    
    print("="*50)


# reanalyze_csv("твій_файл.csv", Fs)