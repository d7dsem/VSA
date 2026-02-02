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