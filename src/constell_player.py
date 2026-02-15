import sys
import traceback
from typing import List, Literal
import numpy as np
import scipy

import pyqtgraph as pg
from pyqtgraph import ScatterPlotItem, LinearRegionItem
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit,
                             QFrame, QLabel, QComboBox, QSlider, QPushButton, QGraphicsProxyWidget) 
from PyQt6.QtCore import  QTimer, Qt
from PyQt6.QtGui import QShortcut, QKeySequence, QColor

from colorizer import colorize, inject_colors_into
# --- color names for IDE/static analysis suppress warnings --------------------
GREEN: str; BRIGHT_GREEN: str; RED: str; BRIGHT_RED: str
YELLOW: str; BRIGHT_YELLOW: str; BLACK: str; BRIGHT_BLACK: str
GRAY: str; BRIGHT_GRAY: str; CYAN: str; BRIGHT_CYAN: str
BLUE: str; BRIGHT_BLUE: str; MAGENTA: str; BRIGHT_MAGENTA: str
WHITE: str; BRIGHT_WHITE: str; RESET: str; WARN: str; ERR: str; INFO: str; DBG: str
inject_colors_into(globals())

# ============================================================================
# === UTILITIES
# ============================================================================

# ============================================================================
# === UTILITIES
# ============================================================================

def _roi_to_indices(center: float, delta: float, n_bins: int) -> tuple[int, int]:
    """
    Перетворити нормалізовані ROI параметри на індекси FFT бінів.
    
    center: float, центр ROI у нормалізованих частотах [-0.5, +0.5]
    delta: float, половина ширини ROI у нормалізованих частотах
    n_bins: int, загальна кількість FFT бінів
    
    Return: (idx_start, idx_end) — індекси для slicing
    """
    freq_start = center - delta
    freq_end = center + delta
    
    idx_center = n_bins // 2
    idx_start = int(idx_center + freq_start * n_bins)
    idx_end = int(idx_center + freq_end * n_bins)
    
    idx_start = max(0, idx_start)
    idx_end = min(n_bins, idx_end)
    
    return idx_start, idx_end

def _indices_to_roi(idx_start: int, idx_end: int, n_bins: int) -> tuple[float, float]:
    """
    Зворотня операція до _roi_to_indices.
    
    idx_start, idx_end: індекси FFT бінів [0, n_bins]
    n_bins: int, загальна кількість FFT бінів
    
    Return: (center, delta) — нормалізовані параметри ROI
    """
    idx_center = n_bins // 2
    center = ((idx_start + idx_end) / 2 - idx_center) / n_bins
    delta = (idx_end - idx_start) / 2 / n_bins
    return center, delta

def _idx_to_shifted(idx: int, offset: int) -> int:
    """Конвертувати індекс [0, n_bins] в зсунуту координату з центром у 0"""
    return idx - offset

def _shifted_to_idx(shifted: int, offset: int) -> int:
    """Конвертувати зсунуту координату назад в індекс [0, n_bins]"""
    return shifted + offset

# n_bins = 512
# idx_start, idx_end = _roi_to_indices(center=0, delta=0.1, n_bins=n_bins)
# print(f"ROI: center=0, delta=0.1 => індекси [{idx_start}, {idx_end}]")

def _test_brush_alpha():
    # === ТЕСТ 1: Лінія (працює) ===
    print("=== ЛІНІЯ ===")
    color = QColor(255, 0, 0)
    color.setAlpha(50)
    pen = pg.mkPen(color, width=2)
    print(f"Color to pen: alpha={color.alpha()}")
    print(f"Pen result:   alpha={pen.color().alpha()}")

    # === ТЕСТ 2: Символ (НЕ ПРАЦЮЄ) ===
    print("\n=== СИМВОЛ ===")
    color = QColor(255, 0, 0)
    color.setAlpha(50)
    brush = pg.mkBrush(color)
    print(f"Color to brush: alpha={color.alpha()}")
    print(f"Brush result:   alpha={brush.color().alpha()}")  # ← завжди 255!

    # === ПРИЧИНА ===
    print("\n=== ПРИЧИНА ===")
    print("pyqtgraph символи (symbol='o') не поважають alpha у brush.")
    print("Вони рисуються як bitmap, а не як вектор.")
    print("Alpha для brush ігнорується на етапі растеризації.")

def _get_slice_size(s0_slices: List[np.ndarray], s1_slices: List[np.ndarray]) -> int:
    if not s0_slices or not s1_slices:
        raise ValueError("Empty slice lists")
    s0_size = len(s0_slices[0])
    s1_size = len(s1_slices[0])
    if s0_size != s1_size:
        raise ValueError(f"s0 and s1 size mismatch: {s0_size} vs {s1_size}")
    for i, s0 in enumerate(s0_slices):
        if len(s0) != s0_size:
            raise ValueError(f"s0_slices[{i}] size mismatch: {len(s0)} vs {s0_size}")
    for i, s1 in enumerate(s1_slices):
        if len(s1) != s1_size:
            raise ValueError(f"s1_slices[{i}] size mismatch: {len(s1)} vs {s1_size}")
    return s0_size

def _get_window(window_mode: Literal["Rect", "Hann", "Hamming", "Blackman"], fft_n: int) -> np.ndarray:
    windows = {
        "Rect": np.ones,
        "Hann": np.hanning,
        "Hamming": np.hamming,
        "Blackman": np.blackman,
    }
    if window_mode not in windows:
        raise ValueError(f"Unknown window: {window_mode}")
    return windows[window_mode](fft_n)

def _alpha_linear(t: float, depth: int, alpha_min: float) -> float:
    """t: 0 (найсвіжіший) до depth-1 (найстарший)"""
    if depth == 1:
        return 1.0
    return max(alpha_min, 1.0 - (t / (depth - 1)) * (1.0 - alpha_min))

def _alpha_exponential(t: float, depth: int, alpha_min: float) -> float:
    """t: 0 (найсвіжіший) до depth-1 (найстарший)"""
    if depth == 1:
        return 1.0
    # e^(-k*t), де k обирається щоб в кінці було alpha_min
    k = -np.log(alpha_min) / (depth - 1)
    return max(alpha_min, np.exp(-k * t))


def _interpolate_color(t: float, depth: int, color_start: str, color_end: str) -> QColor:
    """t: 0 (свіжий) до depth-1 (старий)"""
    ratio = t / (depth - 1) if depth > 1 else 0
    
    c_start = QColor(color_start)
    c_end = QColor(color_end)
    
    r = int(c_start.red() + (c_end.red() - c_start.red()) * ratio)
    g = int(c_start.green() + (c_end.green() - c_start.green()) * ratio)
    b = int(c_start.blue() + (c_end.blue() - c_start.blue()) * ratio)
    
    return QColor(r, g, b)
# ============================================================================
# === CHANNEL PLOT (один канал = 3 графіки) з FADE ЕФЕКТОМ
# ============================================================================

class ChannelPlot(QWidget):
    CONFIG = {
        'background': '#1a1a1a',
        'const_points': {'color': 'blue', 'size': 1.1},
        'power_points': {'color': 'blue', 'size': 1.1},
        'power_stems': {'color': 'lightblue', 'width': 0.75},
        'layout_rows_ratio': (2, 3, 3),
        'history_depth': 5,
        'alpha_min': 0.1,
        'decay_curve': 'linear',  # 'linear' або 'exponential'
        'color_start': '#FFFFFF',    # Білий для свіжих
        'color_end': "#A10505",      # Червоний для старих
        'dot_sz': 1.1
    }

    def __init__(self, **kwargs):
        super().__init__()
        self.setWindowTitle(kwargs.get('wtitle', 'Channel Plot'))
        # if provided then decides show_const span as area arroun spectsral 0 (DC)  
        
        # === ROI ПАРАМЕТРИ ===
        self.const_show_center = kwargs.get('const_show_center', 0.0)
        self.const_show_delta = kwargs.get('const_show_delta', 0.5)  / 2 # 1.0 весь спектр
        
        # === ROI СТАН ===
        self.roi_start_idx = None
        self.roi_end_idx = None
        self.n_bins = None
        self.offset = None
        
        # === CONFIG OVERRIDE ===
        config = self.CONFIG.copy()
        config.update(kwargs)
        self.history_depth = config['history_depth']
        self.alpha_min = config['alpha_min']
        self.decay_curve = config['decay_curve']
        self.color_start = config['color_start']
        self.color_end = config['color_end']
        # === DECAY FUNCTION ===
        if self.decay_curve == 'exponential':
            self.alpha_func = _alpha_exponential
        else:
            self.alpha_func = _alpha_linear
        
        # === HISTORY ===
        self.history_powers = []
        self.history_raw = []
        self.history_diff = []
        
        # === ROW 1: Power Spectrum ===
        self.power_plot = pg.PlotWidget()
        self.power_plot.setLabel('left', 'Power (dB)')
        self.power_plot.setLabel('bottom', 'Frequency bin (DC centered)')
        self.power_plot.setBackground(config['background'])
        
        # === ROI REGION (буде додано потім на plot) ===
        self.roi_region = None  # ініціалізуємо як None, створимо при першому update_frame


        # === ДІАПАЗОНИ КОНСТЕЛЯЦІЇ ===
        self.const_range_raw = kwargs.get('const_range_raw', None)
        self.const_range_diff = kwargs.get('const_range_diff', None)
        
        # === ROW 2: Raw Constellation ===
        self.constell_raw_plot = pg.PlotWidget()
        self.constell_raw_plot.setLabel('left', 'Imaginary')
        self.constell_raw_plot.setLabel('bottom', 'Real')
        self.constell_raw_plot.setAspectLocked(True)
        self.constell_raw_plot.setBackground(config['background'])
        
        # === ROW 3: Diff Constellation ===
        self.constell_diff_plot = pg.PlotWidget()
        self.constell_diff_plot.setLabel('left', 'Imaginary')
        self.constell_diff_plot.setLabel('bottom', 'Real')
        self.constell_diff_plot.setAspectLocked(True)
        self.constell_diff_plot.setBackground(config['background'])
        
        # === LAYOUT ===
        layout = QVBoxLayout()
        r = config['layout_rows_ratio']
        layout.addWidget(self.power_plot, stretch=r[0])
        layout.addWidget(self.constell_raw_plot, stretch=r[1])
        layout.addWidget(self.constell_diff_plot, stretch=r[2])
        self.setLayout(layout)

    def update_frame(self, powers: np.ndarray[float], 
                     ofdm_raw: np.ndarray[complex], 
                     ofdm_diff: np.ndarray[complex]):
        """Оновити всі три графіки для одного кадру, зберегти в історію"""
        # === ЗБЕРЕЖЕННЯ В ІСТОРІЮ ===
        self.history_powers.append(powers.copy())
        self.history_raw.append(ofdm_raw.copy())
        self.history_diff.append(ofdm_diff.copy())
        
        # === ОБРІЗАННЯ ІСТОРІЇ ===
        if len(self.history_powers) > self.history_depth:
            self.history_powers.pop(0)
            self.history_raw.pop(0)
            self.history_diff.pop(0)
        
        # === ОБЧИСЛЕННЯ ROI ІНДЕКСІВ (перший раз) ===
        if self.roi_start_idx is None:
            self.n_bins = len(self.history_powers[0])      # ← кешувати
            self.offset = self.n_bins // 2                  # ← кешувати
            self.roi_start_idx, self.roi_end_idx = _roi_to_indices(
                self.const_show_center, self.const_show_delta, self.n_bins
            )
            self._setup_roi_region()
            print(f"{DBG} ROI: center={self.const_show_center}, delta={self.const_show_delta}")
            print(f"{DBG}      indices: [{self.roi_start_idx}, {self.roi_end_idx}]")

        # === РЕНДЕР З FADE ===
        self._plot_stem_fade(self.power_plot)
        self._plot_constellation_fade(self.constell_raw_plot, self.history_raw, self.const_range_raw)
        self._plot_constellation_fade(self.constell_diff_plot, self.history_diff, self.const_range_diff)

    def _plot_stem_fade(self, plot_widget: pg.PlotWidget):
        """Спектр: вертикальні лінії + точки з fade"""
        plot_widget.clear()
        x = np.arange(self.n_bins) - self.offset
        # === МАЛЮВАННЯ ROI ОБЛАСТІ (вже створена в __init__) ===
        # self.roi_region буде доданий автоматично, його не потрібно пересоздавати
        if self.roi_region is not None:
            plot_widget.addItem(self.roi_region)
        # Рисуємо від найстарішого до найсвіжішого
        for t, powers in enumerate(self.history_powers):
            t_idx = len(self.history_powers) - 1 - t
            alpha = self.alpha_func(t_idx, self.history_depth, self.alpha_min)
            color = _interpolate_color(t_idx, self.history_depth, 
                                self.color_start, 
                                self.color_end)
            color.setAlpha(int(alpha * 255))
            pen = pg.mkPen(color, width=0.75)
            brush = pg.mkBrush(color)
            plot_widget.plot(x, powers, pen=pen)
            plot_widget.plot(x, powers, pen=None, symbol='o',
                           symbolBrush=brush, symbolSize=1.1)

    def _plot_constellation_fade(self, plot_widget, history, range_val=None):
        plot_widget.clear()
        
        # === ФІКСОВАНИЙ ЦЕНТР У 0,0 + СИМЕТРИЧНИЙ ДІАПАЗОН ===
        if range_val is not None:
            plot_widget.setRange(
                xRange=range_val,
                yRange=range_val,
                padding=0
            )
        
        # === ФІЛЬТРАЦІЯ (НЕ впливає на масштаб) ===
        for t, data in enumerate(history):
            if self.roi_start_idx is not None:
                data_filtered = data[self.roi_start_idx:self.roi_end_idx]
            else:
                data_filtered = data
            
            if len(data_filtered) == 0:
                continue
            
            t_idx = len(history) - 1 - t
            alpha = self.alpha_func(t_idx, self.history_depth, self.alpha_min)
            color = _interpolate_color(t_idx, self.history_depth,
                                    self.color_start, self.color_end)
            color.setAlpha(int(alpha * 255))
            brush = pg.mkBrush(color)
            
            # ← Рисуємо FILTERED дані, але в ФІКСОВАНИХ координатах
            scatter = ScatterPlotItem(
                x=data_filtered.real, 
                y=data_filtered.imag,
                brush=brush,
                pen=None,
                size=self.CONFIG['dot_sz'],
                symbol='o'
            )
            plot_widget.addItem(scatter)

    def _setup_roi_region(self):
        """Створити LinearRegionItem та підключити обробник сигналу"""
        if self.roi_region is not None:
            return
        
        # === ВИКОРИСТАТИ HELPER ===
        roi_start_shifted = _idx_to_shifted(self.roi_start_idx, self.offset)
        roi_end_shifted = _idx_to_shifted(self.roi_end_idx, self.offset)
        
        self.roi_region = LinearRegionItem(
            values=(roi_start_shifted, roi_end_shifted),
            orientation=LinearRegionItem.Vertical,
            brush=pg.mkBrush(255, 100, 100, 50),
            pen=pg.mkPen(color='red', width=2)
        )
        
        self.roi_region.sigRegionChanged.connect(self._on_roi_changed)
        self.power_plot.addItem(self.roi_region)
        
        print(f"{DBG} ROI region created: [{roi_start_shifted}, {roi_end_shifted}] (shifted)")

    def _on_roi_changed(self):
        """Обробник: ROI змінився під час drag"""
        # === ОТРИМАТИ ЗСУНУТІ КООРДИНАТИ З РЕГІОНУ ===
        start_shifted, end_shifted = self.roi_region.getRegion()
        
        # === ПЕРЕВЕСТИ НАЗАД В [0, n_bins] для індексації масивів ===
        n_bins = len(self.history_powers[0]) if self.history_powers else 512
        offset = n_bins // 2
        
        self.roi_start_idx = int(start_shifted + offset)
        self.roi_end_idx = int(end_shifted + offset)
        
        # Перерахувати center та delta
        idx_center = n_bins // 2
        new_center = (self.roi_start_idx + self.roi_end_idx) / 2 - idx_center
        new_center = new_center / n_bins
        new_delta = (self.roi_end_idx - self.roi_start_idx) / 2 / n_bins
        
        self.const_show_center = new_center
        self.const_show_delta = new_delta
        
        # Перерендер констеляцій з новим ROI
        self._plot_constellation_fade(self.constell_raw_plot, self.history_raw, self.const_range_raw)
        self._plot_constellation_fade(self.constell_diff_plot, self.history_diff, self.const_range_diff)
            
# ============================================================================
# === CONSTELLATION PLAYER (дві колонки, програвач)
# ============================================================================

class ConstellPlayer(QWidget):
    CONFIG = {
        'colors': {"s0": "blue", "s1": "red"},
        'layout_columns_ratio': (1, 1),
        'history_depth': 5,
        'alpha_min': 0.1,
        'decay_curve': 'linear',
    }

    def __init__(self, 
                 fft_s0: List[np.ndarray], fft_s1: List[np.ndarray], powers,
                 render_fps: int, wtitle: str, **kwargs):
        super().__init__()
        title_suffix = f" fps {render_fps} | hist depth {kwargs['history_depth']} |"
        self.setWindowTitle(wtitle + title_suffix)


        self.fft_s0 = fft_s0
        self.fft_s1 = fft_s1
        
        self.powers = powers
        
        self.ofdm_demod_raw = np.array([self.fft_s0, self.fft_s1])
        
        # diff має на 1 менше — паддуємо нулем в кінець
        diff_s0 = np.vstack([self.fft_s0[:-1] * np.conj(self.fft_s0[1:]),
                             np.zeros((1, self.fft_s0.shape[1]), dtype=complex)])
        diff_s1 = np.vstack([self.fft_s1[:-1] * np.conj(self.fft_s1[1:]),
                             np.zeros((1, self.fft_s1.shape[1]), dtype=complex)])
        self.ofdm_demod_diff = np.array([diff_s0, diff_s1])
        
        # === ОБЧИСЛИТИ ДІАПАЗОНИ КОНСТЕЛЯЦІЇ (один раз) ===
        all_raw = np.concatenate([self.fft_s0.real, self.fft_s0.imag,
                                  self.fft_s1.real, self.fft_s1.imag])
        all_diff = np.concatenate([self.ofdm_demod_diff[0].real, self.ofdm_demod_diff[0].imag,
                                   self.ofdm_demod_diff[1].real, self.ofdm_demod_diff[1].imag])
        
        max_val_raw = float(np.max(np.abs(all_raw)) * 1.2)
        max_val_diff = float(np.max(np.abs(all_diff)) * 1.2)
        
        # ← Один раз перевіримо
        if np.isfinite(max_val_raw) and max_val_raw > 0:
            const_range_raw = (-max_val_raw, max_val_raw)
        else:
            const_range_raw = (-100.0, 100.0)
        
        if np.isfinite(max_val_diff) and max_val_diff > 0:
            const_range_diff = (-max_val_diff, max_val_diff)
        else:
            const_range_diff = (-100.0, 100.0)
        # r_max_raw = np.max(np.abs(self.fft_s0.real))
        # i_max_raw = np.max(np.abs(self.fft_s0.imag))
        # max_val_raw = max(r_max_raw, i_max_raw) * 1.2
        # const_range_raw = (-max_val_raw, max_val_raw)
        
        # r_max_diff = np.max(np.abs(self.ofdm_demod_diff[0].real))
        # i_max_diff = np.max(np.abs(self.ofdm_demod_diff[0].imag))
        # max_val_diff = max(r_max_diff, i_max_diff) * 1.2
        # const_range_diff = (-max_val_diff, max_val_diff)
        # === PLAYBACK STATE ===
        self.render_fps = render_fps
        self.current_frame = 0
        self.is_playing = False
        self.n_frames = len(self.fft_s0)
        self.loop = kwargs.get("loop", False)
        
        # === CONFIG ===
        config = self.CONFIG.copy()
        config.update(kwargs)
        
        # === UI: CHANNELS ===
        self.channel_s0 = ChannelPlot(
            wtitle="S0",
            const_range_raw=const_range_raw,
            const_range_diff=const_range_diff,
            history_depth=config['history_depth'],
            alpha_min=config['alpha_min'],
            decay_curve=config['decay_curve'],
            color_start=config.get('color_start', '#FFFFFF'),
            color_end=config.get('color_end', '#FF0000'),
            const_show_center = kwargs.get('const_show_center', 0.0),
            const_show_delta = kwargs.get('const_show_delta', 0.39)  / 2, # 1.0 весь спектр
        )
        self.channel_s1 = ChannelPlot(
            wtitle="S1",
            history_depth=config['history_depth'],
            const_range_raw=const_range_raw,
            const_range_diff=const_range_diff,
            alpha_min=config['alpha_min'],
            decay_curve=config['decay_curve'],
            color_start=config.get('color_start', '#FFFFFF'),
            color_end=config.get('color_end', '#FF0000'),
            const_show_center = kwargs.get('const_show_center', 0.0),
            const_show_delta = kwargs.get('const_show_delta', 0.5)  / 2, # 1.0 весь спектр
        )
        
        # === UI: CONTROL PANEL ===
        self.control_panel = self._create_control_panel()
        
        # === LAYOUT ===
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.control_panel, stretch=0)
        
        channels_layout = QHBoxLayout()
        channels_layout.addWidget(self.channel_s0, stretch=1)
        channels_layout.addWidget(self.channel_s1, stretch=1)
        main_layout.addLayout(channels_layout, stretch=1)
        
        self.setLayout(main_layout)
        
        # === TIMER ===
        self.timer = QTimer()
        self.timer.timeout.connect(self._on_timer)
        self.timer.setInterval(int(1000 / render_fps))
        
        # === INITIAL DISPLAY ===
        self._update_display()

    def _create_control_panel(self) -> QWidget:
        """Панель: Play/Pause, Slider, Label"""
        panel = QWidget()
        layout = QHBoxLayout()
        
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self._toggle_play)
        
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(self.n_frames - 1)
        self.frame_slider.sliderMoved.connect(self._on_slider_moved)
        
        self.frame_label = QLabel(f"0 / {self.n_frames}")
        
        layout.addWidget(self.play_btn)
        layout.addWidget(self.frame_slider, stretch=1)
        layout.addWidget(self.frame_label)
        
        panel.setLayout(layout)
        return panel

    def _toggle_play(self):
        """Play/Pause toggle"""
        self.is_playing = not self.is_playing
        self.play_btn.setText("Pause" if self.is_playing else "Play")
        if self.is_playing:
            self.timer.start()
        else:
            self.timer.stop()

    def _on_timer(self):
        """Таймер: наступний кадр"""
        self.current_frame += 1
        if self.current_frame >= self.n_frames:
            if self.loop:
                self.current_frame = 0
            else:
                self._toggle_play()
                return
        self._update_display()

    def _on_slider_moved(self, value: int):
        """Slider: перейти на кадр"""
        self.current_frame = value
        self._update_display()

    def _update_display(self):
        """Оновити обидва канали для поточного кадру"""
        idx = self.current_frame
        
        self.channel_s0.update_frame(
            self.powers[0, idx],
            self.ofdm_demod_raw[0, idx],
            self.ofdm_demod_diff[0, idx]
        )
        self.channel_s1.update_frame(
            self.powers[1, idx],
            self.ofdm_demod_raw[1, idx],
            self.ofdm_demod_diff[1, idx]
        )
        
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(idx)
        self.frame_slider.blockSignals(False)
        self.frame_label.setText(f"{idx} / {self.n_frames}")

# ============================================================================
# === ENTRY POINT Animation
# ============================================================================

def animate_constells(fft_s0: List[np.ndarray], fft_s1: List[np.ndarray], power, 
                     render_fps: int = 24, wtitle: str = "Constells player", **kwargs):
    """Запустити програвач констеляцій з fade ефектом"""
    app = QApplication.instance() or QApplication(sys.argv)
    try:
        gui = ConstellPlayer(fft_s0, fft_s1, power, render_fps, wtitle, **kwargs)
        gui.show()
        app.exec()
        return gui
    except Exception:
        print(f"{ERR} Constell Player Crash:")
        traceback.print_exc()
        sys.exit(1)