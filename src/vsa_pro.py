from pathlib import Path
import sys
import traceback
from typing import Any, Dict, Literal, Optional, Union
import numpy as np
from scipy.signal import get_window

import pyqtgraph as pg
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit,
                             QFrame, QLabel, QComboBox, QSlider, QPushButton, QGraphicsProxyWidget) 
from PyQt6.QtCore import  QTimer, Qt
from PyQt6.QtGui import QShortcut, QKeySequence, QColor


# Local imports
from colorizer import colorize, inject_colors_into
from helpers import find_cfo, sec2str
# --- color names for IDE/static analysis suppress warnings --------------------
GREEN: str; BRIGHT_GREEN: str; RED: str; BRIGHT_RED: str
YELLOW: str; BRIGHT_YELLOW: str; BLACK: str; BRIGHT_BLACK: str
GRAY: str; BRIGHT_GRAY: str; CYAN: str; BRIGHT_CYAN: str
BLUE: str; BRIGHT_BLUE: str; MAGENTA: str; BRIGHT_MAGENTA: str
WHITE: str; BRIGHT_WHITE: str; RESET: str; WARN: str; ERR: str; INFO: str; DBG: str
inject_colors_into(globals())


class ConstellationAnalytic:
    DEFAULT_CONFIG = {
        "color": "#f2f4f9",
        "symbol_size": 1.4,
        "symbol": 'o',
        "alpha": 180,
        # "fixed_range": 2.0,  # Діапазон [-range, +range] для I та Q
    }

    def __init__(self, plot_widget: pg.PlotWidget, config: Optional[Dict[str, Any]] = None, **kwargs):
        self.plot = plot_widget
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        
        self.title = kwargs.get("title", "Constellation IQ")
        # Scatter plot для constellation
        self.scatter = pg.ScatterPlotItem(
            size=self.config["symbol_size"],
            pen=pg.mkPen(None),
            brush=pg.mkBrush(self.config["color"], alpha=self.config["alpha"]),
            symbol=self.config["symbol"]
        )
        self.plot.addItem(self.scatter)
        
        # Aspect ratio 1:1
        self.plot.getViewBox().setAspectLocked(lock=True, ratio=1.0)
        self.plot.setLabel('left', 'Q')
        self.plot.setLabel('bottom', 'I')
        self.plot.setTitle(self.title)
        
        # Фіксований діапазон центрований на (0,0)
        r = self.config.get("fixed_range", None)
        if r:
            self.plot.setXRange(-r, r, padding=0)
            self.plot.setYRange(-r, r, padding=0)
        self.r = r

    def update(self, iq_data: np.ndarray):
        """Оновити constellation з комплексного масиву"""
        if len(iq_data) == 0:
            return
        
        i_vals = iq_data.real
        q_vals = iq_data.imag
        
        self.scatter.setData(i_vals, q_vals)
        if self.r:
            self.plot.setXRange(-self.r, self.r, padding=0)
            self.plot.setYRange(-self.r, self.r, padding=0)
        else:
            max_i = np.max(np.abs(i_vals))
            max_q = np.max(np.abs(q_vals))
            max_amp = max(max_i, max_q)
            r = max_amp * 1.1  # 10% padding
            r = min(r, 1e6)
            self.plot.setXRange(-r, r, padding=0)
            self.plot.setYRange(-r, r, padding=0)


class SpectrumAnalytic:
    # Усі візуальні налаштування тепер тільки тут
    DEFAULT_CONFIG = {
        "color_fft_base": '#ffffff',          #  для ROI
        "color_fft_pow2": "#0044ff",       #  для Fixed
        "max_stems_on_screen": 900,     # Поріг включення стему (кількість бінів на екрані)
        "baseline_db": -100,            # "Підлога" для вертикальних ліній
        "line_width_roi": 1.0,
        "line_width_fixed": 1.5,
        "symbol_size": 5,
        "alpha_fixed": 200              # Прозорість для кращого накладання
    }

    def __init__(self, plot_widget: pg.PlotWidget, config: Optional[Dict[str, Any]] = None):
        self.plot = plot_widget
        # Об'єднуємо дефолтний конфіг із тим, що передав користувач
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        
        # Ініціалізація кольорів
        self.color_roi = pg.mkColor(self.config["color_fft_base"])
        self.color_fixed = pg.mkColor(self.config["color_fft_pow2"])
        self.color_fixed.setAlpha(self.config["alpha_fixed"])

        # Створення об'єктів кривих
        self.curve = self.plot.plot(name="ROI", pen=pg.mkPen(self.color_roi, width=self.config["line_width_roi"]))
        self.curve_fixed = self.plot.plot(name="Fixed", pen=pg.mkPen(self.color_fixed, width=self.config["line_width_fixed"]))

        # Сховище для останніх розрахованих даних (щоб перемальовувати при зумі без перерахунку FFT)
        self.last_data = {"roi": (None, None), "fixed": (None, None)}
        
        # Прив'язка до події зміни масштабу ViewBox (зум/пан)
        self.plot.getViewBox().sigRangeChanged.connect(self._auto_restyle_on_zoom)

    def _auto_restyle_on_zoom(self):
        """Автоматично перемикає Line/Stem залежно від поточного масштабу"""
        self._refresh_view(self.curve, *self.last_data["roi"], self.color_roi)
        self._refresh_view(self.curve_fixed, *self.last_data["fixed"], self.color_fixed)

    def _refresh_view(self, curve: pg.PlotDataItem, x: np.ndarray, y: np.ndarray, color: QColor):
        if x is None or y is None:
            return
        
        # Визначаємо, скільки точок (бінів) зараз реально в межах видимості
        view_range = self.plot.viewRange()[0] # Діапазон осі X: [xmin, xmax]
        visible_mask = (x >= view_range[0]) & (x <= view_range[1])
        n_visible = np.count_nonzero(visible_mask)

        # Якщо ми "в'їхали" зумом глибоко в спектр
        if 0 < n_visible <= self.config["max_stems_on_screen"]:
            # --- STEM MODE ---
            visible_indices = np.where(visible_mask)[0]
            # Беремо зріз із невеликим запасом по краях
            idx0, idx1 = max(0, visible_indices[0]-1), min(len(x), visible_indices[-1]+2)
            vx, vy = x[idx0:idx1], y[idx0:idx1]
            
            # Створюємо пари точок для вертикальних ліній (x, y) -> (x, baseline)
            st_x = np.repeat(vx, 2)
            st_y = np.full(len(vx) * 2, self.config["baseline_db"])
            st_y[0::2] = vy

            curve.setData(st_x, st_y, connect='pairs', symbol='o', 
                         symbolSize=self.config["symbol_size"], 
                         symbolBrush=pg.mkBrush(color),
                         pen=pg.mkPen(color, width=1))
        else:
            # --- LINE MODE ---
            # Малюємо звичайний неперервний спектр
            is_fixed = (curve == self.curve_fixed)
            width = self.config["line_width_fixed"] if is_fixed else self.config["line_width_roi"]
            curve.setData(x, y, connect='all', symbol=None, pen=pg.mkPen(color, width=width))

    def update(self, iq_slice: np.ndarray, fs: float, **kwargs):
        """Оновлення даних з динамічним розрахунком параметрів у заголовку"""
        N = len(iq_slice)
        if N < 2: return
        
        fft_n = kwargs.get("fft_n")
        spec_win = kwargs.get("spec_windowing", 'rect')
        method = kwargs.get("method", 'average')
        
        # 1. Розрахунок ROI
        w_n = get_window(spec_win, N) if spec_win != 'rect' else np.ones(N)
        # ENBW (Equivalent Noise Bandwidth)
        enbw = N * np.sum(w_n**2) / (np.sum(w_n)**2)

        # RBW з урахуванням вікна
        rbw_khz = (fs / N) * enbw / 1e3
        sp_n: np.ndarray[complex] = np.fft.fftshift(np.fft.fft(iq_slice * w_n) / np.sum(w_n))
        psd = sp_n.real**2 + sp_n.imag**2 
        psd_db = 10 * np.log10(psd + 1e-12)
        freqs_n = np.fft.fftshift(np.fft.fftfreq(N, 1/fs))
        cfo = find_cfo(psd, freqs_n)
        self.last_data["roi"] = (freqs_n, psd_db)
        
        # ФОРМУЄМО ЗАГОЛОВОК (той самий rbw, що загубився)
        title = f"ROI. {sec2str(N/fs)} ({N:_}) | Fs={fs/1e6:.3f} MHz | rbw={rbw_khz:.3f} KHz | CFO={cfo/1e3:.3f} KHz | window: {spec_win}"

        # 2. Розрахунок Fixed Instrument
        if isinstance(fft_n, int) and fft_n > 0:
            self.curve_fixed.show()
            if N < fft_n:
                # Padding
                iq_p = np.pad(iq_slice * w_n, (0, fft_n - N))
                sp_f = np.fft.fftshift(np.fft.fft(iq_p) / np.sum(w_n))
                psd_f = 20 * np.log10(np.abs(sp_f) + 1e-12)
                title += f" | Fixed: {fft_n} (padding)"
            else:
                # Batching
                batch_n = N // fft_n
                data_2d = iq_slice[:batch_n * fft_n].reshape(batch_n, fft_n)
                w_f = get_window(spec_win, fft_n) if spec_win != 'rect' else np.ones(fft_n)
                fft_m = np.fft.fft(data_2d * w_f, axis=1) / np.sum(w_f)
                
                if method == 'average':
                    psd_f = 10 * np.log10(np.mean(np.abs(fft_m)**2, axis=0) + 1e-12)
                elif method == 'max':
                    psd_f = 10 * np.log10(np.max(np.abs(fft_m)**2, axis=0) + 1e-12)
                else:
                    psd_f = 20 * np.log10(np.abs(fft_m[0]) + 1e-12)
                
                psd_f = np.fft.fftshift(psd_f)
                title += f" | Fixed: {fft_n} ({method})"
            
            freqs_f = np.fft.fftshift(np.fft.fftfreq(fft_n, 1/fs))
            self.last_data["fixed"] = (freqs_f, psd_f)
        else:
            self.curve_fixed.hide()

        # Оновлюємо тайтл віджета
        self.plot.setTitle(title)
        
        # Візуалізація (Line/Stem)
        self._auto_restyle_on_zoom()


class VSAProWindow(QWidget):
    CONFIG = {
        'colors': {
            'I': "#0515f5", 
            'Q': "#0515f5", 
            'Pwr': '#00ff00', 
            'dPh': '#ffff00', 
            'FFT': '#00ff00', 
            'Corr': '#ff8800'  
        },
        'roi_brush': (255, 255, 0, 40),
        'roi_hover_brush': (255, 255, 0, 70),
        'time_mark_color': '#00ffff',
        'time_mark_width':.75,
        'line_width': 1.0,
        'y_axis_width': 65,
        'bg_color': '#1a1a1a',
        'panel_bg': '#222',
        "default_rois_dur_sec": 327.68e-6,
        'min_roi_samples': 2,
        
        'marker_threshold': 800, 
        'marker_size': 1.7,
        'marker_symbol': 'o',
        
        'fft_sizes': ['Dynamic', '32', '64', '128', '256', '512', '1024', '2048', '4096', '8192'],
        'default_fft_size': 'Dynamic',
        'layout_columns_ratio': (2, 3),      # (channels, analytics)
        'layout_ana_rows_ratio': (3, 2),     #  ( fft + conts1, const2)
        'layout_constellation_ratio': (1, 1),# (conts1, const2) в нижній панелі
        'corr_mode_default': 'Magnitude', 
        'corr_phase_degrees': True, 
        'corr_normalize': True,
    }

    def __init__(self, iq: np.ndarray, Fs: float, wtitle: str, **kwargs):
        """
        kwargs supported 
            'extra_channels', ['Pwr', 'dPh'],
            'f_path',
        """
        super().__init__()
        
        # КРИТИЧНІ PRECONDITIONS
        assert iq.size > 0, f"[VSA Pro] Empty IQ data (size={iq.size})"
        assert Fs > 0, f"[VSA Pro] Invalid sample rate: Fs={Fs}"
        assert np.iscomplexobj(iq), f"[VSA Pro] IQ must be complex, got dtype={iq.dtype}"
    
        self.normalize_display = kwargs.get('normalize_display', False)  # DEFAULT: False
        if self.normalize_display:
                # Scale IQ to [-1, 1] range
                scale = np.max(np.abs(self.iq))
                self.iq = self.iq / scale
                print(f"[VSA] Normalized by factor {scale:.2e}")
        self.w_title = wtitle
        self.f_path: Path = kwargs.get("f_path", None)
        
        if self.f_path:
            file_id = f"../{self.f_path.parent.name}/{self.f_path.name}"
            self.w_title += file_id

        self.iq = iq
        self.Fs = Fs
        extra = kwargs.get('extra_channels', [])
        self.rows_names = ['I', 'Q'] + extra
        self.data_len = len(iq)
        self.CONFIG["default_rois_dur_sec"] = kwargs.get("default_rois_dur_sec", 327.68e-6)
        

        # Correaltion block
        self.corr_enabled = 'Corr' in self.rows_names
        self.corr_phase_degrees = kwargs.get('corr_phase_degrees', self.CONFIG['corr_phase_degrees'])
        self.corr_normalize = kwargs.get('corr_normalize', self.CONFIG['corr_normalize'])
        self.corr_complex_cache = None  # Для кешування
        # ROI Width Control
        roi_presets_input = kwargs.get('roi_dur_presets', [])
        default_dur = self.CONFIG["default_rois_dur_sec"]
        
        # Merge presets з default (якщо default не в списку - додати)
        self.roi_presets_sec = list(roi_presets_input)
        if default_dur not in self.roi_presets_sec:
            self.roi_presets_sec.insert(0, default_dur)
        
        # Поточне значення ROI width
        self.current_roi_dur_sec = default_dur
    
        self.channels = {}
        self.plots = {}    
        self.curves = {}   
        self.rois = [] 
        
        self._calculate_projections()
        self.init_layout()
        
        # Time marks 
        self.time_marks = []
        for mark_time in kwargs.get('time_marks', []):
            self.add_time_mark(mark_time)
            
        # Модулі аналітики
        self.fft_module = SpectrumAnalytic(self.ana_widgets['fft'])
        self.const_module_1 = ConstellationAnalytic(self.ana_widgets['constellation_1'], title="Constellation 1")
        self.const_module_2 = ConstellationAnalytic(self.ana_widgets['constellation_2'], title="Constellation 2 (c[i] x c[i+1]*) ")

        self._setup_standard_rois()
        self._fill_initial_data()
        self._setup_slider_logic()
        self._setup_marker_logic()
        self._setup_time_mark_updates()
        self._setup_shortcuts() # ОСЬ ТУТ ТЕПЕР ЖИВУТЬ КНОПКИ
        self.view.scene().sigMouseMoved.connect(self._on_mouse_moved)
        
        # ROI Width Control signals
        self.combo_roi_preset.currentIndexChanged.connect(self._on_preset_selected)
        self.lineedit_roi.returnPressed.connect(self._on_manual_input)
        self.lineedit_roi.editingFinished.connect(self._on_manual_input)
        self.combo_roi_unit.currentTextChanged.connect(self._on_unit_changed)
        
        self.setWindowTitle(f"VSA Pro - {self.w_title}")
        self.resize(1400, 900)
        self.setStyleSheet(f"background-color: {self.CONFIG['bg_color']}; color: #eee;")

    def init_layout(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # --- TOP PANEL ---
        self.top_panel = QFrame()
        self.top_panel.setFixedHeight(45)
        self.top_panel.setStyleSheet(f"background: {self.CONFIG['panel_bg']}; border-bottom: 1px solid #444;")
        top_lay = QHBoxLayout(self.top_panel)
        top_lay.addWidget(QLabel("<b>VSA PRO</b>"))
        
        top_lay.addSpacing(20)
        top_lay.addWidget(QLabel("FFT Size:"))
        self.combo_fft_size = QComboBox()
        self.combo_fft_size.addItems(self.CONFIG['fft_sizes'])
        self.combo_fft_size.setCurrentText(self.CONFIG['default_fft_size'])
        self.combo_fft_size.setFixedWidth(80)
        self.combo_fft_size.currentTextChanged.connect(lambda: self._on_roi_changed(self.rois[0]))
        top_lay.addWidget(self.combo_fft_size)
        
        # --- ROI WIDTH CONTROL ---
        top_lay.addSpacing(20)
        top_lay.addWidget(QLabel("ROI Width:"))
        
        # Preset ComboBox
        self.combo_roi_preset = QComboBox()
        self.combo_roi_preset.setFixedWidth(120)
        
        # Unit ComboBox
        self.combo_roi_unit = QComboBox()
        self.combo_roi_unit.addItems(['s', 'ms', 'us'])
        self.combo_roi_unit.setFixedWidth(60)
        
        # Вибрати оптимальну початкову одиницю
        initial_unit = self._auto_select_unit(self.current_roi_dur_sec)
        self.combo_roi_unit.setCurrentText(initial_unit)
        
        # Наповнити preset combobox
        for preset_sec in self.roi_presets_sec:
            value_in_unit = self._sec_to_unit(preset_sec, initial_unit)
            self.combo_roi_preset.addItem(f"{value_in_unit:.3f} {initial_unit}", preset_sec)
        
        # Manual Input LineEdit
        self.lineedit_roi = QLineEdit()
        self.lineedit_roi.setFixedWidth(100)
        initial_value = self._sec_to_unit(self.current_roi_dur_sec, initial_unit)
        self.lineedit_roi.setText(f"{initial_value:.6g}")
        
        top_lay.addWidget(self.combo_roi_preset)
        top_lay.addWidget(self.lineedit_roi)
        top_lay.addWidget(self.combo_roi_unit)
        
        if self.corr_enabled:
            top_lay.addSpacing(20)
            top_lay.addWidget(QLabel("Corr Mode:"))
            self.combo_corr_mode = QComboBox()
            self.combo_corr_mode.addItems(['Magnitude', 'Phase'])
            self.combo_corr_mode.setCurrentText(self.CONFIG['corr_mode_default'])
            self.combo_corr_mode.setFixedWidth(100)
            self.combo_corr_mode.currentTextChanged.connect(self._on_corr_mode_changed)
            top_lay.addWidget(self.combo_corr_mode)
        
        top_lay.addStretch()
        self.btn_reset = QPushButton("Reset Zoom (A)")
        self.btn_reset.clicked.connect(self._reset_zoom)
        top_lay.addWidget(self.btn_reset)
        main_layout.addWidget(self.top_panel)

        self.view = pg.GraphicsLayoutWidget()
        main_layout.addWidget(self.view)
        
        for i, name in enumerate(self.rows_names):
            p = self.view.addPlot(row=i, col=0)
            p.showGrid(x=True, y=True, alpha=0.3)
            p.getAxis('left').setWidth(self.CONFIG['y_axis_width'])
            p.setMouseEnabled(x=True, y=False)
            p.vb.setLimits(minXRange=2)
            # ФІКС: Встановити розумні limits замість дефолтних ±1e+307
            p.vb.setLimits(
                xMin=-100,
                xMax=self.data_len + 100,
                yMin=-1e6,      # Замість -1e+307
                yMax=1e6,       # Замість +1e+307
                minXRange=2
            )
            # ФІКС: Disable autoRange для Y-axis (обходимо PyQtGraph overflow bug)
            p.enableAutoRange(axis='y', enable=False)
            p.setLabel('left', name, color=self.CONFIG['colors'].get(name, '#fff'))
            self.plots[name] = p
            if i > 0: p.setXLink(self.plots['I'])

        # Time marks PlotDataItem
        self.time_mark_items = {}
        for name in self.rows_names:
            pen = pg.mkPen(self.CONFIG['time_mark_color'], width=self.CONFIG['time_mark_width'])
            item = pg.PlotDataItem(pen=pen, connect='pairs')
            self.plots[name].addItem(item, ignoreBounds=True)
            self.time_mark_items[name] = item
            
        self.slider_proxy = QGraphicsProxyWidget()
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider_proxy.setWidget(self.slider)
        slider_layout = self.view.addLayout(row=4, col=0)
        slider_layout.addItem(self.slider_proxy)
        slider_layout.setContentsMargins(self.CONFIG['y_axis_width'], 5, 0, 10)

        # --- АНАЛІТИЧНА КОЛОНКА ---
        self.analytics_panel = self.view.addLayout(row=0, col=1, rowspan=5)
        self.ana_widgets = {}

        # FFT спектр (верхня частина, 2 частини висоти)
        fft_plot = self.analytics_panel.addPlot(row=0, col=0)
        fft_plot.showGrid(x=True, y=True, alpha=0.3)
        fft_plot.vb.setLimits(xMin=-1e9, xMax=1e9, yMin=-200, yMax=100)
        self.ana_widgets['fft'] = fft_plot

        # Нижня панель — під-сітка з 2 колонками для constellation
        self.constellation_panel = self.analytics_panel.addLayout(row=1, col=0)

        # Два constellation віджети (горизонтально поруч)
        const_plot_1 = self.constellation_panel.addPlot(row=0, col=0)
        const_plot_1.showGrid(x=True, y=True, alpha=0.3)
        const_plot_1.vb.setLimits(xMin=-1e6, xMax=1e6, yMin=-1e6, yMax=1e6)
        self.ana_widgets['constellation_1'] = const_plot_1

        const_plot_2 = self.constellation_panel.addPlot(row=0, col=1)
        const_plot_2.showGrid(x=True, y=True, alpha=0.3)
        const_plot_2.vb.setLimits(xMin=-1e6, xMax=1e6, yMin=-1e6, yMax=1e6)
        self.ana_widgets['constellation_2'] = const_plot_2

        # Пропорції колонок у під-сітці (constellation_1 : constellation_2)
        for i, ratio in enumerate(self.CONFIG['layout_constellation_ratio']):
            self.constellation_panel.layout.setColumnStretchFactor(i, ratio)

        # Пропорції головної аналітичної панелі (FFT : constellation_panel)
        for i, ratio in enumerate(self.CONFIG['layout_ana_rows_ratio']):
            self.analytics_panel.layout.setRowStretchFactor(i, ratio)

        # Пропорції колонок (channels : analytics)
        self.view.ci.layout.setColumnStretchFactor(0, self.CONFIG['layout_columns_ratio'][0])
        self.view.ci.layout.setColumnStretchFactor(1, self.CONFIG['layout_columns_ratio'][1])
        
        # --- FOOTER ---
        self.footer = QFrame()
        self.footer.setFixedHeight(30)
        self.footer.setStyleSheet(f"background: {self.CONFIG['panel_bg']}; border-top: 1px solid #444;")
        self.lbl_cursor = QLabel("Cursor: -")
        self.lbl_cursor.setStyleSheet("font-family: 'Courier New', monospace;")
        footer_lay = QHBoxLayout(self.footer)
        self.lbl_roi_info = QLabel("ROI: -")
        self.lbl_roi_info.setStyleSheet("font-family: 'Courier New', monospace;")
        footer_lay.addWidget(QLabel("Ready"))
        footer_lay.addStretch()
        footer_lay.addWidget(self.lbl_cursor) 
        footer_lay.addWidget(self.lbl_roi_info)
        main_layout.addWidget(self.footer)

    def _setup_shortcuts(self):
        """Гарантована робота хоткеїв незалежно від фокуса"""
        QShortcut(QKeySequence(Qt.Key.Key_Home), self, self._go_home)
        QShortcut(QKeySequence(Qt.Key.Key_End), self, self._go_end)
        QShortcut(QKeySequence(Qt.Key.Key_A), self, self._reset_zoom)

    def _go_home(self):
        vb = self.plots['I'].vb
        vr = vb.viewRange()[0]
        w = vr[1] - vr[0]
        self.plots['I'].setXRange(0, w, padding=0)

    def _go_end(self):
        vb = self.plots['I'].vb
        vr = vb.viewRange()[0]
        w = vr[1] - vr[0]
        self.plots['I'].setXRange(self.data_len - w, self.data_len, padding=0)

    def _calculate_projections(self):
        self.channels['I'] = self.iq.real
        self.channels['Q'] = self.iq.imag
        if 'Pwr' in self.rows_names:
            self.channels['Pwr'] = 10 * np.log10(np.abs(self.iq)**2 + 1e-12)
        if 'dPh' in self.rows_names:
            diff = self.iq[1:] * np.conj(self.iq[:-1])
            dph = np.angle(diff)
            # dph = np.unwrap(dph)
            self.channels['dPh'] = np.append(dph, 0)
        if self.corr_enabled:
            self.channels['Corr'] = np.zeros(self.data_len)

    def _setup_standard_rois(self):
        n_points = int(self.CONFIG['default_rois_dur_sec'] * self.Fs)
        w = max(self.CONFIG['min_roi_samples'], n_points)
        initial_region = [0, w]
        for name in self.rows_names:
            roi = pg.LinearRegionItem(
                values=initial_region, 
                brush=pg.mkBrush(*self.CONFIG['roi_brush']),
                hoverBrush=pg.mkBrush(*self.CONFIG['roi_hover_brush']), 
                movable=True
            )
            self.plots[name].addItem(roi)
            self.rois.append(roi)
            roi.sigRegionChanged.connect(self._on_roi_changed)
        if self.rois: self._on_roi_changed(self.rois[0])

    def mouseDoubleClickEvent(self, event):
        pos = event.position()
        for name in self.rows_names:
            p = self.plots[name]
            if p.sceneBoundingRect().contains(pos):
                mouse_x = p.vb.mapSceneToView(pos).x()
                r1, r2 = self.rois[0].getRegion()
                half_w = (r2 - r1) / 2
                self.rois[0].setRegion([mouse_x - half_w, mouse_x + half_w])
                event.accept()
                return
        super().mouseDoubleClickEvent(event)

    def _on_roi_changed(self, source_roi):
        r0, r1 = source_roi.getRegion()
        if (r1 - r0) < self.CONFIG['min_roi_samples']:
            r1 = r0 + self.CONFIG['min_roi_samples']
            source_roi.blockSignals(True)
            source_roi.setRegion([r0, r1])
            source_roi.blockSignals(False)
        
        region = [r0, r1]
        for roi in self.rois:
            if roi is not source_roi:
                roi.blockSignals(True)
                roi.setRegion(region)
                roi.blockSignals(False)
        
        s0 = int(np.ceil(r0))   
        s1 = int(np.floor(r1)) 
        idx0, idx1 = max(0, s0), min(self.data_len - 1, s1)
        roi_slice = self.iq[idx0:idx1+1]
        # === DEBUG BLOCK ===
        if 0:
            print(f"\n[ROI Debug]")
            print(f"  Region: r0={r0:.2f}, r1={r1:.2f}")
            print(f"  Indices: idx0={idx0}, idx1={idx1}")
            print(f"  roi_slice: size={roi_slice.size}, dtype={roi_slice.dtype}")
            print(f"  I channel: min={self.channels['I'].min():.6e}, max={self.channels['I'].max():.6e}, dtype={self.channels['I'].dtype}")
            print(f"  Q channel: min={self.channels['Q'].min():.6e}, max={self.channels['Q'].max():.6e}, dtype={self.channels['Q'].dtype}")
            print(f"  data_len={self.data_len}")
        # === END DEBUG ===
        if self.corr_enabled:
            corr_complex = np.correlate(self.iq, roi_slice, mode='same')
            # Нормування ЗАВЖДИ (незалежно від display mode)
            norm_factor = np.sqrt(np.sum(np.abs(self.iq)**2) * np.sum(np.abs(roi_slice)**2))
            assert norm_factor > 0, f"[Corr] Zero norm_factor: iq_energy={np.sum(np.abs(self.iq)**2)}, roi_energy={np.sum(np.abs(roi_slice)**2)}"
            corr_complex = corr_complex / norm_factor
            assert np.all(np.isfinite(corr_complex)), "[Corr] Non-finite values after normalization"
            self.corr_complex_cache = corr_complex  # Кешуємо для runtime switch
        self._update_corr_display()
        if hasattr(self, 'fft_module'):
            fft_size_str = self.combo_fft_size.currentText()
            fft_param = int(fft_size_str) if fft_size_str.isdigit() else None
            self.fft_module.update(roi_slice, self.Fs, fft_n=fft_param)

        if hasattr(self, 'const_module_1'):
            fft_result = np.fft.fft(roi_slice) 
            fft_result /= np.max(np.abs(fft_result))
            self.const_module_1.update(fft_result)
        if hasattr(self, 'const_module_2'):
            fft_result = np.fft.fft(roi_slice) 
            fft_result /= np.max(np.abs(fft_result))
            fft_result = fft_result[:-1] * np.conj(fft_result[1:])
            fft_result /= np.max(np.abs(fft_result))
            self.const_module_2.update(fft_result)

        t_str_1 =  sec2str(idx0 / self.Fs)
        ln = len(roi_slice)
        t_str2 = sec2str(ln / self.Fs)

        self.lbl_roi_info.setText(f"RoI:  start {t_str_1} ({idx0:_})  |  dur {t_str2} ({ln:_})")

    def _setup_marker_logic(self):
        self.plots['I'].sigXRangeChanged.connect(self._update_markers)

    def _setup_time_mark_updates(self):
        self.plots['I'].sigXRangeChanged.connect(self._redraw_time_marks)
        for name in self.rows_names:
            self.plots[name].sigYRangeChanged.connect(self._redraw_time_marks)

    def _update_markers(self, _, range_x):
        n_visible = range_x[1] - range_x[0]
        symbol = self.CONFIG['marker_symbol'] if n_visible < self.CONFIG['marker_threshold'] else None
        for name in self.rows_names:
            if name in self.curves:
                self.curves[name].setSymbol(symbol)
                if symbol:
                    self.curves[name].setSymbolSize(self.CONFIG['marker_size'])
                    self.curves[name].setSymbolBrush(self.CONFIG['colors'][name])

    def _on_mouse_moved(self, pos):
        """Оновлення позиції курсора у footer"""
        num_chr = 28
        for name in self.rows_names:
            vb = self.plots[name].vb
            if vb.sceneBoundingRect().contains(pos):
                mouse_point = vb.mapSceneToView(pos)
                x = mouse_point.x()
                
                if 0 <= x < self.data_len:
                    samp_num = int(x)
                    time_str = sec2str(x / self.Fs)
                    view_str = f"{time_str} ({samp_num:_})"
                    self.lbl_cursor.setText(f"Cursor: {view_str:>{num_chr}}")
                else:
                    self.lbl_cursor.setText(f"Cursor: {'out of range':^{num_chr}}")
                return
        
        self.lbl_cursor.setText(f"Cursor: {'---':^{num_chr}}")
                
    def _setup_slider_logic(self):
        self.slider.valueChanged.connect(self._on_slider_moved)
        self.plots['I'].sigXRangeChanged.connect(self._update_slider_from_plot)

    def _on_slider_moved(self, val):
        vb = self.plots['I'].vb
        view_range = vb.viewRange()[0]
        view_width = view_range[1] - view_range[0]
        new_start = (val / 1000.0) * (self.data_len - view_width)
        self.plots['I'].setXRange(new_start, new_start + view_width, padding=0)

    def _update_slider_from_plot(self, _, range_x):
        view_width = range_x[1] - range_x[0]
        if self.data_len > view_width:
            pos = range_x[0] / (self.data_len - view_width)
            self.slider.blockSignals(True)
            self.slider.setValue(int(max(0, min(1000, pos * 1000))))
            self.slider.blockSignals(False)

    def _fill_initial_data(self):
        for name in self.rows_names:
            color = self.CONFIG['colors'].get(name, '#fff')
            data = self.channels[name]
            # Малюємо дані
            self.curves[name] = self.plots[name].plot(
                data, 
                pen=pg.mkPen(color, width=self.CONFIG['line_width'])
            )
            # ФІКС: Manual Y-range (обходимо PyQtGraph overflow при autoRange)
            y_min, y_max = np.min(data), np.max(data)
            y_margin = (y_max - y_min) * 0.1  # 10% padding
            self.plots[name].setYRange(y_min - y_margin, y_max + y_margin, padding=0)
            
            # === ДІАГНОСТИКА ===
            if 0:
                print(f"\n[Plot {name}]")
                print(f"  Y-range set: [{y_min - y_margin:.2f}, {y_max + y_margin:.2f}]")
                vb_state = self.plots[name].vb.state
                print(f"  ViewBox state['limits']: {vb_state.get('limits', 'None')}")
                print(f"  AutoRange enabled (X): {self.plots[name].vb.autoRangeEnabled()[0]}")
                print(f"  AutoRange enabled (Y): {self.plots[name].vb.autoRangeEnabled()[1]}")
                print(f"  Current viewRange: {self.plots[name].vb.viewRange()}")
             # ============================================================================
    #  Time marks
    def add_time_mark(self, time_sec: float):
        """Додати мітку по часу (в секундах)"""
        position = int(time_sec * self.Fs)
        if 0 <= position < self.data_len:
            self.time_marks.append(position)
            self._redraw_time_marks()

    def remove_time_mark(self, time_sec: float):
        """Видалити мітку по часу (в секундах)"""
        position = int(time_sec * self.Fs)
        if position in self.time_marks:
            self.time_marks.remove(position)
            self._redraw_time_marks()

    def clear_time_marks(self):
        """Очистити всі мітки"""
        self.time_marks = []
        self._redraw_time_marks()

    def _redraw_time_marks(self):
        """вертикальні лінії на всіх інфоканалах"""
        if not self.time_marks:
            for item in self.time_mark_items.values():
                item.setData([], [])
            return
        
        x_min, x_max = self.plots['I'].viewRange()[0]
        visible = [m for m in self.time_marks if x_min <= m <= x_max]
        
        if not visible:
            for item in self.time_mark_items.values():
                item.setData([], [])
            return
        
        for channel_name, item in self.time_mark_items.items():
            y_min, y_max = self.plots[channel_name].viewRange()[1]
            x_data = np.repeat(visible, 2)
            y_data = np.tile([y_min, y_max], len(visible))
            item.setData(x_data, y_data)
    # ============================================================================
    #  correlation
    def _update_corr_display(self):
        """Оновити відображення Corr каналу згідно вибраного режиму"""
        if not self.corr_enabled or self.corr_complex_cache is None:
            return
        
        mode = self.combo_corr_mode.currentText()
        
        if mode == 'Magnitude':
            data = np.abs(self.corr_complex_cache)
            if self.corr_normalize:
                data = data / np.max(data)
        else:  # Phase
            data = np.angle(self.corr_complex_cache)
            if self.corr_phase_degrees:
                data = np.degrees(data)
        
        # CLAMP екстремальних значень перед передачею в ViewBox
        data = np.clip(data, -1e12, 1e12)  # <--- HARD LIMIT
        
        self.channels['Corr'] = data
        if 'Corr' in self.curves:
            self.curves['Corr'].setData(data)

    def _on_corr_mode_changed(self):
        """Callback при зміні Corr Mode combo"""
        self._update_corr_display()
    # ============================================================================
    
    def _reset_zoom(self):
        self.plots['I'].setXRange(0, self.data_len)
        self.plots['I'].autoRange(padding=0)


    # ============================================================================
    #  ROI Width Control - Helper methods
    # ============================================================================
    
    def _sec_to_unit(self, sec: float, unit: str) -> float:
        """Конвертація секунд у вибрану одиницю"""
        if unit == 's':
            return sec
        elif unit == 'ms':
            return sec * 1e3
        elif unit == 'us':
            return sec * 1e6
        else:
            return sec
    
    def _unit_to_sec(self, value: float, unit: str) -> float:
        """Конвертація з одиниці в секунди"""
        if unit == 's':
            return value
        elif unit == 'ms':
            return value * 1e-3
        elif unit == 'us':
            return value * 1e-6
        else:
            return value
    
    def _auto_select_unit(self, sec: float) -> str:
        """Вибрати оптимальну одиницю для відображення"""
        if sec < 1e-3:
            return 'us'
        elif sec < 1.0:
            return 'ms'
        else:
            return 's'
    
    def _update_preset_combobox_units(self):
        """Оновити відображення preset при зміні unit"""
        if not hasattr(self, 'combo_roi_preset'):
            return
        
        current_idx = self.combo_roi_preset.currentIndex()
        unit = self.combo_roi_unit.currentText()
        
        for i in range(self.combo_roi_preset.count()):
            preset_sec = self.combo_roi_preset.itemData(i)
            value_in_unit = self._sec_to_unit(preset_sec, unit)
            self.combo_roi_preset.setItemText(i, f"{value_in_unit:.3f} {unit}")
        
        if current_idx >= 0:
            self.combo_roi_preset.setCurrentIndex(current_idx)
            
    def _apply_roi_duration(self, dur_sec: float):
        """Застосувати фіксовану ширину до ROI, зберігаючи center"""
        n_samples = int(dur_sec * self.Fs)
        
        # Отримати поточний center ROI
        r0, r1 = self.rois[0].getRegion()
        center = (r0 + r1) / 2
        
        # Розрахувати нові bounds
        half_width = n_samples / 2
        new_r0 = center - half_width
        new_r1 = center + half_width
        
        # Clamp до меж датасету
        if new_r0 < 0:
            new_r0 = 0
            new_r1 = n_samples
        if new_r1 > self.data_len:
            new_r1 = self.data_len
            new_r0 = self.data_len - n_samples
        
        # Застосувати
        self.rois[0].setRegion([new_r0, new_r1])
    
    def _on_preset_selected(self):
        """Callback при виборі preset з ComboBox"""
        idx = self.combo_roi_preset.currentIndex()
        if idx < 0:
            return
        
        preset_sec = self.combo_roi_preset.itemData(idx)
        self.current_roi_dur_sec = preset_sec
        
        # Оновити LineEdit
        unit = self.combo_roi_unit.currentText()
        value = self._sec_to_unit(preset_sec, unit)
        self.lineedit_roi.setText(f"{value:.6g}")
        
        # Застосувати до ROI
        self._apply_roi_duration(preset_sec)
    
    def _on_manual_input(self):
        """Callback при ручному вводі (Enter або focusOut)"""
        self._validate_and_apply_manual_input()
    
    def _on_unit_changed(self):
        """Callback при зміні одиниці"""
        # Конвертувати поточне значення в LineEdit
        try:
            old_unit_idx = self.combo_roi_unit.currentIndex()
            # Отримати старе значення перед зміною
            current_text = self.lineedit_roi.text()
            if current_text:
                old_value = float(current_text)
                # Знайти яка була стара одиниця (fallback через current_roi_dur_sec)
                unit = self.combo_roi_unit.currentText()
                new_value = self._sec_to_unit(self.current_roi_dur_sec, unit)
                self.lineedit_roi.setText(f"{new_value:.6g}")
        except:
            pass
        
        # Оновити preset combobox
        self._update_preset_combobox_units()
    
    def _update_lineedit_from_current(self):
        """Синхронізувати LineEdit з current_roi_dur_sec"""
        unit = self.combo_roi_unit.currentText()
        value = self._sec_to_unit(self.current_roi_dur_sec, unit)
        self.lineedit_roi.setText(f"{value:.6g}")
    
    def _validate_and_apply_manual_input(self):
        """Валідація та застосування ручного вводу"""
        try:
            value = float(self.lineedit_roi.text())
            if value <= 0:
                raise ValueError("Value must be positive")
            
            unit = self.combo_roi_unit.currentText()
            dur_sec = self._unit_to_sec(value, unit)
            
            # Перевірка меж
            n_samples = int(dur_sec * self.Fs)
            if n_samples < self.CONFIG['min_roi_samples']:
                raise ValueError("Too small")
            if n_samples > self.data_len:
                raise ValueError("Too large")
            
            # Ok - застосувати
            self.current_roi_dur_sec = dur_sec
            self._apply_roi_duration(dur_sec)
            
        except:
            # Відкат до попереднього значення
            self._update_lineedit_from_current()
            self._flash_error_border()
    
    def _flash_error_border(self):
        """Візуальний feedback помилки"""
        self.lineedit_roi.setStyleSheet("border: 1px solid red;")
        QTimer.singleShot(500, lambda: self.lineedit_roi.setStyleSheet(""))


def deploy_vsa_pro(iq: np.ndarray, Fs: float, wtitle:str="", **kwargs):
    ln = iq.size
    app = QApplication.instance() or QApplication(sys.argv)
    try:
        gui = VSAProWindow(iq, Fs, wtitle, **kwargs)
        gui.show()
        app.exec()
        return gui
    except Exception:
        print(f"{ERR} VSA Pro Crash:")
        traceback.print_exc()
        sys.exit(0)