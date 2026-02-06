import sys
import traceback
from typing import Any, Dict, Literal, Optional, Union
import numpy as np
from scipy.signal import get_window

import pyqtgraph as pg
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QFrame, QLabel, QComboBox, QSlider, QPushButton, QGraphicsProxyWidget) 
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QShortcut, QKeySequence, QColor


# Local imports
from colorizer import colorize, inject_colors_into
# --- color names for IDE/static analysis suppress warnings --------------------
GREEN: str; BRIGHT_GREEN: str; RED: str; BRIGHT_RED: str
YELLOW: str; BRIGHT_YELLOW: str; BLACK: str; BRIGHT_BLACK: str
GRAY: str; BRIGHT_GRAY: str; CYAN: str; BRIGHT_CYAN: str
BLUE: str; BRIGHT_BLUE: str; MAGENTA: str; BRIGHT_MAGENTA: str
WHITE: str; BRIGHT_WHITE: str; RESET: str; WARN: str; ERR: str; INFO: str; DBG: str
inject_colors_into(globals())



class SpectrumAnalytic:
    # Усі візуальні налаштування тепер тільки тут
    DEFAULT_CONFIG = {
        "color_roi": '#ffffff',          #  для ROI
        "color_fixed": "#0044ff",       #  для Fixed
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
        self.color_roi = pg.mkColor(self.config["color_roi"])
        self.color_fixed = pg.mkColor(self.config["color_fixed"])
        self.color_fixed.setAlpha(self.config["alpha_fixed"])

        # Створення об'єктів кривих
        self.curve = self.plot.plot(name="ROI", pen=pg.mkPen(self.color_roi, width=self.config["line_width_roi"]))
        self.curve_fixed = self.plot.plot(name="Fixed", pen=pg.mkPen(self.color_fixed, width=self.config["line_width_fixed"]))

        # Сховище для останніх розрахованих даних (щоб перемальовувати при зумі без перерахунку FFT)
        self.last_data = {"roi": (None, None), "fixed": (None, None)}
        
        # Прив'язка до події зміни масштабу ViewBox (зум/пан)
        self.plot.getViewBox().sigRangeChanged.connect(self._auto_resyle_on_zoom)

    def _auto_resyle_on_zoom(self):
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
        sp_n = np.fft.fftshift(np.fft.fft(iq_slice * w_n) / np.sum(w_n))
        psd_n = 20 * np.log10(np.abs(sp_n) + 1e-12)
        freqs_n = np.fft.fftshift(np.fft.fftfreq(N, 1/fs))
        self.last_data["roi"] = (freqs_n, psd_n)
        
        # ФОРМУЄМО ЗАГОЛОВОК (той самий rbw, що загубився)
        rbw_khz = (fs / N) / 1e3
        title = f"ROI rbw={rbw_khz:.3f} KHz (N={N})"

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
        self._auto_resyle_on_zoom()


class VSAProWindow(QWidget):
    CONFIG = {
        'colors': {'I': '#5af', 'Q': '#f5a', 'Pwr': '#0f0', 'dPh': '#ff0', 'FFT': '#0f0'},
        'roi_brush': (255, 255, 0, 40),
        'roi_hover_brush': (255, 255, 0, 70),
        'line_width': 1.0,
        'y_axis_width': 65,
        'bg_color': '#1a1a1a',
        'panel_bg': '#222',
        "defaulyt_rois_dur_sec": 327.7e-6,
        'min_roi_samples': 2,
        'marker_threshold': 800, 
        'marker_size': 7,
        'marker_symbol': 'o',
        'fft_sizes': ['Dynamic', '32', '64', '128', '256', '512', '1024', '2048', '4096', '8192'],
        'default_fft_size': 'Dynamic',
        'layout_columns_ratio': (2, 3),      # (channels, analytics)
        'layout_ana_rows_ratio': (2, 1, 1),  # (ana_0, ana_1, ana_2)
    }

    def __init__(self, iq, Fs, file_id, **kwargs):
        super().__init__()
        self.iq = iq
        self.Fs = Fs
        self.rows_names = ['I', 'Q', 'Pwr', 'dPh']
        self.data_len = len(iq)
        self.channels = {}
        self.plots = {}    
        self.curves = {}   
        self.rois = [] 
        
        self._calculate_projections()
        self.init_layout()
        
        # Модулі аналітики
        self.fft_module = SpectrumAnalytic(self.ana_widgets['ana_0'])

        self._setup_standard_rois()
        self._fill_initial_data()
        self._setup_slider_logic()
        self._setup_marker_logic()
        self._setup_shortcuts() # ОСЬ ТУТ ТЕПЕР ЖИВУТЬ КНОПКИ

        self.setWindowTitle(f"VSA Pro - {file_id}")
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
            self.plots[name] = p
            if i > 0: p.setXLink(self.plots['I'])

        self.slider_proxy = QGraphicsProxyWidget()
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider_proxy.setWidget(self.slider)
        slider_layout = self.view.addLayout(row=4, col=0)
        slider_layout.addItem(self.slider_proxy)
        slider_layout.setContentsMargins(self.CONFIG['y_axis_width'], 5, 0, 10)

        # --- АНАЛІТИЧНА КОЛОНКА (тільки 3 панелі в одній колонці) ---
        self.analytics_panel = self.view.addLayout(row=0, col=1, rowspan=5)
        self.ana_widgets = {}

        for i in range(3):
            a = self.analytics_panel.addPlot(row=i, col=0)
            a.showGrid(x=True, y=True, alpha=0.3)
            self.ana_widgets[f'ana_{i}'] = a

        # Пропорції колонок (channels : analytics)
        self.view.ci.layout.setColumnStretchFactor(0, self.CONFIG['layout_columns_ratio'][0])
        self.view.ci.layout.setColumnStretchFactor(1, self.CONFIG['layout_columns_ratio'][1])

        # Пропорції 2:1:1
        for i, ratio in enumerate(self.CONFIG['layout_ana_rows_ratio']):
            self.analytics_panel.layout.setRowStretchFactor(i, ratio)
        # # Пропорції 2:1:1
        # self.analytics_panel.layout.setRowStretchFactor(0, 2)
        # self.analytics_panel.layout.setRowStretchFactor(1, 1)
        # self.analytics_panel.layout.setRowStretchFactor(2, 1)

        # --- FOOTER ---
        self.footer = QFrame()
        self.footer.setFixedHeight(30)
        self.footer.setStyleSheet(f"background: {self.CONFIG['panel_bg']}; border-top: 1px solid #444;")
        footer_lay = QHBoxLayout(self.footer)
        self.lbl_roi_info = QLabel("ROI: -")
        footer_lay.addWidget(QLabel("Ready"))
        footer_lay.addStretch()
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
        self.channels['Pwr'] = 10 * np.log10(np.abs(self.iq)**2 + 1e-12)
        diff = self.iq[1:] * np.conj(self.iq[:-1])
        self.channels['dPh'] = np.append(np.angle(diff), 0)


    def _setup_standard_rois(self):
        n_points = int(self.CONFIG['defaulyt_rois_dur_sec'] * self.Fs)
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
        
        s0, s1 = int(round(r0)), int(round(r1))
        idx0, idx1 = max(0, s0), min(self.data_len, s1)
        
        if hasattr(self, 'fft_module'):
            fft_size_str = self.combo_fft_size.currentText()
            fft_param = int(fft_size_str) if fft_size_str.isdigit() else None
            self.fft_module.update(self.iq[idx0:idx1], self.Fs, fft_n=fft_param)

        n = idx1 - idx0
        t_sec = n / self.Fs
        if t_sec < 1e-3: t_str = f"{t_sec * 1e6:.1f} us"
        elif t_sec < 1.0: t_str = f"{t_sec * 1e3:.2f} ms"
        else: t_str = f"{t_sec:.3f} s"
        self.lbl_roi_info.setText(f"Start: {idx0:,} | N: {n:,} | Duration: {t_str}")

    def _setup_marker_logic(self):
        self.plots['I'].sigXRangeChanged.connect(self._update_markers)

    def _update_markers(self, _, range_x):
        n_visible = range_x[1] - range_x[0]
        symbol = self.CONFIG['marker_symbol'] if n_visible < self.CONFIG['marker_threshold'] else None
        for name in self.rows_names:
            if name in self.curves:
                self.curves[name].setSymbol(symbol)
                if symbol:
                    self.curves[name].setSymbolSize(self.CONFIG['marker_size'])
                    self.curves[name].setSymbolBrush(self.CONFIG['colors'][name])

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
            self.curves[name] = self.plots[name].plot(
                self.channels[name], 
                pen=pg.mkPen(color, width=self.CONFIG['line_width'])
            )

    def _reset_zoom(self):
        self.plots['I'].setXRange(0, self.data_len)
        self.plots['I'].autoRange(padding=0)


def deploy_vsa_pro(iq: np.ndarray, Fs: float, **kwargs):
    ln = iq.size
    f_id = kwargs.pop('file_id', "---")
    app = QApplication.instance() or QApplication(sys.argv)
    try:
        gui = VSAProWindow(iq, Fs, f_id, **kwargs)
        gui.show()
        app.exec()
        return gui
    except Exception:
        print(f"{ERR} VSA Pro Crash:")
        traceback.print_exc()
        sys.exit(0)