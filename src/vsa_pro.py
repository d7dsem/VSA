import sys
import traceback
import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QFrame, QLabel, QTextEdit, QComboBox, QRadioButton, 
                             QButtonGroup, QSlider, QPushButton, QGraphicsProxyWidget) 
from PyQt6.QtCore import Qt

# Local imports
from colorizer import colorize, inject_colors_into
# --- color names for IDE/static analysis suppress warnings --------------------
GREEN: str; BRIGHT_GREEN: str; RED: str; BRIGHT_RED: str
YELLOW: str; BRIGHT_YELLOW: str; BLACK: str; BRIGHT_BLACK: str
GRAY: str; BRIGHT_GRAY: str; CYAN: str; BRIGHT_CYAN: str
BLUE: str; BRIGHT_BLUE: str; MAGENTA: str; BRIGHT_MAGENTA: str
WHITE: str; BRIGHT_WHITE: str; RESET: str; WARN: str; ERR: str; INFO: str; DBG: str
inject_colors_into(globals())


class VSAProWindow(QWidget):
    CONFIG = {
        'colors': {'I': '#5af', 'Q': '#f5a', 'Pwr': '#0f0', 'dPh': '#ff0'},
        'roi_brush': (255, 255, 0, 40),
        'roi_hover_brush': (255, 255, 0, 70),
        'line_width': 1.0,
        'y_axis_width': 65,
        'bg_color': '#1a1a1a',
        'panel_bg': '#222',
        
        "defaulyt_rois_dur_sec": 0.1e-3,
        'min_roi_samples': 2,        # Мінімум 2 точки для аналізу
        'marker_threshold': 400, 
        'marker_size': 7,
        'marker_symbol': 'o',
        'hotkey_reset': Qt.Key.Key_A
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
        self._setup_standard_rois()
        self._fill_initial_data()
        self._setup_slider_logic()
        self._setup_marker_logic()
        
        self.setWindowTitle(f"VSA Pro - {file_id}")
        self.resize(1400, 900)
        self.setStyleSheet(f"background-color: {self.CONFIG['bg_color']}; color: #eee;")

    def _calculate_projections(self):
        self.channels['I'] = self.iq.real
        self.channels['Q'] = self.iq.imag
        self.channels['Pwr'] = 10 * np.log10(np.abs(self.iq)**2 + 1e-12)
        diff = self.iq[1:] * np.conj(self.iq[:-1])
        self.channels['dPh'] = np.append(np.angle(diff), 0)

    def init_layout(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.top_panel = QFrame()
        self.top_panel.setFixedHeight(45)
        self.top_panel.setStyleSheet(f"background: {self.CONFIG['panel_bg']}; border-bottom: 1px solid #444;")
        top_lay = QHBoxLayout(self.top_panel)
        top_lay.addWidget(QLabel("<b>VSA PRO</b>"))
        top_lay.addStretch()
        self.btn_reset = QPushButton(f"Reset Zoom ({chr(self.CONFIG['hotkey_reset'])})")
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
            
            # Обмеження зуму камери (не ближче 2 семплів)
            p.vb.setLimits(minXRange=2)
            p.vb.setMenuEnabled(False) 

            self.plots[name] = p
            if i > 0: p.setXLink(self.plots['I'])

        self.slider_proxy = QGraphicsProxyWidget()
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider_proxy.setWidget(self.slider)
        slider_layout = self.view.addLayout(row=4, col=0)
        slider_layout.addItem(self.slider_proxy)
        slider_layout.setContentsMargins(self.CONFIG['y_axis_width'], 5, 0, 10)

        self.analytics_panel = self.view.addLayout(row=0, col=1, rowspan=5)
        self.view.ci.layout.setColumnStretchFactor(0, 3)
        self.view.ci.layout.setColumnStretchFactor(1, 1)

        self.footer = QFrame()
        self.footer.setFixedHeight(30)
        self.footer.setStyleSheet(f"background: {self.CONFIG['panel_bg']}; border-top: 1px solid #444;")
        footer_lay = QHBoxLayout(self.footer)
        self.lbl_roi_info = QLabel("ROI: -")
        footer_lay.addWidget(QLabel("Ready"))
        footer_lay.addStretch()
        footer_lay.addWidget(self.lbl_roi_info)
        main_layout.addWidget(self.footer)

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
        
        # Виправлення мінімального розміру
        if (r1 - r0) < self.CONFIG['min_roi_samples']:
            r1 = r0 + self.CONFIG['min_roi_samples']
            source_roi.blockSignals(True)
            source_roi.setRegion([r0, r1])
            source_roi.blockSignals(False)
        
        region = [r0, r1]
        # Синхронізація всіх каналів
        for roi in self.rois:
            if roi is not source_roi:
                roi.blockSignals(True)
                roi.setRegion(region)
                roi.blockSignals(False)
        
        s0, s1 = int(round(r0)), int(round(r1))
        n = max(self.CONFIG['min_roi_samples'], s1 - s0)
        t_sec = n / self.Fs
        
        if t_sec < 1e-3:
            time_str = f"{t_sec * 1e6:.2f} µs"
        elif t_sec < 1.0:
            time_str = f"{t_sec * 1e3:.2f} ms"
        else:
            time_str = f"{t_sec:.3f} s"

        self.lbl_roi_info.setText(f"Start: {s0:,} | N: {n:,} | Duration: {time_str}")

    def _setup_marker_logic(self):
        self.plots['I'].sigXRangeChanged.connect(self._update_markers)

    def _update_markers(self, _, range_x):
        n_visible = range_x[1] - range_x[0]
        symbol = self.CONFIG['marker_symbol'] if n_visible < self.CONFIG['marker_threshold'] else None
        for name in self.rows_names:
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

    def keyPressEvent(self, event):
        vb = self.plots['I'].vb
        view_range = vb.viewRange()[0]
        view_width = view_range[1] - view_range[0]
        if event.key() == Qt.Key.Key_Home:
            self.plots['I'].setXRange(0, view_width, padding=0)
        elif event.key() == Qt.Key.Key_End:
            self.plots['I'].setXRange(self.data_len - view_width, self.data_len, padding=0)
        elif event.key() == self.CONFIG['hotkey_reset']:
            self._reset_zoom()
        super().keyPressEvent(event)




def deploy_vsa_pro(iq: np.ndarray, Fs: float, **kwargs):
    ln = iq.size
    dur = ln / Fs
    f_id = kwargs.pop('file_id', "---")
    start_offs = kwargs.get("offset", 0)
    
    print(f"{INFO} Start {YELLOW}VSA pro{RESET} for file {YELLOW}{f_id}{RESET}. "
          f"From {start_offs} to {start_offs+ln} ({GREEN}{ln:_}{RESET} dur {CYAN}{dur*1e3:.2f}{RESET}ms).")
    
    app = QApplication.instance() or QApplication(sys.argv)
    try:
        gui = VSAProWindow(iq, Fs, f_id, **kwargs)
        gui.show()
        app.exec()
        return gui
    except Exception:
        print(MAGENTA + "="*32 + RESET)
        print(f"{ERR} VSA Pro Crash:")
        traceback.print_exc()
        print(MAGENTA + "="*32 + RESET)
        sys.exit(0)
        
        
        