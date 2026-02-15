from pathlib import Path
import sys
from typing import Dict, Tuple, Optional
import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QApplication, QMainWindow, QHBoxLayout, QVBoxLayout, QWidget, QLabel, QVBoxLayout
from PyQt6.QtCore import Qt, QEvent
from PyQt6.QtGui import QIcon, QPixmap, QPainter, QColor, QBrush

# Спроба імпорту твого колоризатора
try:
    from colorizer import inject_colors_into
    inject_colors_into(globals())
except ImportError:
    INFO = "[INFO]"; WARN = "[WARN]"; ERR = "[ERR]"; DBG = "[DBG]"

class MultiPatternSelector(QMainWindow):
    # Layout configuration
    ZOOM_STRETCH = 1
    PLOTS_STRETCH = 3
    
    def __init__(self, patterns: Dict,  Fs: float, id="", 
                 visual_guard=None, lri_wdt=None):
        super().__init__()
        self.setWindowTitle(f"Multi-Pattern Selector [{id}]")
        self.resize(1200, 900)
        # Встановити іконку (якщо файл існує)
        icon_path = Path('icon.png')
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))
        else:
            # Створити іконку програмно якщо файлу немає
            pixmap = QPixmap(32, 32)
            painter = QPainter(pixmap)
            painter.fillRect(0, 0, 32, 32, QBrush(QColor(40, 40, 50)))
            painter.setBrush(QBrush(QColor(100, 200, 255)))
            painter.drawEllipse(4, 4, 24, 24)
            painter.end()
            self.setWindowIcon(QIcon(pixmap))

        self.selected_region = (0, 0)
        self.Fs = Fs
        self.patterns: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)  # вертикальний
        # Горизонтальна частина для zoom+plots
        h_layout = QHBoxLayout()
        
        # Ліва частина - zoom область
        zoom_widget = QWidget()
        zoom_layout = QVBoxLayout(zoom_widget)
        h_layout.addWidget(zoom_widget, stretch=self.ZOOM_STRETCH)
        
        # Права частина - plots
        plots_widget = QWidget()
        layout = QVBoxLayout(plots_widget)
        h_layout.addWidget(plots_widget, stretch=self.PLOTS_STRETCH)
        
        # Додати h_layout в main_layout
        main_layout.addLayout(h_layout)

        self.plots = []
        self.regions = []
        self.stack_i = None
        self.stack_q = None
        self.stack_pwr = None
        self.stack_dph = None
        # Створити zoom графіки (4 штуки для Power, Phase, I, Q)
        self.zoom_plots = []
        for name in ["Power (dB)", "Phase (rad)", "I Component", "Q Component"]:
            zpw = pg.PlotWidget(title=f"Zoom: {name}")
            zpw.showGrid(x=True, y=True, alpha=0.3)
            zoom_layout.addWidget(zpw)
            self.zoom_plots.append(zpw)

        for i, (name, (stack_data, color)) in enumerate(patterns.items()):
            pw = pg.PlotWidget(title=name)
            
            # --- ЛОГІКА МИШІ: ЛКМ = ТЯГАТИ (PAN) ---
            vb = pw.getViewBox()
            vb.setMouseMode(pg.ViewBox.PanMode) # Режим "руки"
            vb.setMouseEnabled(x=True, y=False) # Тільки по осі часу
            pw.setMenuEnabled(False)            # Прибираємо меню правої кнопки
            
            # Фільтр для кастомного зуму колесом
            pw.viewport().installEventFilter(self)
            # Double-click для переміщення region
            pw.scene().sigMouseClicked.connect(lambda evt, plot_idx=i: self.on_plot_click(evt, plot_idx))
            
            layout.addWidget(pw)
            self.plots.append(pw)

            # Малюємо фоновий стек (сірим)
            for row in stack_data[:30]:
                pw.plot(row, pen=pg.mkPen(color='#888', width=0.5, alpha=50))
            
            # Головна лінія (Медіана)
            median = np.median(stack_data, axis=0)
            if name == "I Component":
                self.stack_i = stack_data
            if name == "Q Component":
                self.stack_q = stack_data
            if name == "Power (dB)":
                self.stack_pwr = stack_data
            if name == "Phase (rad)":
                self.stack_dph = stack_data
            pw.plot(median, pen=pg.mkPen(color, width=2))
            pw.showGrid(x=True, y=True, alpha=0.3)
            if visual_guard is not None:
                left_line = pg.InfiniteLine(pos=visual_guard, angle=90, pen=pg.mkPen('cyan', width=2))
                right_line = pg.InfiniteLine(pos=len(median) - visual_guard, angle=90, pen=pg.mkPen('cyan', width=2))
                pw.addItem(left_line)
                pw.addItem(right_line)
            # Селектор області (максимально високий Z-Value, щоб не конфліктувати з паном)
            if visual_guard is not None and lri_wdt is not None:
                burst_wnd = len(median) - 2*visual_guard
                center = visual_guard + burst_wnd / 2
                bounds = [center - lri_wdt/2, center + lri_wdt/2]
            else:
                bounds = [len(median)//4, len(median)//2]
            rgn = pg.LinearRegionItem(bounds)
            rgn.setZValue(1000) 
            pw.addItem(rgn)
            self.regions.append(rgn)
            rgn.sigRegionChanged.connect(self.sync_regions)

            # Синхронізація осей X
            if i > 0:
                pw.setXLink(self.plots[0])

        # Панель допомоги
        self.info_label = QLabel()
        self.info_label.setStyleSheet("""
            QLabel {
                background-color: #fff9c4; 
                color: #222; 
                padding: 12px; 
                border: 1px solid #d4d0a1;
                border-radius: 4px;
                font-size: 13px;
                font-family: 'Consolas', 'Monaco', monospace;
                line-height: 140%;
            }
        """)
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        main_layout.addWidget(self.info_label)
        self.update_info()

    def eventFilter(self, watched, event):
        """ Кастомний зум: Wheel=X, Ctrl+Wheel=Y """
        if event.type() == QEvent.Type.Wheel:
            modifiers = QApplication.keyboardModifiers()
            delta = event.angleDelta().y()
            zoom_factor = 0.85 if delta > 0 else 1.15
            
            if modifiers == Qt.KeyboardModifier.ControlModifier:
                # Зум Y для конкретного вікна
                for pw in self.plots:
                    if watched is pw.viewport():
                        pw.getViewBox().scaleBy(y=zoom_factor)
                        return True
            else:
                # Зум X для всіх синхронно
                self.plots[0].getViewBox().scaleBy(x=zoom_factor)
                return True
        return super().eventFilter(watched, event)

    def sync_regions(self, sender_region):
        new_range = sender_region.getRegion()
        for rgn in self.regions:
            if rgn is sender_region: continue
            rgn.blockSignals(True)
            rgn.setRegion(new_range)
            rgn.blockSignals(False)
        self.update_info(sender_region)

    def update_info(self, active_region=None):
        r = active_region or self.regions[0]
        s, e = r.getRegion()
        self.selected_region = (int(max(0, s)), int(e))
        if all([self.stack_pwr is not None, self.stack_dph is not None, 
                        self.stack_i is not None, self.stack_q is not None]):
                    s_int, e_int = self.selected_region
                    
                    # Витягти median для кожного стека
                    stacks = [self.stack_pwr, self.stack_dph, self.stack_i, self.stack_q]
                    colors = ['red', 'green', 'blue', 'orange']
                    
                    for idx, (stack, zpw, color) in enumerate(zip(stacks, self.zoom_plots, colors)):
                        zpw.clear()
                        median_zoomed = np.median(stack[:, s_int:e_int], axis=0)
                        zpw.plot(median_zoomed, pen=pg.mkPen(color, width=2))
            
        ln = self.selected_region[1] - self.selected_region[0]
        dur = ln / self.Fs
        # Динамічний текст підказки
        coords = f"📍 REGION: {self.selected_region[0]} — {self.selected_region[1]} (Len: {ln}, dur {dur*1e6:.3f} us)"
        mouse_help = "🖱️ LMB Drag: Pan Time | Wheel: Zoom Time | Ctrl+Wheel: Zoom Amp"
        keys_help = "⌨️ ENTER/SPACE: Confirm | ESC: Cancel | A: Reset View"
        
        self.info_label.setText(f"<b>{coords}</b><br>{mouse_help}<br>{keys_help}")


    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Space, Qt.Key.Key_Return):
            # Розрахувати templates перед закриттям
            if all([self.stack_pwr is not None, self.stack_dph is not None,
                    self.stack_i is not None, self.stack_q is not None]):
                s, e = self.selected_region
                template_pwr = np.median(self.stack_pwr[:, s:e], axis=0)
                template_phase = np.median(self.stack_dph[:, s:e], axis=0)
                template_i = np.median(self.stack_i[:, s:e], axis=0)
                template_q = np.median(self.stack_q[:, s:e], axis=0)
                self.patterns = (template_pwr, template_phase, template_i, template_q)
            self.close()
        elif event.key() == Qt.Key.Key_Escape:
            self.patterns = None  # Скасування
            self.close()
        elif event.key() == Qt.Key.Key_A:
            for pw in self.plots:
                pw.autoRange()
                

    def on_plot_click(self, evt, plot_idx):
        """Double-click переміщує region до позиції кліку"""
        if evt.double():  # Якщо double-click
            pos = evt.scenePos()
            # Отримати ViewBox для цього plot
            vb = self.plots[plot_idx].getViewBox()
            # Конвертувати координати scene → data
            mouse_point = vb.mapSceneToView(pos)
            x_click = mouse_point.x()
            
            # Отримати поточну ширину region
            current_region = self.regions[0].getRegion()
            width = current_region[1] - current_region[0]
            
            # Встановити нову позицію (центр на кліку)
            new_start = x_click - width / 2
            new_end = x_click + width / 2
            
            # Оновити всі regions
            for rgn in self.regions:
                rgn.setRegion([new_start, new_end])


def select_overlayed_signal_region(patterns: Dict, Fs: float, id: str = "---", **kwargs) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Returns mean patterns
        power, dPhase, I, Q
    """
    validated = {}
    for name, (data, color) in patterns.items():
        arr = np.asanyarray(data)
        if arr.size == 0: continue
        if arr.dtype == object:
            min_l = min(len(row) for row in arr if hasattr(row, '__len__'))
            arr = np.array([row[:min_l] for row in arr], dtype=np.float32)
        if np.iscomplexobj(arr): arr = np.abs(arr)
        validated[name] = (arr, color)

    if not validated: return None

    app = QApplication.instance() or QApplication(sys.argv)
    visual_guard = kwargs.get('visual_guard', None)
    lri_wdt = kwargs.get('lri_wdt', None)
    selector = MultiPatternSelector(validated, Fs, id, visual_guard, lri_wdt)
    selector.show()
    app.exec()
    
    # Повертаємо None тільки якщо в MultiPatternSelector натиснули ESC
    return selector.patterns