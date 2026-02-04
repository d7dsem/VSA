import sys
from typing import Dict, Tuple, Optional
import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel
from PyQt6.QtCore import Qt, QEvent

# Спроба імпорту твого колоризатора
try:
    from colorizer import inject_colors_into
    inject_colors_into(globals())
except ImportError:
    INFO = "[INFO]"; WARN = "[WARN]"; ERR = "[ERR]"; DBG = "[DBG]"

class MultiPatternSelector(QMainWindow):
    def __init__(self, patterns: Dict, title="Multi-Pattern Selector"):
        super().__init__()
        self.setWindowTitle(title)
        self.resize(1200, 900)
        self.selected_region = (0, 0)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Панель допомоги (світло-жовта, як у професійному софті)
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
        layout.addWidget(self.info_label)

        self.plots = []
        self.regions = []

        for i, (name, (stack_data, color)) in enumerate(patterns.items()):
            pw = pg.PlotWidget(title=name)
            
            # --- ЛОГІКА МИШІ: ЛКМ = ТЯГАТИ (PAN) ---
            vb = pw.getViewBox()
            vb.setMouseMode(pg.ViewBox.PanMode) # Режим "руки"
            vb.setMouseEnabled(x=True, y=False) # Тільки по осі часу
            pw.setMenuEnabled(False)            # Прибираємо меню правої кнопки
            
            # Фільтр для кастомного зуму колесом
            pw.viewport().installEventFilter(self)
            
            layout.addWidget(pw)
            self.plots.append(pw)

            # Малюємо фоновий стек (сірим)
            for row in stack_data[:30]:
                pw.plot(row, pen=pg.mkPen(color='#888', width=0.5, alpha=50))
            
            # Головна лінія (Медіана)
            median = np.median(stack_data, axis=0)
            pw.plot(median, pen=pg.mkPen(color, width=2))
            pw.showGrid(x=True, y=True, alpha=0.3)

            # Селектор області (максимально високий Z-Value, щоб не конфліктувати з паном)
            rgn = pg.LinearRegionItem([len(median)//4, len(median)//2])
            rgn.setZValue(1000) 
            pw.addItem(rgn)
            self.regions.append(rgn)
            rgn.sigRegionChanged.connect(self.sync_regions)

            # Синхронізація осей X
            if i > 0:
                pw.setXLink(self.plots[0])

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
        
        # Динамічний текст підказки
        coords = f"📍 REGION: {self.selected_region[0]} — {self.selected_region[1]} (Len: {self.selected_region[1] - self.selected_region[0]})"
        mouse_help = "🖱️ LMB Drag: Pan Time | Wheel: Zoom Time | Ctrl+Wheel: Zoom Amp"
        keys_help = "⌨️ ENTER/SPACE: Confirm | ESC: Cancel | A: Reset View"
        
        self.info_label.setText(f"<b>{coords}</b><br>{mouse_help}<br>{keys_help}")

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Space, Qt.Key.Key_Return):
            self.close()
        elif event.key() == Qt.Key.Key_Escape:
            self.selected_region = None
            self.close()
        elif event.key() == Qt.Key.Key_A:
            for pw in self.plots:
                pw.autoRange()

def select_signal_region(patterns: Dict, title: str = "Pattern Selector") -> Optional[Tuple[int, int]]:
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
    selector = MultiPatternSelector(validated, title)
    selector.show()
    app.exec()
    
    # Повертаємо None тільки якщо в MultiPatternSelector натиснули ESC
    return selector.selected_region