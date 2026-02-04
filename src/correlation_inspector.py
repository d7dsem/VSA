import sys
import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel
from PyQt6.QtCore import Qt, QEvent

class CorrelationInspector(QMainWindow):
    def __init__(self, signal_complex, pattern_dphi, Fs, title="Correlation Inspector"):
        super().__init__()
        self.setWindowTitle(title)
        self.resize(1400, 950)
        
        self.sig = signal_complex
        self.pat_dphi = np.asanyarray(pattern_dphi)
        self.pat_len = len(self.pat_dphi)
        
        # Обчислюємо dPhi сигналу (білий графік)
        self.sig_dphi = np.angle(self.sig[1:] * np.conj(self.sig[:-1]))

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # 1. REFERENCE (Жовтий)
        self.p_ref = pg.PlotWidget(title="REFERENCE PATTERN (dPhi)")
        self.p_ref.setFixedHeight(120)
        self.p_ref.plot(self.pat_dphi, pen=pg.mkPen('y', width=2))
        layout.addWidget(self.p_ref)

        # 2. Основні графіки
        self.plots = []
        titles = ["In-Phase (I)", "Quadrature (Q)", "SIGNAL dPhi", "CORRELATION"]
        colors = ['#00ff00', '#00ffff', '#ffffff', '#ff00ff']
        
        for i, t in enumerate(titles):
            pw = pg.PlotWidget(title=t)
            pw.getViewBox().setMouseMode(pg.ViewBox.PanMode)
            pw.getViewBox().setMouseEnabled(x=True, y=False)
            pw.viewport().installEventFilter(self)
            layout.addWidget(pw)
            self.plots.append(pw)
            
            if i > 0: pw.setXLink(self.plots[0])

        # Малюємо дані
        self.plots[0].plot(self.sig.real, pen=pg.mkPen(colors[0], width=0.7))
        self.plots[1].plot(self.sig.imag, pen=pg.mkPen(colors[1], width=0.7))
        self.plots[2].plot(self.sig_dphi, pen=pg.mkPen(colors[2], width=1, alpha=150))
        
        # Обчислюємо повну кореляцію
        self.full_corr = self.compute_full_correlation()
        self.plots[3].plot(self.full_corr, pen=pg.mkPen(colors[3], width=1.5))

        # --- СИНХРОННІ СПАНИ ---
        self.spans = []
        for pw in self.plots:
            # Створюємо спан для кожного графіка
            span = pg.LinearRegionItem([0, self.pat_len], movable=True)
            span.setZValue(1000)
            pw.addItem(span)
            self.spans.append(span)
            # Прив'язуємо рух
            span.sigRegionChanged.connect(self.sync_spans)

        self.info_label = QLabel("Drag any span to inspect")
        self.info_label.setStyleSheet("font-family: monospace; font-size: 14px; padding: 5px; background: #111; color: #eee;")
        layout.addWidget(self.info_label)

    def compute_full_correlation(self):
        s = self.sig_dphi
        p = self.pat_dphi
        s_norm = (s - np.mean(s)) / (np.std(s) * len(p))
        p_norm = (p - np.mean(p)) / (np.std(p))
        return np.convolve(s_norm, p_norm[::-1], mode='same')

    def sync_spans(self, sender):
        """ Синхронізація позиції та кольору всіх спанів """
        s_idx, _ = sender.getRegion()
        s_idx = int(max(0, s_idx))
        e_idx = s_idx + self.pat_len
        
        # Блокуємо сигнали, щоб не було рекурсії
        for span in self.spans:
            if span is sender: continue
            span.blockSignals(True)
            span.setRegion([s_idx, e_idx])
            span.blockSignals(False)
            
        # ОБЧИСЛЮЄМО КОРЕЛЯЦІЮ ДЛЯ КОЛЬОРУ
        chunk = self.sig_dphi[s_idx : e_idx]
        if len(chunk) == self.pat_len:
            corr = np.corrcoef(chunk, self.pat_dphi)[0, 1]
            
            # Визначаємо колір (від білого до яскраво-зеленого)
            # Чим вища кореляція, тим менше прозорості та більше зелені
            alpha = int(50 + (max(0, corr)**2) * 150) # 50 to 200
            if corr > 0.8:
                brush = pg.mkBrush(0, 255, 0, alpha)
                text_clr = "#00ff00"
            elif corr > 0.4:
                brush = pg.mkBrush(255, 255, 0, alpha) # Жовтий для середньої
                text_clr = "#ffff00"
            else:
                brush = pg.mkBrush(255, 255, 255, 50) # Білий для шуму
                text_clr = "#ffffff"
                
            for span in self.spans:
                span.setBrush(brush)
                
            self.info_label.setText(
                f"<span style='color:{text_clr}'>POS: {s_idx:8d} | "
                f"CORRELATION: {corr:.4f}</span>"
            )

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.Wheel:
            delta = event.angleDelta().y()
            zoom = 0.85 if delta > 0 else 1.15
            self.plots[0].getViewBox().scaleBy(x=zoom)
            return True
        return super().eventFilter(watched, event)

def inspect_correlation(signal, pattern_dphi, Fs):
    app = QApplication.instance() or QApplication(sys.argv)
    inspector = CorrelationInspector(signal, pattern_dphi, Fs)
    inspector.show()
    app.exec()