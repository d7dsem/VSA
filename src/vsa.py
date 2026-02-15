from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple, runtime_checkable, Protocol

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backend_bases import Event
from matplotlib.figure import Figure
from matplotlib.collections import LineCollection
from matplotlib.transforms import offset_copy
matplotlib.use("TkAgg")


def get_scale_from_units(unit: Literal["M", "K"]) -> float:
    if unit == "K": return 1e-3
    if unit == "M": return 1e-6
    return 1

@runtime_checkable
class ControledVidget(Protocol):
    stop_requested: bool
    paused: bool
    fig_alive: bool
    delta: int
    render_dt: float
    
    def update(spec, **kwargs):
        ...

@dataclass
class D7Figure:
    fig: Figure
    _paused: bool = False
    _stop_req: bool = False
    _left_req: bool = False
    _right_req: bool = False
    _step_delta: int = 0
    _reset_peaks_req: bool = False
    _peak_hold_toggle_req: bool = False
    
    
    def __post_init__(self):
        """Автоматично підключаємо події після створення об'єкта."""
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _on_key(self, ev: Event) -> None:
        print(f"[_ON_KEY] key={ev.key!r}", flush=True)
        
        match ev.key:
            case "escape":
                plt.close(self.fig)
                self._stop_req = True
            case " " | "space":
                self._paused = not self._paused
            case "right":
                if self._paused:
                    self._step_delta += 1
            case "left":
                if self._paused:
                    self._step_delta -= 1
            case "r" | "к": # підтримка обох розкладок
                self._reset_peaks_req = True
            case "h" | "р": # клавіша 'h' для перемикання (укр. 'р')
                self._peak_hold_toggle_req = True

def deploy_layout() -> Tuple[D7Figure, Axes, Axes, Axes]:
    # Створюємо фігуру
    fig = plt.figure(figsize=(12, 8)) # Трохи ширша для панелі
    
    # Створюємо сітку: 2 рядки, 2 колонки
    # Колонки: 95% під графіки, 5% під Colorbar/Контролі
    # Рядки: 70% під Спектр, 30% під Водоспад
    gs = fig.add_gridspec(
        2, 2, 
        width_ratios=[95, 5], 
        height_ratios=[7, 3],
        hspace=0.0, 
        wspace=0.1
    )

    # Основні осі для графіків
    ax_spec = fig.add_subplot(gs[0, 0])
    ax_wfall = fig.add_subplot(gs[1, 0], sharex=ax_spec)
    
    # Окремі осі для Colorbar (займають обидва рядки в правій колонці)
    ax_side = fig.add_subplot(gs[:, 1])

    # Косметика
    ax_spec.tick_params(labelbottom=False)
    fig.tight_layout()

    d7fg = D7Figure(fig=fig)
    
    return d7fg, ax_spec, ax_wfall, ax_side

class SpecVidget:
    def __init__(
        self,
        ax: Axes,
        freqs: np.ndarray,
        freq_units: Literal["M", "K"] = "M",

        title_base: Optional[str] = None,
        y_lim: Optional[Tuple[float, float]] = None,

        # === shrink/expand y control  ===
        y_guard_frac: float = 0.25,
        spec_y_hyst_up: float = 4.0,
        spec_y_hyst_dn: float = 6.0,

        y_shrink_hold_updates: int = 12,
        y_shrink_recover_updates: int = 4,
        y_shrink_margin_db: float = 2.0,
        **kwargs
    ):
        """
        Spectrum visualization.

        """
        self.ax_spec = ax
        self.fig = ax.get_figure()
        self._y_label_spec = "Power (dBFS)"
        
        self.freq_units = freq_units
        self.freq_scale: float = get_scale_from_units(freq_units)

        self.freq_bins = freqs
        self.x_freq = self.freq_bins * self.freq_scale

        self.dF = float(self.freq_bins[1] - self.freq_bins[0])

        self._base_title = title_base if title_base else (
            f"SPECTRUM  FFT={len(freqs)}  |  dF={self.dF * self.freq_scale:.3f} {freq_units}Hz  |"
        )

        # Hard y-limits
        self._spec_y_lim: Optional[Tuple[float, float]] = y_lim

        # Hysteresis for spectrum
        self._spec_y_hyst_up = float(spec_y_hyst_up)
        self._spec_y_hyst_dn = float(spec_y_hyst_dn)

        # Guard fraction for y-limit updates
        self._y_guard_frac = y_guard_frac

        self._y_shrink_hold_updates = int(y_shrink_hold_updates)
        self._y_shrink_recover_updates = max(int(y_shrink_recover_updates), 1)
        self._y_shrink_margin_db = float(y_shrink_margin_db)

        self._y_quiet_updates = 0
        self._y_shrink_active = False
        self._y_shrink_step_ymax = 0.0
        self._y_shrink_step_ymin = 0.0

        # Spectrum y-scale state
        self._spec_ylim_inited = False
        self._spec_ymin = 0.0
        self._spec_ymax = 0.0

        # Control
        self._stop = False
        self.paused: bool = False
        self._step_delta: int = 0

        # ===== SPECTRUM PANEL (Stems + Markers) =====
        # 1. Створюємо колекцію для вертикальних "ніжок" (гвоздиків)
        # alpha=0.5 зробить їх напівпрозорими, що дуже круто виглядає на сітці
        self._stem_lines = LineCollection([], colors="orange", linewidths=0.5, alpha=0.5, zorder=1)
        self.ax_spec.add_collection(self._stem_lines)

        (self._line_raw_spec,) = self.ax_spec.plot(
            self.x_freq,
            np.zeros_like(self.x_freq),
            linestyle="",
            marker=".",
            markersize=2.0,
            color="orange",
            zorder=2,            # zorder вище, щоб точки були поверх ніжок
        )

        (self._line_smooth_spec,) = self.ax_spec.plot(
            self.x_freq,
            np.zeros_like(self.x_freq),
            linestyle="-",
            linewidth=1.0,
            color="black",
            zorder=3,
        )
        self._line_smooth_spec.set_visible(False)

        self.peak_hold_enabled = kwargs.get('peak_hold', False)
        self._peak_hold_data = np.full_like(self.x_freq, -150.0) # Початкове значення нижче плінтуса
        (self._line_peak_hold,) = self.ax_spec.plot(
            self.x_freq,
            self._peak_hold_data,
            linestyle="-",
            linewidth=1.0,
            color="red",
            alpha=0.6,
            zorder=2,
            label="Peak Hold",
            visible=self.peak_hold_enabled
        )

        # ===== AUX LINES (V-lines & H-lines) =====
        
        # Колекція для вертикальних ліній (наприклад, піки, межі каналів)
        self._v_lines = LineCollection([], colors="red", linewidths=1.0, 
                                       linestyle="--", alpha=0.7, zorder=4)
        self.ax_spec.add_collection(self._v_lines)

        # Колекція для горизонтальних ліній (наприклад, рівні потужності, пороги)
        self._h_lines = LineCollection([], colors="green", linewidths=1.0, 
                                       linestyle=":", alpha=0.7, zorder=4)
        self.ax_spec.add_collection(self._h_lines)
        
        self.ax_spec.set_xlabel(f"frequency ({freq_units}Hz)")
        self.ax_spec.set_ylabel(self._y_label_spec)
        self.ax_spec.set_title(self._base_title)
        self.ax_spec.grid(True)
        
        self._v_texts = []
        self._h_texts = []

        if self._spec_y_lim is not None:
            self.ax_spec.set_ylim(self._spec_y_lim[0], self._spec_y_lim[1])

        plt.ion()
        self.fig.tight_layout(pad=2)
        plt.show(block=False)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def _on_figure_close(self, ev) -> None:
        print(f"[_ON_FIGURE_CLOSE] Figure closed by user", flush=True)
        self._stop = True

    @property
    def stop_requested(self) -> bool:
        return self._stop

    def reset_peak_hold(self):
        """Метод для скидання піків (можна повісити на клавішу)"""
        self._peak_hold_data[:] = -150.0

    def figure_alive(self) -> bool:
        try:
            return (self.fig is not None) and plt.fignum_exists(self.fig.number)
        except Exception:
            return False

    def _shrink_reset(self) -> None:
        self._y_quiet_updates = 0
        self._y_shrink_active = False
        self._y_shrink_step_ymax = 0.0
        self._y_shrink_step_ymin = 0.0

    def _shrink_arm(self, tgt_ymin: float, tgt_ymax: float) -> None:
        # fixed step so "recover in N updates" is literal
        dy_max = max(self._spec_ymax - tgt_ymax, 0.0)
        dy_min = max(tgt_ymin - self._spec_ymin, 0.0)
        self._y_shrink_step_ymax = dy_max / self._y_shrink_recover_updates
        self._y_shrink_step_ymin = dy_min / self._y_shrink_recover_updates
        self._y_shrink_active = True

    def _update_spec_ylim(self, y_ref: np.ndarray) -> None:
        if self._spec_y_lim is not None:
            return

        y_min = float(np.min(y_ref))
        y_max = float(np.max(y_ref))

        # target window (always from current data span)
        data_span = max(y_max - y_min, 1e-12)
        guard = self._y_guard_frac * data_span
        tgt_ymin = y_min - guard
        tgt_ymax = y_max + guard

        if not self._spec_ylim_inited:
            self._spec_ymin = tgt_ymin
            self._spec_ymax = tgt_ymax
            self._spec_ylim_inited = True
            self._shrink_reset()
            self.ax_spec.set_ylim(self._spec_ymin, self._spec_ymax)
            return

        expanded = False

        # expand-only (hysteresis)
        if y_max > self._spec_ymax + self._spec_y_hyst_up:
            self._spec_ymax = y_max + guard
            expanded = True
        if y_min < self._spec_ymin - self._spec_y_hyst_dn:
            self._spec_ymin = y_min - guard
            expanded = True

        if expanded:
            self._shrink_reset()
        else:
            # quiet detector (must be safely inside window)
            if (y_max < self._spec_ymax - self._y_shrink_margin_db) and (y_min > self._spec_ymin + self._y_shrink_margin_db):
                self._y_quiet_updates += 1
            else:
                self._shrink_reset()

            # arm shrink after hold
            if (not self._y_shrink_active) and (self._y_quiet_updates >= self._y_shrink_hold_updates):
                self._shrink_arm(tgt_ymin, tgt_ymax)

            # apply shrink with fixed steps
            if self._y_shrink_active:
                if self._spec_ymax > tgt_ymax:
                    step = max(self._y_shrink_step_ymax, 0.0)
                    self._spec_ymax = max(tgt_ymax, self._spec_ymax - step)

                if self._spec_ymin < tgt_ymin:
                    step = max(self._y_shrink_step_ymin, 0.0)
                    self._spec_ymin = min(tgt_ymin, self._spec_ymin + step)

                # done?
                if (self._spec_ymax <= tgt_ymax) and (self._spec_ymin >= tgt_ymin):
                    self._shrink_reset()

        self.ax_spec.set_ylim(self._spec_ymin, self._spec_ymax)

    def update(
            self,
            spectr_arr: np.ndarray,
            smoothed: Optional[np.ndarray] = None,
            title: Optional[str] = None,
            v_coords: Optional[np.ndarray] = None,  # Масив частот (x)
            h_coords: Optional[np.ndarray] = None   # Масив рівнів (y)
        ) -> None:
            self._peak_hold_data = np.maximum(self._peak_hold_data, spectr_arr)
            self._line_peak_hold.set_ydata(self._peak_hold_data)
            # 1. Оновлюємо капелюшки (точки)
            self._line_raw_spec.set_ydata(spectr_arr)

            # 2. ФОРМУЄМО ГВОЗДИКИ (Stems)
            # Використовуємо поточну нижню межу як базу для ніжок
            y_low = self._spec_ymin
            
            # Створюємо масив сегментів: [[[x1, y_low], [x1, y1]], [[x2, y_low], [x2, y2]], ...]
            # Стакємо x та y для нижніх точок
            stems_low = np.column_stack([self.x_freq, np.full_like(self.x_freq, y_low)])
            # Стакємо x та y для верхніх точок (значення спектра)
            stems_high = np.column_stack([self.x_freq, spectr_arr])
            
            # Об'єднуємо їх у пари сегментів
            segments = np.stack([stems_low, stems_high], axis=1)
            
            # Завантажуємо сегменти в колекцію
            self._stem_lines.set_segments(segments)




            # SMOOTH
            if smoothed is None:
                self._line_smooth_spec.set_visible(False)
            else:
                self._line_smooth_spec.set_ydata(smoothed)
                self._line_smooth_spec.set_visible(True)

            _title = self._base_title
            if title:
                _title += f"\n{title}"
            self.ax_spec.set_title(_title)

            # Оновлення ВЕРТИКАЛЬНИХ ліній
            if v_coords is not None and len(v_coords) > 0:
                v_coords_scaled = v_coords * self.freq_scale
                y_low, y_high = self._spec_ymin, self._spec_ymax
                
                # Малюємо лінії (твоя векторизація - вогонь!)
                v_segs = np.stack([
                    np.column_stack([v_coords_scaled, np.full_like(v_coords_scaled, y_low)]),
                    np.column_stack([v_coords_scaled, np.full_like(v_coords_scaled, y_high)])
                ], axis=1)
                self._v_lines.set_segments(v_segs)

                # --- ОНОВЛЮЄМО ПІДПИСИ (Senior style) ---
                y_text_pos = y_high - (y_high - y_low) * 0.02
                
                # 1. Підганяємо кількість текстових об'єктів під кількість координат
                while len(self._v_texts) < len(v_coords):
                    t = self.ax_spec.text(0, 0, "", color='red', fontsize=8, rotation=90, 
                                        verticalalignment='top', horizontalalignment='right',
                                        bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1))
                    self._v_texts.append(t)

                # 2. Оновлюємо та показуємо потрібні, ховаємо зайві
                for i, txt in enumerate(self._v_texts):
                    if i < len(v_coords):
                        f_raw = v_coords[i]
                        f_scaled = v_coords_scaled[i]
                        txt.set_text(f"{f_raw/1e6:.3f}")
                        txt.set_position((f_scaled, y_text_pos))
                        txt.set_visible(True)
                    else:
                        txt.set_visible(False)
            else:
                self._v_lines.set_segments([])
                for txt in self._v_texts:
                    txt.set_visible(False)

            # Оновлення ГОРИЗОНТАЛЬНИХ ліній (h_lines)
            if h_coords is not None and len(h_coords) > 0:
                for txt in self._h_texts:
                    txt.remove()
                self._h_texts.clear()
                x_pos = self.x_freq[-1] # Текст буде біля правого краю
                for val in h_coords:
                    t = self.ax_spec.text(
                        x_pos, val, f"{val:.1f}", 
                        color='green', fontsize=12, 
                        verticalalignment='bottom', horizontalalignment='right',
                        alpha=1.0
                    )
                    self._h_texts.append(t)
                x_low = self.x_freq[0]
                x_high = self.x_freq[-1]
                # Кожна лінія: [(x_low, y), (x_high, y)]
                h_segs = np.stack([
                    np.column_stack([np.full_like(h_coords, x_low), h_coords]),
                    np.column_stack([np.full_like(h_coords, x_high), h_coords])
                ], axis=1)
                self._h_lines.set_segments(h_segs)
            else:
                self._h_lines.set_segments([])
            
            if self.peak_hold_enabled:
                self._peak_hold_data = np.maximum(self._peak_hold_data, spectr_arr)
                self._line_peak_hold.set_ydata(self._peak_hold_data)
                if not self._line_peak_hold.get_visible():
                    self._line_peak_hold.set_visible(True)
            else:
                if self._line_peak_hold.get_visible():
                    self._line_peak_hold.set_visible(False)
                    
            # Оновлюємо межі (тут якраз перераховується self._spec_ymin для наступного кадру)
            y_ref = spectr_arr
            self._update_spec_ylim(y_ref)

            self.fig.canvas.draw_idle()
            plt.pause(0.001)

CMapType = Literal['viridis', 'magma', 'inferno', 'plasma', 'turbo', 'jet']

class WaterfallVidget:
    def __init__(
        self, ax: Axes, 
        freqs: np.ndarray, freq_units: str = "M",
        history_len: int = 100,
        ax_cbar: Optional[Axes] = None,
        cmap_name: CMapType = 'viridis'
    ):
        self.ax = ax
        self.freq_scale = get_scale_from_units(freq_units)
        self.history_len = history_len
        
        # Створюємо порожній буфер для даних (history_len x FFT_size)
        self.data_buffer = np.full((history_len, len(freqs)), -100.0) 
        
        self.img = self.ax.imshow(
            self.data_buffer,
            aspect='auto',
            extent=[freqs[0] * self.freq_scale, freqs[-1] * self.freq_scale, 0, history_len],
            cmap=cmap_name,
            origin='lower',
            interpolation='nearest'
        )
        self.cbar = None
        if ax_cbar:
            # Використовуємо cax для малювання у вже виділеному місці
            self.cbar = self.ax.figure.colorbar(self.img, cax=ax_cbar)
            self.cbar.set_label('Power (dBFS)', fontsize=9)
            ax_cbar.tick_params(labelsize=8)

        self.ax.set_ylabel("Time (frames)")
        self.ax.set_xlabel(f"frequency ({freq_units}Hz)")
        
        # Ініціалізуємо зображення
        self.img = self.ax.imshow(
            self.data_buffer,
            aspect='auto',
            extent=[freqs[0] * self.freq_scale, freqs[-1] * self.freq_scale, 0, history_len],
            cmap='viridis', # Класичний "SDR" колір: від синього до жовтого
            origin='lower',
            interpolation='nearest'
        )
        self.ax.set_ylabel("Time (frames)")
        self.ax.set_xlabel(f"frequency ({freq_units}Hz)")

    def update(self, new_spec: np.ndarray):
        # Зсуваємо дані вгору (старі видаляємо, додаємо новий рядок)
        self.data_buffer = np.roll(self.data_buffer, -1, axis=0)
        self.data_buffer[-1, :] = new_spec
        
        # Оновлюємо картинку
        self.img.set_data(self.data_buffer)
        # Динамічно підлаштовуємо колір під поточні рівні (опціонально)
        self.img.set_clim(vmin=np.min(self.data_buffer), vmax=np.max(self.data_buffer))

class VSA:
    def __init__(self, freq_bins: np.ndarray, fig: D7Figure, 
                 ax_spec: Axes, ax_wfall: Optional[Axes]=None, ax_cbar: Optional[Axes]=None, **kwargs):
        self.d7f = fig
        self.render_dt = kwargs.get('render_dt', 0.001)
        # Ініціалізуємо внутрішній віджет спектра
        self.spec_w = SpecVidget(ax_spec, freq_bins, **kwargs)
        self.wfal_w = WaterfallVidget(
                ax_wfall, 
                freq_bins, 
                freq_units=kwargs.get('freq_units', "M"),
                ax_cbar=ax_cbar,
                history_len=kwargs.get('history_len', 512),
                cmap_name=kwargs.get('cmap_name', 'viridis')
            ) if ax_wfall else None
            
    # --- Реалізація протоколу ControledVidget ---
    
    @property
    def stop_requested(self) -> bool:
        return self.d7f._stop_req or not self.fig_alive

    @property
    def paused(self) -> bool:
        return self.d7f._paused

    @property
    def fig_alive(self) -> bool:
        return plt.fignum_exists(self.d7f.fig.number)

    @property
    def delta(self) -> int:
        # Забираємо дельту і СКИДАЄМО її, щоб не стрибати вічно
        d = self.d7f._step_delta
        self.d7f._step_delta = 0
        return d

    def update(self, spec, smooth=None, title="", **kwargs):
        if self.fig_alive:
            # Обробка скидання піків (клавіша 'r')
            if self.d7f._reset_peaks_req:
                self.spec_w.reset_peak_hold()
                self.d7f._reset_peaks_req = False
            
            # Обробка перемикання видимості (клавіша 'h')
            if self.d7f._peak_hold_toggle_req:
                self.spec_w.peak_hold_enabled = not self.spec_w.peak_hold_enabled
                # Якщо вимкнули — скидаємо дані, щоб при увімкненні почати з чистого листа
                if not self.spec_w.peak_hold_enabled:
                    self.spec_w.reset_peak_hold()
                self.d7f._peak_hold_toggle_req = False

            self.spec_w.update(spec, smooth, title, **kwargs)
            if self.wfal_w: 
                self.wfal_w.update(spec)