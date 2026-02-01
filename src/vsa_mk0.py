from typing import Final, Literal, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import MultipleLocator

# Припускаємо, що fft_core лежить поруч
try:
    from fft_core import get_scale_from_units
except ImportError:
    def get_scale_from_units(units: str) -> float:
        return 1e-6 if units == "M" else 1e-3

matplotlib.use("TkAgg")

class VSA:
    def __init__(
        self,
        freqs: np.ndarray,
        freq_units: Literal["M", "K"] = "M",
        freq_show_idx: Optional[Tuple[int, int]] = None,
        freq_grid_step: Optional[int] = None,
        use_dbfs: bool = True,
        spec_y_lim: Optional[Tuple[float, float]] = None,
        iq_y_lim: Optional[Tuple[float, float]] = None,
        spec_y_hyst_up: float = 3.0,
        spec_y_hyst_dn: float = 3.0,
        iq_y_hyst_up: float = 0.5,
        iq_y_hyst_dn: float = 0.5,
        y_shrink_hold_updates: int = 5,
        y_shrink_recover_updates: int = 10,
        render_dt: float = 0.001,
        title_base: Optional[str] = None
    ):
        self.freq_units = freq_units
        self.freq_scale: float = get_scale_from_units(freq_units)
        
        # Робота з індексами частот
        self.freqs_full = freqs
        self.dF = float(self.freqs_full[1] - self.freqs_full[0])
        
        if freq_show_idx is not None:
            self.idx_l, self.idx_r = freq_show_idx
        else:
            self.idx_l, self.idx_r = 0, len(self.freqs_full) - 1
            
        self.freqs = self.freqs_full[self.idx_l:self.idx_r + 1]
        self.x_freq = self.freqs * self.freq_scale
        
        self.use_dbfs = use_dbfs
        self._y_label_spec = "power (dBFS)" if use_dbfs else "magn (linear)"
        self._base_title = title_base or f"VSA SPECTRUM | dF={self.dF * self.freq_scale:.3f} {freq_units}Hz"

        # Налаштування лімітів та гістерезису
        self._spec_y_lim = spec_y_lim
        self._iq_y_lim = iq_y_lim
        self._spec_y_hyst_up = spec_y_hyst_up
        self._spec_y_hyst_dn = spec_y_hyst_dn
        self._iq_y_hyst_up = iq_y_hyst_up
        self._iq_y_hyst_dn = iq_y_hyst_dn
        
        # Параметри плавного звуження (Relim)
        self._y_shrink_hold_updates = y_shrink_hold_updates
        self._y_shrink_recover_updates = max(y_shrink_recover_updates, 1)
        self._y_guard_frac = 0.15

        # Стан осей
        self._spec_ylim_state = {"min": 0.0, "max": 0.0, "inited": False, "quiet_cnt": 0, "active": False}
        self._iq_ylim_state = {"min": 0.0, "max": 0.0, "inited": False, "quiet_cnt": 0, "active": False}

        self._v_lines_artists = []
        self._h_lines_artists = []
        self._stop = False
        self.render_dt = render_dt

        # Створення фігури
        self.fig, (self.ax_spec, self.ax_i, self.ax_q) = plt.subplots(3, 1, figsize=(12, 10))
        self.fig.canvas.mpl_connect("close_event", lambda e: setattr(self, '_stop', True))
        
        # Ініціалізація графіків
        self._init_plots(freq_grid_step)
        plt.ion()
        self.fig.tight_layout(pad=2.0)
        plt.show(block=False)

    def _init_plots(self, freq_grid_step):
        # Спектр (Bar для сирого, Line для згладженого)
        bar_w = (self.dF * self.freq_scale) * 0.9
        self._bars_raw = self.ax_spec.bar(self.x_freq, np.zeros_like(self.x_freq), width=bar_w, color="orange", alpha=0.6)
        self._line_smooth, = self.ax_spec.plot(self.x_freq, np.zeros_like(self.x_freq), color="purple", lw=1.2, zorder=3)
        self._line_ema, = self.ax_spec.plot(self.x_freq, np.zeros_like(self.x_freq), color="blue", lw=0.8, alpha=0.7)
        
        self.ax_spec.set_ylabel(self._y_label_spec)
        self.ax_spec.grid(True, alpha=0.3)

        if freq_grid_step:
            major = self.dF * self.freq_scale * freq_grid_step
            self.ax_spec.xaxis.set_major_locator(MultipleLocator(major))

        # I/Q панелі
        self._line_i, = self.ax_i.plot([], [], color="navy", lw=1, marker=".", ms=2)
        self._line_q, = self.ax_q.plot([], [], color="darkred", lw=1, marker=".", ms=2)
        self.ax_i.set_title("I (In-phase)")
        self.ax_q.set_title("Q (Quadrature)")
        for ax in [self.ax_i, self.ax_q]: ax.grid(True, alpha=0.3)

    def _apply_relim(self, ax, current_min, current_max, state, hyst_up, hyst_dn, margin):
        """Універсальна логіка авто-масштабування з плавним звуженням"""
        data_span = max(current_max - current_min, 1e-9)
        guard = data_span * self._y_guard_frac
        tgt_min, tgt_max = current_min - guard, current_max + guard

        if not state["inited"]:
            state.update({"min": tgt_min, "max": tgt_max, "inited": True})
        else:
            # 1. Розширення (миттєве)
            expanded = False
            if current_max > state["max"] + hyst_up:
                state["max"] = current_max + guard
                expanded = True
            if current_min < state["min"] - hyst_dn:
                state["min"] = current_min - guard
                expanded = True

            if expanded:
                state["quiet_cnt"] = 0
                state["active"] = False
            else:
                # 2. Детектор тиші
                is_quiet = (current_max < state["max"] - margin) and (current_min > state["min"] + margin)
                if is_quiet:
                    state["quiet_cnt"] += 1
                else:
                    state["quiet_cnt"] = 0
                    state["active"] = False

                # 3. Плавне звуження
                if state["quiet_cnt"] >= self._y_shrink_hold_updates:
                    state["active"] = True
                
                if state["active"]:
                    state["max"] -= (state["max"] - tgt_max) / self._y_shrink_recover_updates
                    state["min"] += (tgt_min - state["min"]) / self._y_shrink_recover_updates
                    if abs(state["max"] - tgt_max) < 0.1: state["active"] = False

        ax.set_ylim(state["min"], state["max"])
        return state["min"]

    def update(self, spectr_arr: np.ndarray, **kwargs) -> None:
        if self._stop: return

        # Спектральні дані
        spec_sliced = spectr_arr[self.idx_l:self.idx_r + 1]
        
        # Оновлення ліній
        if kwargs.get('smoothed') is not None:
            self._line_smooth.set_ydata(kwargs['smoothed'][self.idx_l:self.idx_r + 1])
        if kwargs.get('ema') is not None:
            self._line_ema.set_ydata(kwargs['ema'][self.idx_l:self.idx_r + 1])

        # Релім спектра
        y_bot = self._apply_relim(self.ax_spec, np.min(spec_sliced), np.max(spec_sliced), 
                                  self._spec_ylim_state, self._spec_y_hyst_up, self._spec_y_hyst_dn, 2.0)
        
        for rect, val in zip(self._bars_raw, spec_sliced):
            rect.set_y(y_bot)
            rect.set_height(max(val - y_bot, 0))

        # Вертикальні лінії (v_lines у Hz -> переводимо в масштаб)
        while self._v_lines_artists: self._v_lines_artists.pop().remove()
        v_lines = kwargs.get('v_lines')
        if v_lines is not None:
            for f_hz in v_lines:
                f_scaled = f_hz * self.freq_scale
                if self.x_freq[0] <= f_scaled <= self.x_freq[-1]:
                    self._v_lines_artists.append(self.ax_spec.axvline(f_scaled, color='red', ls='--', alpha=0.6))

        # Горизонтальні лінії
        while self._h_lines_artists: self._h_lines_artists.pop().remove()
        h_lines = kwargs.get('h_lines')
        if h_lines is not None:
            for lvl in h_lines:
                self._h_lines_artists.append(self.ax_spec.axhline(lvl, color='green', ls=':', alpha=0.5))

        # I/Q панелі
        I, Q = kwargs.get('I'), kwargs.get('Q')
        if I is not None and Q is not None:
            self._apply_relim(self.ax_i, min(I.min(), Q.min()), max(I.max(), Q.max()), 
                              self._iq_ylim_state, self._iq_y_hyst_up, self._iq_y_hyst_dn, 0.2)
            self.ax_q.set_ylim(self.ax_i.get_ylim()) # Синхронізація
            
            x_iq = np.arange(len(I)) + kwargs.get('curr_sampl_pos', 0)
            self._line_i.set_data(x_iq, I)
            self._line_q.set_data(x_iq, Q)
            self.ax_i.set_xlim(x_iq[0], x_iq[-1])
            self.ax_q.set_xlim(x_iq[0], x_iq[-1])

        if kwargs.get('title'): self.ax_spec.set_title(f"{self._base_title}\n{kwargs['title']}")
        
        self.fig.canvas.draw_idle()
        plt.pause(self.render_dt)

    @property
    def stop_requested(self) -> bool: return self._stop