from typing import Final, Literal, Optional, Tuple
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.ticker import MultipleLocator
from fft_core import get_scale_from_units


class VSA_SPECTR:
    def __init__(
        self,
        freqs: np.ndarray,
        use_dbfs: bool,
        freq_units: Literal["M", "K"] = "M",
        spec_y_lim: Optional[Tuple[float, float]] = None,
        spec_y_hyst_up: float = 3.0,
        spec_y_hyst_dn: float = 3.0,
        title_base: Optional[str] = None,
        # === NEW: shrink control (update-count based) ===
        y_shrink_hold_updates: int = 2,
        y_shrink_recover_updates: int = 4,
        y_shrink_margin_db: float = 2.0,
        # === legacy sink (keeps contract for dead params like render_dt) ===
        **_legacy_kwargs,
    ):
        """
        Spectrum visualization only. (raw + optional smoothed)
        Render as Line2D (faster than bars).
        """
        self.freq_units = freq_units
        self.freq_scale: float = get_scale_from_units(freq_units)

        self.freqs = freqs
        self.x_freq = self.freqs * self.freq_scale

        self.dF = float(self.freqs[1] - self.freqs[0])
        self._y_label_spec = "power (dBFS)" if use_dbfs else "magn (linear)"
        self.use_dbfs = use_dbfs

        self._base_title = title_base if title_base else (
            f"SPECTRUM  FFT={len(freqs)}  |  dF={self.dF * self.freq_scale:.3f} {freq_units}Hz"
        )

        # Hard y-limits
        self._spec_y_lim: Optional[Tuple[float, float]] = spec_y_lim

        # Hysteresis for spectrum
        self._spec_y_hyst_up = float(spec_y_hyst_up)
        self._spec_y_hyst_dn = float(spec_y_hyst_dn)

        # Guard fraction for y-limit updates
        self._y_guard_frac = 0.10

        # === NEW: shrink state/params (update-count based) ===
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

        # Figure + subplot
        self.fig, self.ax_spec = plt.subplots(figsize=(12, 6))
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("close_event", self._on_figure_close)

        # ===== SPECTRUM PANEL (Line2D) =====
        (self._line_raw_spec,) = self.ax_spec.plot(
            self.x_freq,
            np.zeros_like(self.x_freq),
            linestyle="-",
            marker=".",
            markersize=2.0,
            linewidth=1.0,
            color="orange",
            zorder=2,
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

        self.ax_spec.set_xlabel(f"frequency ({freq_units}Hz)")
        self.ax_spec.set_ylabel(self._y_label_spec)
        self.ax_spec.set_title(self._base_title)
        self.ax_spec.grid(True)
        # # Частотна сітка
        # self._freq_grid_step = freq_grid_step
        # if self._freq_grid_step is not None and self._freq_grid_step > 0:
        #     major_step = self.dF * self.freq_scale * self._freq_grid_step
        #     minor_step = self.dF * self.freq_scale
        #     self.ax_spec.xaxis.set_major_locator(MultipleLocator(major_step))
        #     self.ax_spec.xaxis.set_minor_locator(MultipleLocator(minor_step))
        #     self.ax_spec.grid(True, which='major', linestyle='-', linewidth=0.7, alpha=0.5)
        #     self.ax_spec.grid(True, which='minor', linestyle=':', linewidth=0.3, alpha=0.3)
        
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

    def _on_key(self, ev) -> None:
        print(f"[_ON_KEY] key={ev.key!r}", flush=True)
        if ev.key == "escape":
            self._stop = True
            print(f"[_ON_KEY] pressed ESC (closing figure requested)", flush=True)
            plt.close(self.fig)
            self.fig = None

    @property
    def stop_requested(self) -> bool:
        return self._stop

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
    ) -> None:
        # RAW
        self._line_raw_spec.set_ydata(spectr_arr)
        y_ref = spectr_arr

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

        self._update_spec_ylim(y_ref)

        self.fig.canvas.draw_idle()
        # keep UI responsive; external loop controls pacing
        plt.pause(0.001)


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
        iq_y_hyst_up: float = 3.0,
        iq_y_hyst_dn: float = 3.0,
        render_dt: float = 0.1,
        title_base: Optional[str] = None
    ):
        """
        Tri-panel visualization: spectrum, I signal, Q signal (time-domain).
        
        Spectrum has independent hysteresis.
        I and Q share synchronized hysteresis.
        
        freq_show_idx: (idx_l, idx_r) to restrict displayed frequency range
        """
        self.freq_units = freq_units
        self.freq_scale: float = get_scale_from_units(freq_units)

        # Full frequency array
        self.freqs_full = freqs
        self.dF = float(self.freqs_full[1] - self.freqs_full[0])
        
        # ===== LINES STATE =====
        self._v_lines_artists = []  # Список для зберігання об'єктів vlines
        self._h_lines_artists = []  # Для майбутніх горизонтальних ліній
        
        # Determine display range indices
        if freq_show_idx is not None:
            self.idx_l, self.idx_r = freq_show_idx
            self.idx_l = max(0, min(self.idx_l, len(self.freqs_full) - 1))
            self.idx_r = max(self.idx_l, min(self.idx_r, len(self.freqs_full) - 1))
        else:
            self.idx_l = 0
            self.idx_r = len(self.freqs_full) - 1
        
        # Displayed frequency slice
        self.freqs = self.freqs_full[self.idx_l:self.idx_r + 1]
        self.x_freq = self.freqs * self.freq_scale
        
        self._y_label_spec = "power (dBFS)" if use_dbfs else "magn (linear)"
        self.use_dbfs = use_dbfs
        
        self._base_title = title_base if title_base else (
            f"SPECTRUM  FFT={len(self.freqs_full)}  |  dF={self.dF * self.freq_scale:.3f} {freq_units}Hz"
        )

        # Hard y-limits
        self._spec_y_lim: Optional[Tuple[float, float]] = spec_y_lim
        self._iq_y_lim: Optional[Tuple[float, float]] = iq_y_lim

        # Hysteresis for spectrum (independent)
        self._spec_y_hyst_up = float(spec_y_hyst_up)
        self._spec_y_hyst_dn = float(spec_y_hyst_dn)

        # Hysteresis for I/Q (synchronized)
        self._iq_y_hyst_up = float(iq_y_hyst_up)
        self._iq_y_hyst_dn = float(iq_y_hyst_dn)
        
        self._y_shrink_hold_updates = 2      # Скільки кадрів чекаємо перед початком звуження
        self._y_shrink_recover_updates = 4   # За скільки кадрів плавно звужуємося
        self._y_shrink_margin_db = 2.0       # Запас (dB), нижче якого вважаємо сигнал "тихим"

        self._y_quiet_updates = 0
        self._y_shrink_active = False
        self._y_shrink_step_ymax = 0.0
        self._y_shrink_step_ymin = 0.0
        
        
        # Додайте ці поля до існуючих для спектра
        self._iq_quiet_updates = 0
        self._iq_shrink_active = False
        self._iq_shrink_step_ymax = 0.0
        self._iq_shrink_step_ymin = 0.0
        # Маржа для I/Q (наприклад, 20% від поточного діапазону)
        self._iq_shrink_margin = 0.2

        # Guard fraction for y-limit updates
        self._y_guard_frac = 0.10

        # Render pacing
        self.render_dt = float(render_dt)

        # Spectrum y-scale state
        self._spec_ylim_inited = False
        self._spec_ymin = 0.0
        self._spec_ymax = 0.0

        # I/Q y-scale state (shared)
        self._iq_ylim_inited = False
        self._iq_ymin = 0.0
        self._iq_ymax = 0.0

        # Control
        self._stop = False
        self.paused: bool = False
        self._step_delta: int = 0

        # Figure + subplots (3 rows, 1 column)
        self.fig, (self.ax_spec, self.ax_i, self.ax_q) = plt.subplots(3, 1, figsize=(12, 10))
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("close_event", self._on_figure_close)

        # ===== SPECTRUM PANEL =====
        bar_w = (self.dF * self.freq_scale) * 0.95
        self._bars_raw = self.ax_spec.bar(
            self.x_freq,
            np.zeros_like(self.x_freq),
            width=bar_w,
            bottom=0.0,
            align="center",
            color="orange",
            linewidth=0.0,
            zorder=1,
        )

        (self._line_smooth_spec,) = self.ax_spec.plot(
            self.x_freq,
            np.zeros_like(self.x_freq),
            color="purple",
            linewidth=0.75,
            zorder=3,
        )
        self._line_smooth_spec.set_visible(False)

        (self._line_ema_spec,) = self.ax_spec.plot(
            self.x_freq,
            np.zeros_like(self.x_freq),
            color="blue",
            linewidth=0.75,
            zorder=3,
        )
        
        self.ax_spec.set_xlabel(f"frequency ({freq_units}Hz)")
        self.ax_spec.set_ylabel(self._y_label_spec)
        self.ax_spec.set_title(self._base_title)
        self.ax_spec.grid(True)
        
        # Frequency grid
        self._freq_grid_step = freq_grid_step
        if self._freq_grid_step is not None and self._freq_grid_step > 0:
            major_step = self.dF * self.freq_scale * self._freq_grid_step
            minor_step = self.dF * self.freq_scale
            self.ax_spec.xaxis.set_major_locator(MultipleLocator(major_step))
            self.ax_spec.xaxis.set_minor_locator(MultipleLocator(minor_step))
            self.ax_spec.grid(True, which='major', linestyle='-', linewidth=0.7, alpha=0.5)
            self.ax_spec.grid(True, which='minor', linestyle=':', linewidth=0.3, alpha=0.3)
        
        if self._spec_y_lim is not None:
            self.ax_spec.set_ylim(self._spec_y_lim[0], self._spec_y_lim[1])

        _line_width = 1.0
        _linestyle = "-"
        _marker = "o"
        _markersize = 1.75
        _alpha = 0.8
        _color = "navy"
        
        # ===== I PANEL =====
        (self._line_i,) = self.ax_i.plot([], [], color=_color,
            linewidth=_line_width, linestyle=_linestyle, alpha=_alpha, 
            marker=_marker, markersize=_markersize,
            zorder=2
        )
        self.ax_i.set_xlabel("sample index")
        self.ax_i.set_ylabel("I (In-phase)")
        self.ax_i.set_title("I (In-phase)")
        self.ax_i.grid(True)

        if self._iq_y_lim is not None:
            self.ax_i.set_ylim(self._iq_y_lim[0], self._iq_y_lim[1])

        # ===== Q PANEL =====
        (self._line_q,) = self.ax_q.plot([], [], color=_color,
            linewidth=_line_width, linestyle=_linestyle, alpha=_alpha, 
            marker=_marker, markersize=_markersize,
            zorder=2
        )
        self.ax_q.set_xlabel("sample index")
        self.ax_q.set_ylabel("Q (Quadrature)")
        self.ax_q.set_title("Q (Quadrature)")
        self.ax_q.grid(True)

        if self._iq_y_lim is not None:
            self.ax_q.set_ylim(self._iq_y_lim[0], self._iq_y_lim[1])

        self.fig.tight_layout(pad=2.0)
        self.fig.canvas.draw_idle()
        plt.pause(self.render_dt)

    def _shrink_reset(self) -> None:
        self._y_quiet_updates = 0
        self._y_shrink_active = False

    def _shrink_arm(self, tgt_ymin: float, tgt_ymax: float) -> None:
        # Розраховуємо крок для плавного переходу за N кадрів
        dy_max = max(self._spec_ymax - tgt_ymax, 0.0)
        dy_min = max(tgt_ymin - self._spec_ymin, 0.0)
        self._y_shrink_step_ymax = dy_max / self._y_shrink_recover_updates
        self._y_shrink_step_ymin = dy_min / self._y_shrink_recover_updates
        self._y_shrink_active = True
    
    def _on_figure_close(self, ev) -> None:
        """Handle window close by user (X button)"""
        print(f"[_ON_FIGURE_CLOSE] Figure closed by user", flush=True)
        self._stop = True

    def _on_key(self, ev) -> None:
        print(f"[_ON_KEY] key={ev.key!r}", flush=True)
        if ev.key == "escape":
            self._stop = True
            print(f"[_ON_KEY] closing figure via ESC", flush=True)
            plt.close(self.fig)
        elif ev.key in (" ", "space"):
            self.paused = not self.paused
            print(f"[_ON_KEY] paused toggled to {self.paused}", flush=True)
        elif ev.key == "right":
            if self.paused:
                self._step_delta += 1
                print(f"[_ON_KEY] step_delta incremented to {self._step_delta}", flush=True)
        elif ev.key == "left":
            if self.paused:
                self._step_delta -= 1
                print(f"[_ON_KEY] step_delta decremented to {self._step_delta}", flush=True)

    def get_and_clear_step_delta(self) -> int:
        """Returns accumulated step delta and clears it."""
        delta = self._step_delta
        self._step_delta = 0
        return delta

    @property
    def stop_requested(self) -> bool:
        return self._stop

    def figure_alive(self) -> bool:
        """Check if figure window still exists"""
        try:
            return plt.fignum_exists(self.fig.number)
        except Exception:
            return False

    def _update_spec_ylim(self, y_ref: np.ndarray) -> float:
        if self._spec_y_lim is not None:
            return float(self._spec_y_lim[0])

        y_min = float(np.min(y_ref))
        y_max = float(np.max(y_ref))
        
        # Цільове вікно з невеликим запасом (guard)
        data_span = max(y_max - y_min, 1e-12)
        guard = self._y_guard_frac * data_span
        tgt_ymin = y_min - guard
        tgt_ymax = y_max + guard

        if not self._spec_ylim_inited:
            self._spec_ymin, self._spec_ymax = tgt_ymin, tgt_ymax
            self._spec_ylim_inited = True
            self._shrink_reset()
        else:
            # 1. ПЕРЕВІРКА НА РОЗШИРЕННЯ (Миттєво)
            expanded = False
            if y_max > self._spec_ymax + self._spec_y_hyst_up:
                self._spec_ymax = y_max + guard
                expanded = True
            if y_min < self._spec_ymin - self._spec_y_hyst_dn:
                self._spec_ymin = y_min - guard
                expanded = True

            if expanded:
                self._shrink_reset() # Скидаємо таймер звуження, якщо розширились
            else:
                # 2. ПЕРЕВІРКА НА ЗВУЖЕННЯ (З затримкою)
                # Сигнал став значно меншим за поточні межі?
                is_quiet = (y_max < self._spec_ymax - self._y_shrink_margin_db) and \
                        (y_min > self._spec_ymin + self._y_shrink_margin_db)
                
                if is_quiet:
                    self._y_quiet_updates += 1
                else:
                    self._shrink_reset()

                # Якщо сигнал "тихий" достатньо довго — вмикаємо shrink
                if (not self._y_shrink_active) and (self._y_quiet_updates >= self._y_shrink_hold_updates):
                    self._shrink_arm(tgt_ymin, tgt_ymax)

                # 3. ПРОЦЕС ПЛАВНОГО ЗВУЖЕННЯ
                if self._y_shrink_active:
                    if self._spec_ymax > tgt_ymax:
                        self._spec_ymax = max(tgt_ymax, self._spec_ymax - self._y_shrink_step_ymax)
                    if self._spec_ymin < tgt_ymin:
                        self._spec_ymin = min(tgt_ymin, self._spec_ymin + self._y_shrink_step_ymin)
                    
                    # Завершення процесу
                    if (self._spec_ymax <= tgt_ymax) and (self._spec_ymin >= tgt_ymin):
                        self._shrink_reset()

        self.ax_spec.set_ylim(self._spec_ymin, self._spec_ymax)
        return self._spec_ymin

    # def _update_spec_ylim(self, y_ref: np.ndarray) -> float:
    #     """
    #     Update spectrum y-limits with hysteresis.
    #     Returns y_bottom for bar positioning.
    #     """
    #     if self._spec_y_lim is not None:
    #         return float(self._spec_y_lim[0])

    #     y_min = float(np.min(y_ref))
    #     y_max = float(np.max(y_ref))
    #     if y_min ==  y_max:
    #         return 0.0
    #     if not self._spec_ylim_inited:
    #         span = max(y_max - y_min, 1e-12)
    #         guard = self._y_guard_frac * span
    #         self._spec_ymin = y_min - guard
    #         self._spec_ymax = y_max + guard
    #         self._spec_ylim_inited = True
    #     else:
    #         span = max(self._spec_ymax - self._spec_ymin, y_max - y_min, 1e-12)
    #         guard = self._y_guard_frac * span

    #         if y_max > self._spec_ymax + self._spec_y_hyst_up:
    #             self._spec_ymax = y_max + guard
    #         if y_min < self._spec_ymin - self._spec_y_hyst_dn:
    #             self._spec_ymin = y_min - guard

    #     self.ax_spec.set_ylim(self._spec_ymin, self._spec_ymax)
    #     return self._spec_ymin

    # def _update_iq_ylim(self, y_i: np.ndarray, y_q: np.ndarray) -> float:
    #     """
    #     Update I/Q y-limits with shared hysteresis.
    #     Returns y_bottom for I/Q line positioning.
    #     """
    #     if self._iq_y_lim is not None:
    #         return float(self._iq_y_lim[0])

    #     y_min = min(float(np.min(y_i)), float(np.min(y_q)))
    #     y_max = max(float(np.max(y_i)), float(np.max(y_q)))

    #     if not self._iq_ylim_inited:
    #         span = max(y_max - y_min, 1e-12)
    #         guard = self._y_guard_frac * span
    #         self._iq_ymin = y_min - guard
    #         self._iq_ymax = y_max + guard
    #         self._iq_ylim_inited = True
    #     else:
    #         span = max(self._iq_ymax - self._iq_ymin, y_max - y_min, 1e-12)
    #         guard = self._y_guard_frac * span

    #         if y_max > self._iq_ymax + self._iq_y_hyst_up:
    #             self._iq_ymax = y_max + guard
    #         if y_min < self._iq_ymin - self._iq_y_hyst_dn:
    #             self._iq_ymin = y_min - guard

    #     self.ax_i.set_ylim(self._iq_ymin, self._iq_ymax)
    #     self.ax_q.set_ylim(self._iq_ymin, self._iq_ymax)

    def _update_iq_ylim(self, y_i: np.ndarray, y_q: np.ndarray) -> None:
        if self._iq_y_lim is not None:
            return

        y_min = min(float(np.min(y_i)), float(np.min(y_q)))
        y_max = max(float(np.max(y_i)), float(np.max(y_q)))

        data_span = max(y_max - y_min, 1e-12)
        guard = self._y_guard_frac * data_span
        tgt_ymin, tgt_ymax = y_min - guard, y_max + guard

        if not self._iq_ylim_inited:
            self._iq_ymin, self._iq_ymax = tgt_ymin, tgt_ymax
            self._iq_ylim_inited = True
        else:
            # 1. Розширення (миттєве)
            expanded = False
            if y_max > self._iq_ymax + self._iq_y_hyst_up:
                self._iq_ymax = y_max + guard
                expanded = True
            if y_min < self._iq_ymin - self._iq_y_hyst_dn:
                self._iq_ymin = y_min - guard
                expanded = True

            if expanded:
                self._iq_quiet_updates = 0
                self._iq_shrink_active = False
            else:
                # 2. Звуження (з затримкою)
                # Для I/Q перевіряємо, чи сигнал став суттєво меншим (наприклад, менше 60% вікна)
                current_window = self._iq_ymax - self._iq_ymin
                is_quiet = (y_max < self._iq_ymax - 0.2 * current_window) and \
                        (y_min > self._iq_ymin + 0.2 * current_window)

                if is_quiet:
                    self._iq_quiet_updates += 1
                else:
                    self._iq_quiet_updates = 0
                    self._iq_shrink_active = False

                if (not self._iq_shrink_active) and (self._iq_quiet_updates >= self._y_shrink_hold_updates):
                    # Arm iq shrink
                    dy_max = max(self._iq_ymax - tgt_ymax, 0.0)
                    dy_min = max(tgt_ymin - self._iq_ymin, 0.0)
                    self._iq_shrink_step_ymax = dy_max / self._y_shrink_recover_updates
                    self._iq_shrink_step_ymin = dy_min / self._y_shrink_recover_updates
                    self._iq_shrink_active = True

                if self._iq_shrink_active:
                    self._iq_ymax = max(tgt_ymax, self._iq_ymax - self._iq_shrink_step_ymax)
                    self._iq_ymin = min(tgt_ymin, self._iq_ymin + self._iq_shrink_step_ymin)
                    if (self._iq_ymax <= tgt_ymax) and (self._iq_ymin >= tgt_ymin):
                        self._iq_shrink_active = False

        self.ax_i.set_ylim(self._iq_ymin, self._iq_ymax)
        self.ax_q.set_ylim(self._iq_ymin, self._iq_ymax)

    def update( self, spectr_arr: np.ndarray, **kwargs) -> None: 
        """
        Update all three panels: spectrum, I, Q.

        Contract:
        - spectr_arr: 1D array, len == len(freqs_full)
        - smoothed: optional 1D array, same length as spectr_arr
        - ema: optional 1D array, same length as spectr_arr
        - I, Q: optional 1D arrays, len == len(freqs_full) (time-domain samples)
        - curr_sampl_pos: current sample position in file (for x-axis scaling)
        - title: optional suffix for main title
        """
        smoothed: Optional[np.ndarray] = kwargs.get('smoothed')
        ema: Optional[np.ndarray] = kwargs.get('ema')
        I: Optional[np.ndarray] = kwargs.get('I')
        Q: Optional[np.ndarray] = kwargs.get('Q')
        curr_sampl_pos: int = kwargs.get('curr_sampl_pos', 0)
        title: Optional[str] = kwargs.get('title', None)
        v_lines: Optional[np.ndarray] = kwargs.get('v_lines')
        h_lines: Optional[np.ndarray] = kwargs.get('h_lines')
        assert isinstance(spectr_arr, np.ndarray), "spectr_arr - np.ndarray[float]"
        
        # Slice spectrum data to display range
        spec_sliced = spectr_arr[self.idx_l:self.idx_r + 1]
        smooth_sliced = smoothed[self.idx_l:self.idx_r + 1] if smoothed is not None else None
        ema_sliced = ema[self.idx_l:self.idx_r + 1] if ema is not None else None

        # ===== SPECTRUM PANEL =====
        if smooth_sliced is None:
            self._line_smooth_spec.set_visible(False)
        else:
            self._line_smooth_spec.set_ydata(smooth_sliced)
            self._line_smooth_spec.set_visible(True)
        
        if ema_sliced is None:
            self._line_ema_spec.set_visible(False)
        else:
            self._line_ema_spec.set_ydata(ema_sliced)
            self._line_ema_spec.set_visible(True)
        
        y_ref = spec_sliced
        _title = self._base_title
        if title:
            _title += f"\n{title}"
        self.ax_spec.set_title(_title)

        # ===== ОБРОБКА ВЕРТИКАЛЬНИХ ЛІНІЙ (v_lines) =====
        # Очищуємо попередні лінії (щоб не нашаровувались)
        while self._v_lines_artists:
            art = self._v_lines_artists.pop()
            art.remove()

        if v_lines is not None:
            v_lines_scaled = np.asanyarray(v_lines) * self.freq_scale
            # Використовуємо x_freq (масштабовані частоти) для перевірки меж
            for freq in v_lines_scaled:
                # Малюємо лінію тільки якщо вона в межах поточного перегляду (idx_l:idx_r)
                if self.x_freq[0] <= freq <= self.x_freq[-1]:
                    l_art = self.ax_spec.axvline(
                        x=freq, color='green', linestyle='--', 
                        linewidth=0.8, alpha=0.99
                    )
                    self._v_lines_artists.append(l_art)

        # ===== ОБРОБКА ГОРИЗОНТАЛЬНИХ ЛІНІЙ (h_lines) =====
        while self._h_lines_artists:
            art = self._h_lines_artists.pop()
            art.remove()

        if h_lines is not None:
            for level in h_lines:
                l_art = self.ax_spec.axhline(
                    y=level, color='green', linestyle=':', 
                    linewidth=0.8, alpha=0.5, zorder=0
                )
                self._h_lines_artists.append(l_art)
                
        spec_y_bottom = self._update_spec_ylim(y_ref)

        for rect, yv in zip(self._bars_raw, spec_sliced):
            rect.set_y(spec_y_bottom)
            rect.set_height(float(yv) - spec_y_bottom)

        # ===== I/Q PANELS =====
        if I is not None and Q is not None:
            y_i = I.astype(np.float32)
            y_q = Q.astype(np.float32)

            self._update_iq_ylim(y_i, y_q)

            # X-axis: absolute sample position in file
            x_start = curr_sampl_pos
            x_end = curr_sampl_pos + len(y_i)
            x_iq = np.linspace(x_start, x_end - 1, len(y_i))

            self._line_i.set_data(x_iq, y_i)
            self._line_q.set_data(x_iq, y_q)

            # Set x-limits based on file position
            self.ax_i.set_xlim(x_start, x_end - 1)
            self.ax_q.set_xlim(x_start, x_end - 1)

        self.fig.canvas.draw_idle()
        plt.pause(self.render_dt)

