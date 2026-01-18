from typing import Final, Literal, Optional, Tuple
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

from fft_core import get_scale_from_units


class VSA_SPECTR_old:
    def __init__(
        self,
        freqs: np.ndarray,
        use_dbfs: bool,
        freq_units: Literal["M", "K"] = "M",
        spec_y_lim: Optional[Tuple[float, float]] = None,
        spec_y_hyst_up: float = 3.0,
        spec_y_hyst_dn: float = 3.0,
        render_dt: float = 0.1,
        title_base: Optional[str] = None,
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

        # Render pacing
        self.render_dt = float(render_dt)

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

        if self._spec_y_lim is not None:
            self.ax_spec.set_ylim(self._spec_y_lim[0], self._spec_y_lim[1])

        self.fig.tight_layout(pad=2)
        self.fig.canvas.draw_idle()
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

    def _update_spec_ylim(self, y_ref: np.ndarray) -> None:
        if self._spec_y_lim is not None:
            return

        y_min = float(np.min(y_ref))
        y_max = float(np.max(y_ref))

        if not self._spec_ylim_inited:
            span = max(y_max - y_min, 1e-12)
            guard = self._y_guard_frac * span
            self._spec_ymin = y_min - guard
            self._spec_ymax = y_max + guard
            self._spec_ylim_inited = True
        else:
            span = max(self._spec_ymax - self._spec_ymin, y_max - y_min, 1e-12)
            guard = self._y_guard_frac * span

            if y_max > self._spec_ymax + self._spec_y_hyst_up:
                self._spec_ymax = y_max + guard
            if y_min < self._spec_ymin - self._spec_y_hyst_dn:
                self._spec_ymin = y_min - guard

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
        plt.pause(self.render_dt)

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
        use_dbfs: bool,
        freq_units: Literal["M", "K"] = "M",
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
        self._iq_y_lim: Optional[Tuple[float, float]] = iq_y_lim

        # Hysteresis for spectrum (independent)
        self._spec_y_hyst_up = float(spec_y_hyst_up)
        self._spec_y_hyst_dn = float(spec_y_hyst_dn)

        # Hysteresis for I/Q (synchronized)
        self._iq_y_hyst_up = float(iq_y_hyst_up)
        self._iq_y_hyst_dn = float(iq_y_hyst_dn)

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
            color="black",
            zorder=3,
        )
        self._line_smooth_spec.set_visible(False)

        self.ax_spec.set_xlabel(f"frequency ({freq_units}Hz)")
        self.ax_spec.set_ylabel(self._y_label_spec)
        self.ax_spec.set_title(self._base_title)
        self.ax_spec.grid(True)

        if self._spec_y_lim is not None:
            self.ax_spec.set_ylim(self._spec_y_lim[0], self._spec_y_lim[1])

        _line_width = 1.0
        _linestyle="-"      # "-", "--", "-.", ":", "solid", "dashed" тощо
        _marker = "o"       # "o", ".", "s", "^", "x" тощо (зміна розміру для дот)
        _markersize=1.75
        _alpha=0.8
        _color="navy"
        # "blue", "red", "green", "cyan", "magenta", "yellow", 
        # "black", "white", "gray", "orange", "purple", "brown",
        # "pink", "lime", "navy", "teal", "olive", "coral"
        # ===== I PANEL =====
        (self._line_i,) = self.ax_i.plot([], [], color=_color,
            linewidth=_line_width, linestyle=_linestyle, alpha=_alpha, marker=_marker, markersize=_markersize,
            zorder=2
        )
        self.ax_i.set_xlabel("sample index")
        self.ax_i.set_ylabel("I (In-phase)")
        self.ax_i.set_title("I (In-phase)")
        self.ax_i.grid(True)

        if self._iq_y_lim is not None:
            self.ax_i.set_ylim(self._iq_y_lim[0], self._iq_y_lim[1])

        # ===== Q PANEL =====
        (self._line_q,) = self.ax_q.plot( [], [], color=_color,
            linewidth=_line_width, linestyle=_linestyle, alpha=_alpha, marker=_marker, markersize=_markersize,
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
        """
        Update spectrum y-limits with hysteresis.
        Returns y_bottom for bar positioning.
        """
        if self._spec_y_lim is not None:
            return float(self._spec_y_lim[0])

        y_min = float(np.min(y_ref))
        y_max = float(np.max(y_ref))

        if not self._spec_ylim_inited:
            span = max(y_max - y_min, 1e-12)
            guard = self._y_guard_frac * span
            self._spec_ymin = y_min - guard
            self._spec_ymax = y_max + guard
            self._spec_ylim_inited = True
        else:
            span = max(self._spec_ymax - self._spec_ymin, y_max - y_min, 1e-12)
            guard = self._y_guard_frac * span

            if y_max > self._spec_ymax + self._spec_y_hyst_up:
                self._spec_ymax = y_max + guard
            if y_min < self._spec_ymin - self._spec_y_hyst_dn:
                self._spec_ymin = y_min - guard

        self.ax_spec.set_ylim(self._spec_ymin, self._spec_ymax)
        return self._spec_ymin

    def _update_iq_ylim(self, y_i: np.ndarray, y_q: np.ndarray) -> float:
        """
        Update I/Q y-limits with shared hysteresis.
        Returns y_bottom for I/Q line positioning.
        """
        if self._iq_y_lim is not None:
            return float(self._iq_y_lim[0])

        y_min = min(float(np.min(y_i)), float(np.min(y_q)))
        y_max = max(float(np.max(y_i)), float(np.max(y_q)))

        if not self._iq_ylim_inited:
            span = max(y_max - y_min, 1e-12)
            guard = self._y_guard_frac * span
            self._iq_ymin = y_min - guard
            self._iq_ymax = y_max + guard
            self._iq_ylim_inited = True
        else:
            span = max(self._iq_ymax - self._iq_ymin, y_max - y_min, 1e-12)
            guard = self._y_guard_frac * span

            if y_max > self._iq_ymax + self._iq_y_hyst_up:
                self._iq_ymax = y_max + guard
            if y_min < self._iq_ymin - self._iq_y_hyst_dn:
                self._iq_ymin = y_min - guard

        self.ax_i.set_ylim(self._iq_ymin, self._iq_ymax)
        self.ax_q.set_ylim(self._iq_ymin, self._iq_ymax)

    def update(
        self,
        spectr_arr: np.ndarray,
        smoothed: Optional[np.ndarray] = None,
        I: Optional[np.ndarray] = None,
        Q: Optional[np.ndarray] = None,
        curr_sampl_pos: int = 0,
        title: Optional[str] = None,
    ) -> None:
        """
        Update all three panels: spectrum, I, Q.

        Contract:
        - spectr_arr: 1D array, len == len(freqs)
        - smoothed: optional 1D array, same length as spectr_arr
        - I, Q: optional 1D arrays, len == len(freqs) (time-domain samples)
        - curr_sampl_pos: current sample position in file (for x-axis scaling)
        - title: optional suffix for main title
        """
        # ===== SPECTRUM PANEL =====
        if smoothed is None:
            self._line_smooth_spec.set_visible(False)
            y_ref = spectr_arr
        else:
            self._line_smooth_spec.set_ydata(smoothed)
            self._line_smooth_spec.set_visible(True)
            y_ref = smoothed

        _title = self._base_title
        if title:
            _title += f"\n{title}"
        self.ax_spec.set_title(_title)

        spec_y_bottom = self._update_spec_ylim(y_ref)

        for rect, yv in zip(self._bars_raw, spectr_arr):
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


