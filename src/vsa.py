from typing import Final, Literal, Optional, Tuple
import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

from fft_core import P_FS, IQInterleavedF32, IQInterleavedI16, i16_to_f32
from io_stuff import FReader


def get_scale_from_units(unit: Literal["M", "K"]) -> float:
    if unit == "K": return 1e-3
    if unit == "M": return 1e-6
    return 1

class VSA_SPECTR:
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

        # SMOOTH
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

        self._update_spec_ylim(y_ref)

        self.fig.canvas.draw_idle()
        plt.pause(self.render_dt)


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




def do_vsa_file(
    fr: FReader,
    Fs: float,
    Fc: float = 0,
    fft_n: int = 1024,
    sigma: float = 1.75,
    start_pos: int = 99_990,
    step_samples: Optional[int] = 1,
    spec_y_lim: Optional[Tuple[float, float]] = (-80, 10),
    iq_y_lim: Optional[Tuple[float, float]] = None,
    render_dt: float = 0.0001,
    use_dbfs: bool = True,
) -> None:
    """
    Process WAV file with tri-panel visualization.
    
    spec_y_lim: hard y-limits for spectrum (optional)
    iq_y_lim: hard y-limits for I/Q signals (optional)
    """
    dF = Fs / fft_n

    i16_buf: IQInterleavedI16 = np.empty(fft_n * 2, dtype=np.int16)
    f32_buf: IQInterleavedF32 = np.empty(fft_n * 2, dtype=np.float32)
    x_c: np.ndarray = np.empty(fft_n, dtype=np.complex64)

    freq_bins = (np.arange(fft_n, dtype=np.float32) - (fft_n // 2)) * dF + Fc

    vsa = VSA(
        freq_bins,
        render_dt=render_dt,
        spec_y_lim=spec_y_lim,
        iq_y_lim=iq_y_lim,
        use_dbfs=use_dbfs
    )
    fr.jump_to_pos(start_pos)
    n_iter = 0

    def _process_frame(on_update=None):
        """Process one frame: FFT for spectrum, extract I/Q for time-domain."""
        i16_to_f32(i16_buf, f32_buf, fft_n, normalize=False)

        # Extract I and Q components
        y_i = i16_buf[0::2].copy()
        y_q = f32_buf[1::2].copy()

        # Compute spectrum
        x_c.real[:] = y_i
        x_c.imag[:] = y_q

        X = np.fft.fft(x_c, n=fft_n)
        X = np.fft.fftshift(X)
        P = (X.real * X.real + X.imag * X.imag)

        if use_dbfs:
            y_spec = 10.0 * np.log10(P / P_FS)
        else:
            y_spec = np.sqrt(P)

        y_spec_smooth = gaussian_filter1d(y_spec, sigma=sigma)

        if on_update:
            on_update(y_spec, y_spec_smooth, y_i, y_q, f" {fr.progress_str()}")

    update_callback = lambda y_spec, y_smooth, y_i, y_q, title: vsa.update(
        y_spec, y_smooth, I=y_i, Q=y_q, curr_sampl_pos=fr.curr_sampl_pos, title=title
    )

    while True:
        if vsa.stop_requested:
            print(f"[MAIN] stop_requested=True: terminating...", flush=True)
            break

        if not vsa.figure_alive():
            print(f"[MAIN] figure closed: terminating...", flush=True)
            break

        if vsa.paused:
            delta = vsa.get_and_clear_step_delta()
            if delta != 0:
                new_pos = fr.curr_sampl_pos + delta * step_samples
                new_pos = max(0, min(new_pos, fr.samples_total - fft_n))
                fr.jump_to_pos(new_pos)

                n_read = fr.read_samples_into(i16_buf, fft_n)
                if n_read == fft_n:
                    _process_frame(on_update=update_callback)

            plt.pause(vsa.render_dt)
            continue

        n_read = fr.read_samples_into(i16_buf, fft_n)
        if n_read != fft_n:
            print(f"read {n_read} not match chunk {fft_n} : terminate...")
            break

        _process_frame(on_update=update_callback)
        n_iter += 1
        if step_samples is not None:
            fr.jump_to_pos(start_pos + step_samples * n_iter)

    print(f"[MAIN] do_vsa_file() exited cleanly", flush=True)


