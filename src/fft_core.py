# src\fft_core.py
from typing import Annotated, Final, TypeAlias
import numpy as np
from scipy.fft import fft, fftshift


IQInterleavedI16: TypeAlias = Annotated[np.typing.NDArray[np.int16], "int16 LE 1D: i0 q0 i1 q1 ... (interleaved IQ)"]
IQInterleavedF32: TypeAlias = Annotated[np.typing.NDArray[np.float32], "float32 1D: i0 q0 i1 q1 ... (interleaved IQ)"]
ArrC64: TypeAlias = Annotated[np.typing.NDArray[np.complex64],"1D C-contiguous, len == fft_batch*fft_n"]
ArrF32_1D: TypeAlias = Annotated[np.typing.NDArray[np.float32], "float32 1D C-contiguous"]
ArrF32_2D: TypeAlias = Annotated[np.typing.NDArray[np.float32], "float32 2D C-contiguous"]

def i16_to_f32(
        src: IQInterleavedI16,
        dst: IQInterleavedF32,
        n_iq: int,
        normalize: bool = True
    ) -> None:
        """
        Конвертує n_iq IQ-семплів з int16 у float32.
        src: 1D int16  [i0 q0 i1 q1 ...]
        dst: 1D float32[i0 q0 i1 q1 ...]
        """
        n = n_iq * 2  # кількість scalar-елементів

        if normalize:
            dst[:n] = src[:n].astype(np.float32) / 32768.0
        else:
            dst[:n] = src[:n].astype(np.float32)
 
 
# =====================================================
# FFT batch core
# =====================================================

INT16_FULL_SCALE: Final = 32768.0
COMPLEX_SCALING_FACTOR: Final = 2.0
P_FS: Final = COMPLEX_SCALING_FACTOR * (INT16_FULL_SCALE ** 2)

def batch_fft(
    batch_inp: ArrC64,        # (fft_batch*fft_n,), complex64
    fft_n: int,
    fft_batch: int,
    power: ArrF32_1D,         # (fft_n,) OUT
    scratch: ArrF32_2D,       # (fft_batch, fft_n) PREALLOC
    tmp: ArrF32_1D,           # (fft_n,) PREALLOC
    *,
    workers: int = -1,
) -> None:
    """
    Computes averaged power spectrum over fft_batch windows.

    power[k] = (1/fft_batch) * sum_w |FFT_w[k]|^2 / (fft_n^2)
    NOT fftshifted, linear power.

    Output is written in-place to power.
    Use preallocated auxiliary buffers to prevent hot loop alloctions.
    """

    # --- batch FFT ---
    # Semantics (C-equivalent):
    # - fft_batch independent FFTs of length fft_n
    # - |X|^2 computed per window
    # - averaged over windows
    X = fft(batch_inp.reshape(fft_batch, fft_n), axis=1, workers=workers)

    # power = sum_w Re^2
    np.square(X.real, out=scratch)
    np.sum(scratch, axis=0, out=power)

    # tmp = sum_w Im^2 ; power += tmp
    np.square(X.imag, out=scratch)
    np.sum(scratch, axis=0, out=tmp)
    power += tmp

    # mean over windows + FFT scaling
    power *= (1.0 / (fft_batch * fft_n * fft_n))


# --- frequency centering ---
def freq_swap(power: np.ndarray):
    # power: np.ndarray(fft_n,), float32, preallocated)
    power[:] = fftshift(power)


def to_dbfs(power: np.ndarray, p_fs: float = P_FS) -> None:
    """
    Convert linear power to dBFS in-place.
    0 dBFS == full-scale power (p_fs).
    """
    np.maximum(power, 1e-30, out=power)
    np.divide(power, p_fs, out=power)
    np.log10(power, out=power)
    np.multiply(power, 10.0, out=power)


def build_power_spectr(samples_raw: IQInterleavedI16, f32_buf: IQInterleavedF32, x_c: np.ndarray[np.complex64], y_spec: np.ndarray[np.float32], fft_n: int) -> None:
    """Process one frame: FFT for spectrum. Use preallocated buffers"""
    x_c.real[:] = samples_raw[0::2].astype(np.float32)
    x_c.imag[:] = samples_raw[1::2].astype(np.float32)
    
    X = np.fft.fft(x_c, n=fft_n)
    X = np.fft.fftshift(X)
    P = (X.real * X.real + X.imag * X.imag)
    P_safe = np.maximum(P, 1e-15)  # уникаємо log10(0)
    y_spec[:] = 10.0 * np.log10(P_safe / P_FS)

