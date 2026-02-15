import sys, os

import argparse
from pathlib import Path
import signal
import traceback
from typing import Annotated, Final, List, Literal, Optional, Tuple, TypeAlias
from matplotlib import pyplot as plt
import numpy as np

import scipy.signal.windows as windows
from scipy.ndimage import gaussian_filter1d
from scipy.fft import fft as sfft
from scipy.signal import find_peaks, resample
from tqdm import tqdm


from io_stuff import _apply_vsa_file_contract, show_cli, ArrF32_2D, FReader
from wav import read_wav_header, get_iq_wav_prm  # uses your existing module
from colorizer import colorize, inject_colors_into
# --- color names for IDE/static analysis suppress warnings --------------------
GREEN: str; BRIGHT_GREEN: str; RED: str; BRIGHT_RED: str
YELLOW: str; BRIGHT_YELLOW: str; BLACK: str; BRIGHT_BLACK: str
GRAY: str; BRIGHT_GRAY: str; CYAN: str; BRIGHT_CYAN: str
BLUE: str; BRIGHT_BLUE: str; MAGENTA: str; BRIGHT_MAGENTA: str
WHITE: str; BRIGHT_WHITE: str; RESET: str; WARN: str; ERR: str; INFO: str; DBG: str
inject_colors_into(globals())

# =====================================================
# Identity
_MODULE_MARKER = Path(__file__).stem
# ============

INT16_FULL_SCALE: Final = 32768.0
LOAD_LIM: Final = 512*1024**2  # bytes
IQInterleavedF32: TypeAlias = Annotated[np.typing.NDArray[np.float32], "float32 1D: i0 q0 i1 q1 ... (interleaved IQ)"]

def resample_file(data_in: np.ndarray, sr_inp: float, sr_out: float, out_path: Path):
    """Resample full IQ array and write to file."""
    
    if np.isclose(sr_inp, sr_out):
        print(f"{WARN} Sample rates are equal ({sr_inp} Hz). Nothing to resample. Exiting.")
        return
    
    resample_ratio = sr_out / sr_inp
    samples_out_total = int(len(data_in) * resample_ratio)
    
    print(f"Resampling {len(data_in)} samples...")
    data_out = resample(data_in, samples_out_total)
    
    # Convert to int16 interleaved
    print("Converting to int16...")
    data_out_i16 = np.empty(len(data_out) * 2, dtype=np.int16)
    data_out_i16[0::2] = np.clip(data_out.real * INT16_FULL_SCALE, -32768, 32767).astype(np.int16)
    data_out_i16[1::2] = np.clip(data_out.imag * INT16_FULL_SCALE, -32768, 32767).astype(np.int16)
    
    print(f"Writing to {out_path}...")
    with open(out_path, "wb") as f:
        f.write(data_out_i16.tobytes())

# CLI
def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="iq_rsmp.py",
        description="Resempler of IQ recordings (.bin/.wav).",
    )

    p.add_argument(
        "-i", "--input", dest="file", type=Path, required=True,
        help="Path to IQ file: .bin or .wav",
    )

    p.add_argument(
        "--sr-in", dest="samp_rate", type=float, default=None,
        help="Input Sample rate (Hz). Required for .bin; for .wav taken from header (if provided: must match).",
    )
    
    p.add_argument(
        "--sr-out", dest="samp_rate_out", type=float, default=None,
        help="Output Sample rate (Hz).",
    )

    p.add_argument(
        "--samp-offs", dest="samp_offs", type=int, default=0,
        help="Sample offset in IQ samples (I/Q pairs). 1 sample = 4 bytes for int16 IQ.",
    )

    p.add_argument(
        "--dtype", dest="dtype", type=str, default="int16",
        help="Expected sample dtype. For .wav mismatch -> exception. (Current reader: int16 only)",
    )
    p.add_argument(
        "-o", "--out", dest="f_out", type=Path, default=None,
        help="Otput IQ file: ( makes .bin, if not provided creates default with <path>/<f_inp_stem>_rsmp_<sr-out>.bin)",
    )
    return p

def handle_sigint(signum, frame):
    raise KeyboardInterrupt
# ==============================================================================
# === CLI Entry Point
# ==============================================================================
if __name__ == "__main__":
    os.system("")  # Colorizing 'on'
    # Стандартна ініціалізація
    # signal.signal(signal.SIGINT, handle_sigint)

    try:
        # show_cli()
        args: argparse.Namespace = _build_cli().parse_args()
        args = _apply_vsa_file_contract(args)
        
        fp_in = args.file
        fp_out = args.f_out if args.f_out else fp_in.parent / f"{fp_in.stem}_rsmp_{args.samp_rate_out/1e6:.1f}MHz.bin"
        
        # Check if WAV
        is_wav = fp_in.suffix.lower() == '.wav'
        
        if is_wav:
            # Read WAV parameters
            Fs, dur_sec, samp_count, header_sz, dtype_wav, bytes_per_samp = get_iq_wav_prm(fp_in)
            
            # Validate sample rate if provided
            if args.samp_rate is not None and not np.isclose(Fs, args.samp_rate):
                raise ValueError(f"WAV sample rate {Fs} != provided {args.samp_rate}")
            
            file_size = fp_in.stat().st_size
            data_size = file_size - header_sz
        else:
            # BIN file
            if args.samp_rate is None:
                raise ValueError("--sr-in required for .bin files")
            Fs = args.samp_rate
            header_sz = 0
            dtype_wav = np.int16
            file_size = fp_in.stat().st_size
            data_size = file_size
            samp_count = data_size // 4  # int16 I + int16 Q
        
        Fs_out = args.samp_rate_out
        
        print(f"Input file: {YELLOW}{fp_in}{RESET}")
        print(f"File size: {file_size / 1024**2:.1f} MB ({samp_count} IQ samples)")
        print(f"Resampling: {Fs/1e6:.1f} MHz -> {Fs_out/1e6:.1f} MHz")
        
        if data_size > LOAD_LIM:
            raise ValueError(f"File size {data_size/1024**2:.0f} MB exceeds {LOAD_LIM/1024**2:.0f} MB limit. Chunked processing with correct overlap needs verification.")
        
        # Load file
        print("Loading file...")
        with open(fp_in, "rb") as f:
            if is_wav:
                f.seek(header_sz)  # Skip WAV header
            data_raw = np.fromfile(f, dtype=dtype_wav)
        
        # Convert to complex64
        data_iq = np.zeros(len(data_raw) // 2, dtype=np.complex64)
        data_iq.real = data_raw[0::2] / INT16_FULL_SCALE
        data_iq.imag = data_raw[1::2] / INT16_FULL_SCALE
        
        # Resample
        resample_file(data_iq, sr_inp=Fs, sr_out=Fs_out, out_path=fp_out)
        
        print(f"Results saved to: {YELLOW}{fp_out}{RESET}")

    except KeyboardInterrupt:
        print("Ctrl+C pressed. Graceful shutdown...")
        pass
    except FileNotFoundError as e:
        print(f"{ERR} {str(e)}")
    except Exception as e:
        print("-"*32)
        traceback.print_exc()
        print("-"*32)
        print(f"{ERR} Unhandled exception: {GRAY}{e}{RESET}")
        show_cli()