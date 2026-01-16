
import argparse
import os
from pathlib import Path
import signal
import sys
import traceback
from colorizer import colorize, inject_colors_into
from file_stuff import FReader
from vsa import do_vsa_file
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



# Helpers

def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vsa-file.py",
        description="VSA-file: spectral analysis & visualization for IQ recordings (.bin/.wav).",
    )

    p.add_argument(
        "-i", "--input",
        dest="file",
        type=Path,
        required=True,
        help="Path to IQ file: .bin or .wav",
    )

    p.add_argument(
        "--samp-rate",
        dest="samp_rate",
        type=float,
        default=None,
        help="Sample rate (Hz). Required for .bin; for .wav taken from header (if provided: must match).",
    )

    p.add_argument(
        "--samp-offs",
        dest="samp_offs",
        type=int,
        default=0,
        help="Sample offset in IQ samples (I/Q pairs). 1 sample = 4 bytes for int16 IQ.",
    )

    p.add_argument(
        "--dtype",
        dest="dtype",
        type=str,
        default="int16",
        help="Expected sample dtype. For .wav mismatch -> exception. (Current reader: int16 only)",
    )

    p.add_argument(
        "--center-freq",
        dest="center_freq",
        type=float,
        default=0.0,
        help="Center frequency (Hz). Default: 0",
    )

    p.add_argument(
        "--fft-n",
        dest="fft_n",
        type=int,
        default=4096,
        help="FFT size. Default: 4096",
    )

    p.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        help="Verbose output.",
    )

    return p


def _apply_vsa_file_contract(args: argparse.Namespace) -> argparse.Namespace:
    """
    Contract enforcement (no guessing):
    - file must be .bin or .wav
    - .bin requires samp_rate
    - .wav uses Fs from header; if args.samp_rate provided must match
    - dtype is currently restricted to int16 (per current reader contract)
    - samp_offs is in IQ samples (pairs): byte offset = samp_offs * 4
    """
    f: Path = args.file
    if f.suffix.lower() not in (".bin", ".wav"):
        raise RuntimeError(f"unsupported file type '{f.absolute()}'")

    if args.dtype != "int16":
        raise RuntimeError(f"unsupported dtype '{args.dtype}' (current contract: int16 only)")

    if f.suffix.lower() == ".bin":
        if args.samp_rate is None:
            raise RuntimeError("--samp-rate is required for .bin input")
    else:
        # wav: Fs from header; validate if user also provided samp_rate
        from wav import read_wav_header  # uses your existing module
        props = read_wav_header(f)
        wav_fs = float(props["sample_rate"])
        if args.samp_rate is not None and float(args.samp_rate) != wav_fs:
            raise RuntimeError(f"--samp-rate mismatch: cli={args.samp_rate} wav={wav_fs}")
        args.samp_rate = wav_fs

    return args
 
def show_cli():
    print(f"\n{DBG} CLI: {GRAY}" + " ".join(map(str, sys.argv)) + RESET)
    
def handle_sigint(signum, frame):
    print("Ctrl+C pressed. Graceful shutdown...")
    raise KeyboardInterrupt
# ==============================================================================
# === CLI Entry Point
# ==============================================================================
if __name__ == "__main__":
    os.system("")  # Colorizing 'on'
    # Стандартна ініціалізація
    signal.signal(signal.SIGINT, handle_sigint)
    
    args: argparse.Namespace = None
    if len(sys.argv) > 1:
        # show_cli()
        args = _build_cli().parse_args()
    else:
        # _fpath = r""
        # samp_rate
        
        _fpath = r"e:\home\d7\Public\signals\OFDM\5MHz-11-14-25.bin"
        samp_rate = 1/1.00e-7
        
        args = argparse.Namespace(
            file=Path(_fpath),
            samp_offset=0,
            samp_rate=samp_rate,
            center_freq=5e6,
            fft_n=1024,
            dtype="int16",
            verbose=True
        )
    try:

        args = _apply_vsa_file_contract(args)
        with FReader(args) as fr:
            do_vsa_file(fr, Fs=args.samp_rate)
    except KeyboardInterrupt:
        pass
    except FileNotFoundError as e:
        print(f"{ERR} {str(e)}")
    except Exception as e:
        print(f"{ERR} Unhandled exception: {e}")
        traceback.print_exc()
        show_cli()
        