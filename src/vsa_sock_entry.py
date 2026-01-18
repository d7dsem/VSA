
import argparse
import os
from pathlib import Path
import signal
import sys
import traceback
from colorizer import colorize, inject_colors_into

from vsa_socket import  do_vsa_socket
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
        prog="vsa-sock.py",
        description="VSA-sock: spectral analysis & visualization for IQ udp data stream.",
    )

    p.add_argument(
        "-p", "--port",
        dest="port",
        type=int,
        default=0,
        help="UDP stream port",
    )
    
    p.add_argument(
        "--samp-rate",
        dest="samp_rate",
        type=float,
        default=480e3,
        help="Sample rate (Hz).",
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
        "--dtype",
        dest="dtype",
        type=str,
        default="int16",
        help="Expected sample dtype (Default int16)",
    )
    
    p.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        help="Verbose output.",
    )

    return p

def _apply_vsa_sock_contract(args: argparse.Namespace) -> argparse.Namespace:
    """
    Contract enforcement (no guessing):
    - dtype is currently restricted to int16 (per current reader contract)
    """

    if args.dtype != "int16":
        raise RuntimeError(f"unsupported dtype '{args.dtype}' (current contract: int16 only)")

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
        print(f"{WARN} USe dev defaults!")
        port = 9999
        samp_rate = 480e3
        
        args = argparse.Namespace(
            port=port,
            samp_rate=samp_rate,
            center_freq=0,
            fft_n=1024,
            dtype="int16",
            verbose=True
        )
    try:

        args = _apply_vsa_sock_contract(args)
        do_vsa_socket(port=args.port, Fs=args.samp_rate, Fc=args.center_freq)
    except KeyboardInterrupt:
        pass
    except FileNotFoundError as e:
        print(f"{ERR} {str(e)}")
    except Exception as e:
        print(f"{ERR} Unhandled exception: {e}")
        traceback.print_exc()
        show_cli()
        