
""" 
io_stuff.py
Low level stuf for hsndle input data of iq int16 interlived in raw semantic
"""
from argparse import Namespace
import argparse
from pathlib import Path
import socket
import sys
from typing import Annotated, BinaryIO, Final, Optional, TypeAlias
import numpy as np
from wav import WAVProps, get_iq_wav_prm, read_wav_header

ArrI16_1D: TypeAlias = Annotated[np.typing.NDArray[np.int16], "int16 1D C-contiguous"]
ArrF32_1D: TypeAlias = Annotated[np.typing.NDArray[np.float32], "float32 1D C-contiguous"]
ArrF32_2D: TypeAlias = Annotated[np.typing.NDArray[np.float32], "float32 2D C-contiguous"]

_IO_BUF_SZ: Final = 4 * 1024 * 1024 # practical approach


def show_cli():
    print(f"\n[DBG] CLI:" + " ".join(map(str, sys.argv)))


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

        props = read_wav_header(f)
        wav_fs = float(props["sample_rate"])
        if args.samp_rate is not None and float(args.samp_rate) != wav_fs:
            raise RuntimeError(f"--samp-rate mismatch: cli={args.samp_rate} wav={wav_fs}")
        args.samp_rate = wav_fs

    return args

def validate_wav(f_wav: Path, Fs: float, wav_bps: int = 16, n_cnan: int = 2)->bool:
    props : WAVProps = read_wav_header(f_wav)
    if props["codec_tag"] != 0x0001: return False # PCM integer
    if props["channels"] != 2: return False
    if props["sample_rate"] != int(Fs): return False
    if props["bits_per_sample"] != wav_bps: return False
    return True


def create_socket(port: int, rd_timeout_ms: int, sock_buf_sz: int = _IO_BUF_SZ):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("0.0.0.0", port))
    s.settimeout(rd_timeout_ms / 1000.0)

    s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, sock_buf_sz)
    return s


class FReader:
    """
    For IQ int16 interlibed LE raw or wav.
    """
    def __init__(self, args: Namespace):
        f_path: Path = args.file
        if not f_path.exists():
            raise RuntimeError(f"file not exists '{f_path.absolute()}'")
        if f_path.suffix not in (".bin", ".wav"):
            raise RuntimeError(f"unsupported file type'{f_path.absolute()}'")

        self._hdr_sz: int = 0
        self.samples_total: int = 0
        self.Fs: float = args.samp_rate
        self.dur_sec: float = 0

        if f_path.suffix == ".wav":
            if not validate_wav(f_path, args.samp_rate):
                raise RuntimeError(f"unsupported wav '{f_path.absolute()}' (check is it 2 chan int16)")
            self.Fs, self.dur_sec, self.samples_total, self._hdr_sz, _, _ = get_iq_wav_prm(f_path)
        else:
            # .bin: raw int16 IQ interleaved => 4 bytes per IQ sample
            f_sz = f_path.stat().st_size
            if (f_sz % 4) != 0:
                raise RuntimeError(f"corrupted bin size (not multiple of 4): '{f_path.absolute()}'")
            self.samples_total = f_sz // 4
            self.dur_sec = self.samples_total / self.Fs

        # current sample position = sample index at the beginning of the last read block - use DSP domain semantic - IQ int16 pair. Meta info, not for navigation!
        self.curr_sampl_pos: int = 0

        self._file: BinaryIO = open(f_path, "rb", buffering=_IO_BUF_SZ)
        self._file.seek(self._hdr_sz)
        self.f_path = f_path
        
        self._raw_i16_buf: Optional[ArrI16_1D] = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._file:
            self._file.close()

    def read_raw_into(self, arr_int: ArrI16_1D, el_count: int) -> int:
        """
        Reads el_count int16 into prealocated 1D raw buffer. 
        Returns actual number of elements read.
        Side effect:
        - curr_sampl_pos is set to the IQ-sample index at the beginning of this read block.
        Assume sample is IQ pair.
        """
        # sample index at current file position (begin of block)
        byte_offs = self._file.tell() - self._hdr_sz
        self.curr_sampl_pos = byte_offs // 4
        n_bytes_read = self._file.readinto(memoryview(arr_int[:el_count]))
        if not n_bytes_read: # n_bytes_read може бути None або 0
            return 0
        return n_bytes_read // 2

    def read_samples_into(self, arr: ArrF32_2D, samp_count:int) -> int:
        required_count = samp_count * 2
        if self._raw_i16_buf is None or self._raw_i16_buf.size < required_count:
            self._raw_i16_buf = np.empty(required_count, dtype=np.int16)
        n_i16_read = self.read_raw_into(self._raw_i16_buf, required_count)
        n_samp_read = n_i16_read // 2
        arr.flat[:n_i16_read] = self._raw_i16_buf[:n_i16_read].astype(np.float32)
        return n_samp_read

    def jump_to_samp_pos(self, sample_pos: int) -> None:
        """
        Jump to absolute position in IQ samples (I/Q pairs).
        (In domain semantic)
        """
        if sample_pos < 0 or sample_pos >= self.samples_total:
            raise RuntimeError(
                f"sample_pos {sample_pos} out of range (total {self.samples_total})"
            )

        byte_pos = self._hdr_sz + sample_pos * 4
        self._file.seek(byte_pos, 0)

        # next read block will start here
        self.curr_sampl_pos = sample_pos

    def progress_str(self) -> str:
        """
        Returns progress string:
        current IQ-sample index and percentage of total.
        """
        if self.samples_total <= 0:
            return f"sample {self.curr_sampl_pos} / ?"

        pct = 100.0 * self.curr_sampl_pos / self.samples_total
        return f"sample {self.curr_sampl_pos:_} / {self.samples_total:_} ({pct:.2f}%)"