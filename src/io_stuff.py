from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
import socket
from typing import Annotated, BinaryIO, Final, TypeAlias

import numpy as np


from wav import WAVProps, get_iq_wav_prm, read_wav_header

ArrI16_1D: TypeAlias = Annotated[np.typing.NDArray[np.int16], "float32 1D C-contiguous"]
ArrF32_1D: TypeAlias = Annotated[np.typing.NDArray[np.float32], "float32 1D C-contiguous"]
ArrF32_2D: TypeAlias = Annotated[np.typing.NDArray[np.float32], "float32 2D C-contiguous"]

SOCK_BUF_SZ: Final = 4 * 1024 * 1024

def validate_wav(f_wav: Path, Fs: float, wav_bps: int = 16, n_cnan: int = 2)->bool:
    props : WAVProps = read_wav_header(f_wav)
    if props["codec_tag"] != 0x0001: return False # PCM integer
    if props["channels"] != 2: return False
    if props["sample_rate"] != int(Fs): return False
    if props["bits_per_sample"] != wav_bps: return False
    return True

def create_socket(port: int, rd_timeout_ms: int, sock_buf_sz: int = SOCK_BUF_SZ):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("0.0.0.0", port))
    s.settimeout(rd_timeout_ms / 1000.0)

    s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, sock_buf_sz)
    return s
    
class FReader:
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
                raise RuntimeError(f"unsupported wav '{f_path.absolute()}'")
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

        _FILE_BUFF_SZ = 4 * 1024**2  # practical approach
        self._file: BinaryIO = open(f_path, "rb", buffering=_FILE_BUFF_SZ)
        self._file.seek(self._hdr_sz)
        self.f_path = f_path

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._file:
            self._file.close()

    def read_raw_into(self, arr_int: ArrI16_1D, el_count: int) -> int:
        """
        Reads el_count int16 into 1D raw buffer. Read iq samples require el_count = 2 * samp_count.
        Returns actual number of elements read.

        Side effect:
        - curr_sampl_pos is set to the IQ-sample index at the beginning of this read block.
        """
        # sample index at current file position (begin of block)
        byte_offs = self._file.tell() - self._hdr_sz
        self.curr_sampl_pos = byte_offs // 4

        n_i16_req = el_count
        raw = self._file.read(n_i16_req * 2)
        if not raw:
            return 0

        n_i16_read = len(raw) // 2
        data = np.frombuffer(raw, dtype="<i2", count=n_i16_read)
        arr_int[:n_i16_read] = data
        return n_i16_read

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