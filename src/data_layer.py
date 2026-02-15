from dataclasses import astuple, dataclass, field
import socket
from time import monotonic, perf_counter
from typing import Any, Callable, Dict, Final, Optional, Tuple
from collections.abc import Buffer

import numpy as np

from fft_core import IQInterleavedI16, batch_fft
from io_stuff import ArrF32_1D, create_socket

from scipy.ndimage import gaussian_filter1d

HDR_SZ: Final = 0

@dataclass(frozen=True)
class ParamsStreamUDP:
    port: int = 9999
    rd_timeout_ms: int = 750

    hdr_sz: int = HDR_SZ
    payload_sz: int = 8192

    pack_sz: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "pack_sz", self.hdr_sz + self.payload_sz)

    def to_str(self) -> str:
        return (
            f"PKT SZ: full={self.pack_sz:_}, "
            f"hdr={self.hdr_sz:_}, "
            f"payload={self.payload_sz:_} | "
            f"port={self.port}"
        )


@dataclass(frozen=True)
class ParamsDSP:
    Fs: float
    Fc: float = 0.0
    fft_n: int = 256
    fft_batch_depth: int = 128  # fft windows in batch
    sigma: float = 1.75         # gaussian
    ema_alpha: Optional[float] = None     # for time result acumulate
    
    def gen_freq_bins(self)->np.ndarray:
        dF = self.Fs / self.fft_n
        freq_bins = (np.arange(self.fft_n, dtype=np.float32) - (self.fft_n // 2)) * dF + self.Fc
        return freq_bins


# =====================================================
# UDP Payload обробка
# =====================================================

def proc_udp_payload(data: Buffer, hdr_sz: int) -> Tuple[memoryview, Optional[memoryview]]:
    """
    Обробка UDP payload: пропуск заголовка.
    Повертає в'ю (zero-copy), не копію.
    
    Args:
        data: Buffer of bytes - дані пакету
        hdr_sz: розмір заголовка
        
    Returns:
        (memoryview без заголовка і опціоанльно  в'ю на заголорок)
    """
    if not hdr_sz:
        return memoryview(data), None
    return memoryview(data)[hdr_sz:], memoryview(data)[:hdr_sz]

class RingBuffer:
    """
    Циклічний буфер для накопичення int16 даних без переалокацій.
    Архітектура:
    - Основний буфер: int16 дані (циклічний write_pos/read_pos)
    """
    
    def __init__(self, el_count: int):
        self.buffer = np.empty(el_count, dtype=np.int16)
        self.el_count = el_count
        self.write_pos = 0
        self.read_pos = 0
        self.el_available = 0
        self.overrun_events = 0          # скільки разів сталося переповнення
        self.overrun_raw_dropped = 0     # скільки int16 елементів "випало" сумарно
        
    def push_raw_data(self, data: IQInterleavedI16, seq_n: Optional[int]=None) -> int:
        """
        Додаємо IQInterleavedI16 дані
        Args:
            data: IQInterleavedI16 (memoryview або np.ndarray з int16 1D continious)
            seq_n: seq_num пакету (опціонально, для узгодженності з легасі)
        Returns:
            кількість доданих елементів (int16)
        """
        count: int = len(data)
        free = self.el_count - self.el_available
        if count > free:
            drop = count - free
            self.overrun_events += 1
            self.overrun_raw_dropped += drop

            # advance read_pos to discard oldest `drop` elements
            self.read_pos = (self.read_pos + drop) % self.el_count
            self.el_available -= drop
        # Записуємо дані в основний буфер
        if self.write_pos + count <= self.el_count:
            self.buffer[self.write_pos:self.write_pos + count] = data[:count]
        else:
            first_chunk = self.el_count - self.write_pos
            self.buffer[self.write_pos:] = data[:first_chunk]
            self.buffer[:count - first_chunk] = data[first_chunk:count]
        
        # Обюробляємо seq_num (додам пізніше)
        if seq_n is not None:
            ...
        self.write_pos = (self.write_pos + count) % self.el_count
        self.el_available = min(self.el_available + count, self.el_count)
        return count
    
    def pop_raw_data(self, count: int) -> Optional[IQInterleavedI16]:
        """
        Повертає count int16 елементів (або всі наявні).
        """
        if self.el_available == 0:
            return None
        count = min(count, self.el_available)
        out = np.empty(count, dtype=np.int16)
        
        # Читаємо дані з циклічного буфера
        if self.read_pos + count <= self.el_count:
            out[:] = self.buffer[self.read_pos:self.read_pos + count]
        else:
            first_chunk = self.el_count - self.read_pos
            out[:first_chunk] = self.buffer[self.read_pos:]
            out[first_chunk:] = self.buffer[:count - first_chunk]
        self.read_pos = (self.read_pos + count) % self.el_count
        self.el_available -= count
        return out

    def pop_into(self, out: IQInterleavedI16) -> int:
        """
        Заповнює out даними з рінга.
        Returns: скільки int16 реально записано (0 якщо нема даних)
        """
        if self.el_available == 0:
            return 0

        count = min(len(out), self.el_available)

        if self.read_pos + count <= self.el_count:
            out[:count] = self.buffer[self.read_pos:self.read_pos + count]
        else:
            first_chunk = self.el_count - self.read_pos
            out[:first_chunk] = self.buffer[self.read_pos:]
            out[first_chunk:count] = self.buffer[:count - first_chunk]

        self.read_pos = (self.read_pos + count) % self.el_count
        self.el_available -= count
        return count

    def read_samples(self, samp_count: int) -> Optional[IQInterleavedI16]:
        """
        Domain semantic. Wrapper over pop_raw_data
        Читаємо samp_count IQ пар.
        sample is 2x raw
        """
        raw_count = samp_count * 2
        return self.pop_raw_data(raw_count)
    
    def available(self) -> int:
        """Скільки raw int16 елементів доступно"""
        return self.el_available


def do_hard_work(
    prm_stream: ParamsStreamUDP ,
    prm_dsp: ParamsDSP,
    *,
    stop_requested: Optional[Callable[[],bool]] = None,
    on_spectrum_cb: Optional[Callable[[np.ndarray, Optional[ np.ndarray], Dict], None]] = None,
    out_fps: float = 24.0,
) -> None:
    """
    one call for life time
    """
    s = create_socket(prm_stream.port, prm_stream.rd_timeout_ms)
    print(f"[INFO] sock created.  {prm_stream.to_str()}.",flush=True)
    pack_sz = prm_stream.pack_sz
    hdr_sz = prm_stream.hdr_sz
    # ===== RING BUFFER =====
    Fs, _, fft_n, fft_batch_depth, sigma, ema_alpha = astuple(prm_dsp)
    
    ring_buffer_size = fft_n * 1_000
    ring_buf = RingBuffer(ring_buffer_size)

    batch_len = fft_n * fft_batch_depth
    need_raw = batch_len * 2  # бо в рінгу interleaved i16: i0 q0 i1 q1 ...
    
    # ===== PRE-ALLOC BUFFERS (FOR FFT and EMA accumulation) =====
    batch_inp = np.empty(batch_len, dtype=np.complex64)
    scratch = np.empty((fft_batch_depth, fft_n), dtype=np.float32)
    tmp = np.empty(fft_n, dtype=np.float32)
    power: ArrF32_1D = np.empty(fft_n, dtype=np.float32)      # linear power (NOT dB, NOT shifted)
    acc_power: ArrF32_1D = np.empty(fft_n, dtype=np.float32)  # EMA buffer (same domain as power_db)
    raw_iq_i16 = np.empty(need_raw, dtype=np.int16)
    i_f32 = np.empty(batch_len, dtype=np.float32)
    q_f32 = np.empty(batch_len, dtype=np.float32)
    
    # ===== FREQUENCY BINS =====
    dF = Fs / fft_n
    # freq_bins = (np.arange(fft_n, dtype=np.float32) - (fft_n // 2)) * dF + Fc
    pkt_buf = np.empty(pack_sz, dtype=np.uint8)
    batch_duration = fft_batch_depth * fft_n / Fs

    pkt_count = 0
    batches_processed = 0
    seq_num: int = None
    gap_count = 0
    gap_len = 0
    t_start = t0 = monotonic()
    rb_ovr_events_0 = 0
    rb_ovr_drop_0 = 0
    pkt_count_0 = pkt_count
    byte_count_0 = 0
    fft_calls = 0
    fft_time_acc = 0.0
    silence_dur = 0.0
    silence_start: float = None
    show_silence_every_sec = 1.0
    t_next_show = 0.0
    stream_start: float = None
    mbps_rate: float = None
    avg_fft_ms: float = None
    log_period = 1.0
    t_next_log = t_start + log_period
    
    cb_period = 1.0 / max(float(out_fps), 0.1)
    t_next_callback = t_start + cb_period
    
    mbps_rate = 0.0
    avg_fft_ms = 0.0
                
    meta: Dict[str,Any] = {
        "batch_dur": batch_duration,
        "batch_depth": fft_batch_depth,
        "batches_processed": None,
        "freq_resolution_hz": dF,
        "fft_n": fft_n,
        "pkt_sz": pack_sz,
        "gap_count": None,
        "gap_len": None,
        "mbps": mbps_rate,
        "avg_fft_ms":avg_fft_ms,
        "pkt_total": None,
        "time_stamp": None,
        "uptime": None
    }
    try:
        while True:
            if stop_requested is not None and stop_requested():
                print()
                break
            try:
                n = s.recv_into(pkt_buf, pack_sz)
                byte_count_0 += n
                silence_start = None
                if stream_start is None:
                    stream_start = monotonic() 
            except socket.timeout:
                t_cur = monotonic() 
                if silence_start is None:
                    silence_start = t_cur
                    silence_dur = 0.0
                    print(f"\n[silence begin]")
                    t_next_show = silence_start + show_silence_every_sec
                    stream_start = None
                else:
                    if t_cur >= t_next_show:
                        silence_dur = t_cur - silence_start
                        print(f"[silence {silence_dur:.1f}s]", end="\r")
                        t_next_show = t_cur + show_silence_every_sec
                continue
            pkt_count += 1
            payload, hdr = proc_udp_payload(pkt_buf, hdr_sz)
            if (len(payload) & 0x3) != 0:
                raise ValueError(f"Bad IQ payload size: {len(payload)} bytes (not multiple of 4)")
            raw = np.frombuffer(payload, dtype="<i2") # LE
            if hdr is not None:
                curr_num = int(np.frombuffer(hdr, dtype=np.int64, count=1)[0])
                if seq_num is not None:
                    gap = curr_num - seq_num
                    if gap > 1:
                        gap_count += 1
                        gap_len += gap
                        print(f"\n[DBG] {gap=} ({gap_count=}  {pkt_count:_})")
                seq_num = curr_num

            ring_buf.push_raw_data(raw, seq_num)

            if ring_buf.available() < need_raw:
                continue

            got = ring_buf.pop_into(raw_iq_i16)
            if got != need_raw:
                continue
            
            np.copyto(i_f32, raw_iq_i16[0::2], casting="unsafe")
            np.copyto(q_f32, raw_iq_i16[1::2], casting="unsafe")
            batch_inp.real = i_f32
            batch_inp.imag = q_f32

            t_fft0 = perf_counter()
            batch_fft(batch_inp, fft_n, fft_batch_depth, power, scratch, tmp, workers=-1)
            t_fft1 = perf_counter()
            fft_time_acc += (t_fft1 - t_fft0)
            fft_calls += 1
            
            batches_processed = fft_calls
            if ema_alpha is None:
                # EMA вимкнена: acc_power не оновлюємо (залишаємо як буфер, але не використовуємо)
                pass
            else:
                if batches_processed == 1:
                    acc_power[:] = power
                else:
                    np.multiply(acc_power, (1.0 - ema_alpha), out=acc_power)
                    np.multiply(power, ema_alpha, out=tmp)
                    np.add(acc_power, tmp, out=acc_power)

            t1 = monotonic()
            if on_spectrum_cb and t1 >= t_next_callback:
                meta["batches_processed"] = batches_processed
                meta["time_stamp"] = t1
                meta["mbps"] = mbps_rate
                meta["avg_fft_ms"] = avg_fft_ms
                meta["gap_count"] = gap_count
                meta["gap_len"] = gap_len
                meta["pkt_total"] = pkt_count
                meta["uptime"] = t1 - t_start
                meta["rb_ovr_events"] = ring_buf.overrun_events
                meta["rb_ovr_drop_raw"] = ring_buf.overrun_raw_dropped
                ema_out = None if ema_alpha is None else acc_power
                on_spectrum_cb(power, ema_out, meta)
                # schedule next callback tick (без дрейфу)
                t_next_callback += cb_period
                if t_next_callback < t1:
                    t_next_callback = t1 + cb_period

            # Terminal log (time-based)
            if t1 >= t_next_log:
                stt_str = f"pkt={pkt_count:6_}  batches={batches_processed:6_}  gaps={gap_count:_} | "
                uptime = t1 - t_start
                dt = t1 - t0
                if dt > 0:
                    pkt_rate = (pkt_count - pkt_count_0) / dt
                    Bps = byte_count_0 / dt
                    mib_rate = Bps / (1024 * 1024)
                    mbps_rate = (Bps * 8) / 1_000_000
                    avg_fft_ms = (fft_time_acc / max(fft_calls, 1)) * 1e3

                    ovr_events = ring_buf.overrun_events - rb_ovr_events_0
                    ovr_drop_raw = ring_buf.overrun_raw_dropped - rb_ovr_drop_0
                    ovr_drop_iq = ovr_drop_raw // 2

                    print(
                        f"[{uptime:5.1f} s]"
                        f"  | RATE: {pkt_rate:8.1f} pkt/s  {mib_rate:6.2f} MiB/s  {mbps_rate:6.2f} Mbps  (dt={dt:6.2f}s)"
                        f"  | fft_avg_total: {avg_fft_ms:6.3f} ms"
                        f"  | RB_OVR: {ovr_events:_} ev  drop={ovr_drop_raw:_} raw ({ovr_drop_iq:_} IQ)"
                        f"  | ",
                        end="\r",
                    )

                # reset window
                t0 = t1
                rb_ovr_events_0 = ring_buf.overrun_events
                rb_ovr_drop_0 = ring_buf.overrun_raw_dropped
                pkt_count_0 = pkt_count
                byte_count_0 = 0

                # schedule next log tick (без дрейфу)
                t_next_log += log_period
                if t_next_log < t1:
                    t_next_log = t1 + log_period
    finally:
        try:
            s.close()
        except Exception:
            pass
    print(f"[do_hard_work] Done!")
