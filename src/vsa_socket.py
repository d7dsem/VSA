import socket
from time import monotonic, perf_counter
from typing import Annotated, Literal, Optional, Tuple, TypeAlias, Final
from matplotlib import pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

from fft_core import ArrF32_1D, batch_fft, swap_freq, to_dbfs, IQInterleavedI16
from io_stuff import SOCK_BUF_SZ, create_socket
from vsa import P_FS, VSA_SPECTR
from data_layer import HDR_SZ, RingBuffer, proc_udp_payload

def do_sock_work(
    port: int,
    rd_timeout_ms: int = 750,
    hdr_sz: int = HDR_SZ,
    pack_sz: int = 8192 + HDR_SZ,
    Fs: float = 480e3,
    # Fc: float = 430.1e6,
    Fc: float = 423.85e6,
    fft_n: int = 256,
    fft_batch_depth : int = 128, # fft windows in batch
    sigma: float = 1.75,
    ema_alpha: float = 0.1,
    target_fps: float = 24.0,
    draw_every: int = 100,
    vsa_on: bool = True
) -> None:

    s = create_socket(port, rd_timeout_ms)
    print(f"[do_vsa_socket] listening on port {port}", flush=True)
    print(f"[do_vsa_socket] {pack_sz=:_}, {hdr_sz=}", flush=True)
    
    render_period = 1.0 / target_fps
    # ===== RING BUFFER =====
    
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
    freq_bins = (np.arange(fft_n, dtype=np.float32) - (fft_n // 2)) * dF + Fc
    pkt_buf = np.empty(pack_sz, dtype=np.uint8)
    batch_duration = fft_batch_depth * fft_n / Fs
    if vsa_on:
        vsa = VSA_SPECTR(
            freq_bins,
            render_dt=0.0001,
            spec_y_lim=None,
            use_dbfs=True,
            title_base=f"FC {int(round(Fc)):_} Hz  | {fft_n=}  {fft_batch_depth=} dF={Fs/fft_n/1e3:.2f} KHz"
        )
    else:
        vsa = None
    
    pkt_count = 0
    batches_processed = 0
    seq_num: int = None
    gap_count = 0
    gap_len = 0
    t_next_render = t_start = t0 = monotonic()
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

    while True:
        if vsa and vsa.stop_requested:
            print(f"[DBG] fig close requested")
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
        if batches_processed == 1:
            acc_power[:] = power
        else:
            np.multiply(acc_power, (1.0 - ema_alpha), out=acc_power)
            np.multiply(power, ema_alpha, out=tmp)
            np.add(acc_power, tmp, out=acc_power)


        stt_str = f"pkt={pkt_count:6_}  batches={batches_processed:6_}  gaps={gap_count:_} | "
        # Terminal log
        if batches_processed % draw_every == 0:
            t1 = monotonic()
            uptime = t1 - t_start
            dt = t1 - t0
            if dt > 0:
                pkt_rate = (pkt_count - pkt_count_0) / dt          # pkt/s

                Bps = byte_count_0 / dt                          # bytes per second
                mib_rate = Bps / (1024 * 1024)                   # MiB/s
                mbps_rate = (Bps * 8) / 1_000_000                # Mbps (SI)
                avg_fft_ms = (fft_time_acc / max(fft_calls, 1)) * 1e3
                
                ovr_events = ring_buf.overrun_events - rb_ovr_events_0
                ovr_drop_raw = ring_buf.overrun_raw_dropped - rb_ovr_drop_0
                ovr_drop_iq = ovr_drop_raw // 2

                print(f"[{uptime:5.1f} s]  "
                    f"  | RATE: {pkt_rate:8.1f} pkt/s  {mib_rate:6.2f} MiB/s  {mbps_rate:6.2f} Mbps  (dt={dt:6.2f}s)"
                    f"  | fft_avg_total: {avg_fft_ms:6.3f} ms"
                    f"  | RB_OVR: {ovr_events:_} ev  drop={ovr_drop_raw:_} raw ({ovr_drop_iq:_} IQ)"
                    f"  | ", end="\r"
                )

            # reset window
            t0 = t1
            rb_ovr_events_0 = ring_buf.overrun_events
            rb_ovr_drop_0 = ring_buf.overrun_raw_dropped
            pkt_count_0 = pkt_count
            byte_count_0 = 0
        # Visualization
        t_now = monotonic()
        if t_now < t_next_render:
            continue
        else:
            t_next_render += render_period
            if t_next_render < t_now:
                t_next_render = t_now + render_period
            display_power_db = acc_power.copy()
            to_dbfs(display_power_db, p_fs=P_FS)
            swap_freq(display_power_db)
            y_spec_smooth = gaussian_filter1d(display_power_db, sigma=sigma)
            time_str = "silence" if stream_start is None else f"uptime: {monotonic()-stream_start:4.1f} s | "
            stt_str = time_str  + stt_str
            if mbps_rate:
                stt_str += f"{mbps_rate:6.2f} Mbps | "
            if avg_fft_ms:
                rate = batch_duration*1e3 / avg_fft_ms
                stt_str += f"batch_depth={fft_batch_depth} fft_avg_total: {avg_fft_ms:6.3f} ms  batch: {batch_duration*1e3:6.2f} ms  (x{rate:.2f})"
            vsa.update(display_power_db, y_spec_smooth, stt_str)

    print(f"\n\nDone")

if __name__ == '__main__':
    try:
        do_sock_work(9999)
    except KeyboardInterrupt:
        print(f"\n\nDone (Ctrl+C)")
        pass
    except Exception as e:
        print(f"[ERR] exception: {e}")
        
