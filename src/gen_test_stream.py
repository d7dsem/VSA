#!/usr/bin/env python3
from pathlib import Path
import sys
# print(sys.executable)
# print(sys.version)
# raise SystemExit
# AI do not remove!
# print(f"====> Running: {sys.executable}")

import argparse
import numpy as np

from time import perf_counter, sleep
import socket
import signal

from fft_core import IQInterleavedI16
from io_stuff import FReader


def signal_handler(sig, frame):
    print('\n\nCtrl+C detected. Terminating...')
    sys.exit(0)


def main_udp_stream(args: argparse.Namespace):
    # attach signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)

    # calc send delay between packets
    delay_sec: float = None
    if args.spd_pkt:
        delay_sec = 1.0 / args.spd_pkt
    elif args.spd_bit:
        delay_sec = (args.size * 8) / args.spd_bit
    elif args.spd_byte:
        delay_sec = args.size / args.spd_byte

    # create send buf with int64 view to first 8 bytes for seq_num (LE)
    send_buf = np.zeros(args.size, dtype=np.uint8)
    _hdr_sz = np.int64().itemsize
    seq_view = send_buf[:_hdr_sz].view(np.int64)
    # parse addr
    host, port = args.addr.split(':')
    port = int(port)

    # open udp sock and connect to dest ip:port
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024 * 1024)  # 1 MB
    # Does not allow localhost conn to on Liunux w/o stream consumer
    # sock.connect((host, port))

    seq_num = 0
    t_start = perf_counter()
    t_last_stat = t_start
    pkt_count = 0

    print(f"Starting UDP stream to {host}:{port}, pkt_size={args.size:_}B, delay={delay_sec}s")
    
    payload_sz = args.size - _hdr_sz  
    iq_count = payload_sz // 4
    if args.file:
        fr_args = argparse.Namespace(
            file=args.file,
            samp_rate=480e3
        )
        fr = FReader(fr_args)
        iq_data: IQInterleavedI16 = np.empty(iq_count*2, dtype=np.int16)
    else:
        fr = None
    n_rounds = 0
    while True:
        # update seq_num in buffer
        seq_view[0] = seq_num
        
        if fr == None:
            iq_data = np.random.randint(-24000, +24000, size=iq_count, dtype=np.int16)
        else:
            rd = fr.read_samples_into(iq_data, iq_count)
            if rd != iq_count:
                fr.jump_to_pos(0)
                n_rounds += 1
                print(f"\n{n_rounds:2} Restart file!")
        
        send_buf[_hdr_sz:_hdr_sz+len(iq_data)*2] = iq_data.view(np.uint8)
        # send packet
        # sock.send(send_buf.tobytes()) # if conn succes
        sock.sendto(send_buf.tobytes(), (host,port))

        seq_num += 1
        pkt_count += 1
        elapsed = perf_counter() - t_start

        # every sec stat print
        if perf_counter() - t_last_stat >= 1.0:
            rate_pps = pkt_count / (perf_counter() - t_start)
            rate_mbps = (rate_pps * args.size * 8) / 1_000_000
            seq_str = f"{seq_num:_}"
            pkt_str = f"{int(rate_pps):_}"
            print(f"Elapsed: {elapsed:4.1f}s | Pkts: {seq_str:12} | Rate: {pkt_str:12} pps / {rate_mbps:>6.2f} Mbps", end='\r')
            t_last_stat = perf_counter()

        # check duration limit
        if args.dur_sec and elapsed >= args.dur_sec:
            print(f"\n\nTransmission duration limit {args.dur_sec}s reached. Terminate.")
            break

        # apply delay if needed
        if delay_sec:
            t_target = t_start + seq_num * delay_sec
            t_remaining = t_target - perf_counter()

            # sleep for bulk of delay, leave 1ms for busy-wait
            if t_remaining > 0.001:
                 sleep(t_remaining - 0.001)

            # precise busy-wait for last millisecond
            while perf_counter() < t_target:
                pass


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Generate test udp data stream')
    parser.add_argument('-sz', '--size', type=int, default=1024,
                        help='Desired pkt size in B (int)')
    
    parser.add_argument("--file", type=Path, default=None, help="File for data (if not provided gen random data)")
    parser.add_argument('--dur-sec', type=float,
                        help='Stream duration in seconds (optional, infinite if not set)')
    parser.add_argument('--addr', type=str, default="127.0.0.1:9999",
                        help="Dst Addr ip:port for send data")

    speed_group = parser.add_mutually_exclusive_group(required=False)
    speed_group.add_argument('--spd-pkt', type=int, metavar='RATE',
                             help='Target rate in packets/sec')
    speed_group.add_argument('--spd-bit', type=int, metavar='RATE',
                             help='Target rate in bits/sec')
    speed_group.add_argument('--spd-byte', type=int, metavar='RATE',
                             help='Target rate in bytes/sec')

    return parser


if __name__ == '__main__':

    if len(sys.argv) > 1:
        parser = _build_cli()
        args = parser.parse_args()
    else:
        # create dev default
        # spd_<> only one or no one allowed
        # as X310
        args = argparse.Namespace(
            dur_sec=None,     # inf send loop
            addr="127.0.0.1:9999",
            size=7184,
            spd_pkt=68_571,
            spd_bit=None,
            spd_byte=None
        )
        # Override
        args = argparse.Namespace(
            dur_sec=None,     # inf send loop
            addr="127.0.0.1:9999",
            file=Path(r"e:\home\d7\Public\signals\store\dmr-x310-480\08_02_2025\Fc_421000000Hz_ch_4_v2.bin"),
            size=8192+8,
            spd_pkt=68_571//256,
            spd_bit=None,
            spd_byte=None
        )

    main_udp_stream(args)