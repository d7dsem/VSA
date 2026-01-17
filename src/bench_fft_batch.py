# bench_fft_batch.py
import argparse
from time import perf_counter
import numpy as np

from fft_core import batch_fft  # expects: batch_fft(batch_inp, fft_n, fft_batch, power, scratch, tmp, workers=..)


def _bench_point_core(
    fft_n: int,
    fft_batch: int,
    workers: int,
    seconds: float,
    warmup_iters: int,
    rng: np.random.Generator,
) -> dict:
    batch_len = fft_n * fft_batch

    # prealloc
    batch_inp = np.empty(batch_len, dtype=np.complex64)
    scratch = np.empty((fft_batch, fft_n), dtype=np.float32)
    tmp = np.empty(fft_n, dtype=np.float32)
    power = np.empty(fft_n, dtype=np.float32)

    # init data (fixed, no allocations in hot loop)
    batch_inp.real = rng.standard_normal(batch_len, dtype=np.float32)
    batch_inp.imag = rng.standard_normal(batch_len, dtype=np.float32)

    # warmup
    for _ in range(warmup_iters):
        batch_fft(batch_inp, fft_n, fft_batch, power, scratch, tmp, workers=workers)

    # timed
    t0 = perf_counter()
    calls = 0
    while True:
        batch_fft(batch_inp, fft_n, fft_batch, power, scratch, tmp, workers=workers)
        calls += 1
        if (perf_counter() - t0) >= seconds:
            break
    t1 = perf_counter()
    dt = t1 - t0

    samples = calls * batch_len
    return {
        "calls": calls,
        "dt": dt,
        "ms_per_batch": (dt / calls) * 1e3,
        "ns_per_sample": (dt / samples) * 1e9,
        "samples_per_sec": samples / dt,
    }


def _bench_point_pipe(
    fft_n: int,
    fft_batch: int,
    workers: int,
    seconds: float,
    warmup_iters: int,
    rng: np.random.Generator,
) -> dict:
    """
    Bench approximating your real path:
      raw int16 interleaved IQ -> float32 I/Q (no alloc via np.copyto) -> complex64 batch_inp -> batch_fft
    """
    batch_len = fft_n * fft_batch
    need_raw = batch_len * 2  # int16 count

    # prealloc
    raw_i16 = np.empty(need_raw, dtype=np.int16)
    i_f32 = np.empty(batch_len, dtype=np.float32)
    q_f32 = np.empty(batch_len, dtype=np.float32)

    batch_inp = np.empty(batch_len, dtype=np.complex64)
    scratch = np.empty((fft_batch, fft_n), dtype=np.float32)
    tmp = np.empty(fft_n, dtype=np.float32)
    power = np.empty(fft_n, dtype=np.float32)

    # init raw (fixed)
    raw_i16[:] = rng.integers(-32768, 32767, size=need_raw, dtype=np.int16)

    def step() -> None:
        # int16 -> float32 without allocating: np.copyto does cast into prealloc dst
        np.copyto(i_f32, raw_i16[0::2], casting="unsafe")
        np.copyto(q_f32, raw_i16[1::2], casting="unsafe")
        batch_inp.real = i_f32
        batch_inp.imag = q_f32
        batch_fft(batch_inp, fft_n, fft_batch, power, scratch, tmp, workers=workers)

    # warmup
    for _ in range(warmup_iters):
        step()

    # timed
    t0 = perf_counter()
    calls = 0
    while True:
        step()
        calls += 1
        if (perf_counter() - t0) >= seconds:
            break
    t1 = perf_counter()
    dt = t1 - t0

    samples = calls * batch_len
    input_bytes = calls * (need_raw * 2)  # int16 -> 2 bytes
    return {
        "calls": calls,
        "dt": dt,
        "ms_per_batch": (dt / calls) * 1e3,
        "ns_per_sample": (dt / samples) * 1e9,
        "samples_per_sec": samples / dt,
        "MiB_per_sec_in": (input_bytes / dt) / (1024 * 1024),
    }


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fft-n", type=int, default=1024)
    ap.add_argument("--batches", type=str, default="1,2,4,8,16,32,64")
    ap.add_argument("--workers", type=str, default="-1,1,2,4,8")
    ap.add_argument("--mode", choices=["core", "pipe", "both"], default="both")
    ap.add_argument("--seconds", type=float, default=1.0)
    ap.add_argument("--warmup-iters", type=int, default=50)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    fft_n = args.fft_n
    batch_list = _parse_int_list(args.batches)
    workers_list = _parse_int_list(args.workers)

    rng = np.random.default_rng(args.seed)

    def run_one(mode: str, fft_batch: int, workers: int) -> dict:
        if mode == "core":
            return _bench_point_core(fft_n, fft_batch, workers, args.seconds, args.warmup_iters, rng)
        if mode == "pipe":
            return _bench_point_pipe(fft_n, fft_batch, workers, args.seconds, args.warmup_iters, rng)
        raise ValueError(mode)

    def median_of(recs: list[dict], key: str) -> float:
        xs = sorted(r[key] for r in recs)
        return xs[len(xs) // 2]

    modes = ["core", "pipe"] if args.mode == "both" else [args.mode]

    print(f"fft_n={fft_n}  seconds={args.seconds}  warmup={args.warmup_iters}  repeats={args.repeats}")
    print("mode  fft_batch  workers   ms/batch    ns/sample    Msamples/s   MiB/s_in")
    print("----  ---------  -------  ---------  ----------  ----------  --------")

    best = None  # (ns_per_sample, mode, fft_batch, workers)
    for mode in modes:
        for fft_batch in batch_list:
            for workers in workers_list:
                recs = [run_one(mode, fft_batch, workers) for _ in range(args.repeats)]

                ms_per_batch = median_of(recs, "ms_per_batch")
                ns_per_sample = median_of(recs, "ns_per_sample")
                msamples_s = median_of(recs, "samples_per_sec") / 1e6
                mib_in = median_of(recs, "MiB_per_sec_in") if mode == "pipe" else float("nan")

                print(
                    f"{mode:4s}  {fft_batch:9d}  {workers:7d}  "
                    f"{ms_per_batch:9.3f}  {ns_per_sample:10.3f}  {msamples_s:10.3f}  "
                    f"{mib_in:8.2f}"
                )

                cand = (ns_per_sample, mode, fft_batch, workers)
                if best is None or cand[0] < best[0]:
                    best = cand

    if best is not None:
        ns, mode, fft_batch, workers = best
        print(f"\nBEST by ns/sample: mode={mode}  fft_batch={fft_batch}  workers={workers}  ns/sample={ns:.3f}")


if __name__ == "__main__":
    main()
