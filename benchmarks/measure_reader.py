"""Measure deterministic native PLE page gathers."""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from sglang_ssd_stream._io import PageReader


def _evict(path: str) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(descriptor)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--rows", type=int, default=11800)
    parser.add_argument("--row-count", type=int, default=320001536)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--evict", action="store_true")
    args = parser.parse_args()

    if args.evict:
        _evict(args.path)
    reader = PageReader(
        args.path,
        160,
        0,
        0,
        args.row_count,
        measure_physical_io=True,
    )
    output = np.empty((args.rows, 160), dtype=np.uint8)

    for label, seed in (
        ("cold", args.seed),
        ("repeat", args.seed),
        ("unrelated", args.seed + 1),
    ):
        ids = np.random.default_rng(seed).integers(
            0, args.row_count, size=args.rows, dtype=np.int64
        )
        started = time.perf_counter_ns()
        stats = reader.gather(ids, output)
        wall_ns = time.perf_counter_ns() - started
        print(
            json.dumps(
                {
                    "label": label,
                    "wall_ms": wall_ns / 1e6,
                    "rows": stats.rows,
                    "unique_pages": stats.unique_pages,
                    "submitted_bytes": stats.submitted_bytes,
                    "physical_bytes": stats.physical_bytes,
                    "peak_queue_depth": stats.peak_queue_depth,
                    "read_batches": stats.read_batches,
                    "io_ms": stats.io_ns / 1e6,
                    "scatter_ms": stats.scatter_ns / 1e6,
                    "completion_latency_mean_us": (
                        stats.completion_latency_mean_ns / 1e3
                    ),
                    "completion_latency_max_us": (
                        stats.completion_latency_max_ns / 1e3
                    ),
                    "checksum": int(output.sum(dtype=np.uint64)),
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
