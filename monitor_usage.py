#!/usr/bin/env python3
"""
monitor_usage.py  —  Poll GPU and CPU/RAM usage at fixed intervals.

Runs as a background process alongside training:
  python monitor_usage.py --output profile_logs/usage_london.csv --interval 1 &
  MONITOR_PID=$!
  python -u train_stretch.py ...
  kill $MONITOR_PID

Columns written:
  timestamp, elapsed_s,
  gpu0_util_pct, gpu0_mem_used_mb, gpu0_mem_total_mb, gpu0_power_w,
  cpu_util_pct, ram_used_gb, ram_total_gb
"""

import argparse
import csv
import os
import signal
import subprocess
import sys
import time


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output",   required=True, help="Path to output CSV file")
    p.add_argument("--interval", type=float, default=1.0,
                   help="Sampling interval in seconds (default: 1)")
    p.add_argument("--gpu_id",   type=int,   default=0,
                   help="GPU index to report (default: 0; use -1 for all)")
    return p.parse_args()


def query_nvidia_smi(gpu_id):
    """Return a list of dicts, one per GPU (or just the requested one)."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=5,
        )
        rows = []
        for idx, line in enumerate(result.stdout.strip().split("\n")):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            power_raw = parts[3]
            power = None
            try:
                power = float(power_raw)
            except ValueError:
                pass
            rows.append({
                "gpu_util_pct":     float(parts[0]),
                "gpu_mem_used_mb":  float(parts[1]),
                "gpu_mem_total_mb": float(parts[2]),
                "gpu_power_w":      power,
            })
        if gpu_id >= 0 and rows:
            rows = [rows[gpu_id]] if gpu_id < len(rows) else rows[:1]
        return rows
    except Exception:
        return [{"gpu_util_pct": None, "gpu_mem_used_mb": None,
                 "gpu_mem_total_mb": None, "gpu_power_w": None}]


def query_cpu():
    try:
        import psutil
        vm = psutil.virtual_memory()
        return {
            "cpu_util_pct": psutil.cpu_percent(interval=None),
            "ram_used_gb":  round(vm.used  / 1e9, 2),
            "ram_total_gb": round(vm.total / 1e9, 2),
        }
    except ImportError:
        return {"cpu_util_pct": None, "ram_used_gb": None, "ram_total_gb": None}


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    fieldnames = [
        "timestamp", "elapsed_s",
        "gpu_util_pct", "gpu_mem_used_mb", "gpu_mem_total_mb", "gpu_power_w",
        "cpu_util_pct", "ram_used_gb", "ram_total_gb",
    ]

    # Warm up cpu_percent (first call always returns 0.0)
    try:
        import psutil
        psutil.cpu_percent(interval=None)
    except ImportError:
        pass

    start = time.time()
    running = True

    def handle_signal(signum, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT,  handle_signal)

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        while running:
            t = time.time()
            gpu_rows = query_nvidia_smi(args.gpu_id)
            cpu_row  = query_cpu()

            g = gpu_rows[0] if gpu_rows else {}
            row = {
                "timestamp":       time.strftime("%Y-%m-%d %H:%M:%S"),
                "elapsed_s":       round(t - start, 1),
                "gpu_util_pct":    g.get("gpu_util_pct"),
                "gpu_mem_used_mb": g.get("gpu_mem_used_mb"),
                "gpu_mem_total_mb":g.get("gpu_mem_total_mb"),
                "gpu_power_w":     g.get("gpu_power_w"),
                **cpu_row,
            }
            writer.writerow(row)
            f.flush()

            elapsed = time.time() - t
            sleep_for = max(0.0, args.interval - elapsed)
            time.sleep(sleep_for)


if __name__ == "__main__":
    main()
