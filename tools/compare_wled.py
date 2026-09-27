#!/usr/bin/env python3
"""
Captures WLED audiosync-v2 UDP packets from multiple senders (e.g. this app's
WledSRServer.Cli and a real WLED device, both fed the same line-out signal) and
renders a side-by-side comparison, for tuning this app's audio processing to
better match WLED's own.

Packets are told apart by source IP - no protocol-level sender identity exists,
so run this on the machine that can see both (e.g. the one also running the CLI,
with the WLED device on the same LAN broadcasting/sending to it too). Any source
IP belonging to this machine is auto-labelled "local"; everything else is assumed
to be the WLED device and auto-labelled "wled" - no --label needed for the usual
two-sender case. --label is only for overriding those defaults (e.g. more than
one non-local sender, or you just want different names).

Usage:
    python3 compare_wled.py --port 11988 --duration 30

    # or just Ctrl+C to stop whenever you've captured enough:
    python3 compare_wled.py --port 11988

Output (in --out-dir, default ./wled-compare-<timestamp>/):
    packets_<label-or-ip>.csv   - one row per packet, per source
    comparison.png              - major peak / envelope / FFT-bin plots (needs matplotlib+numpy)

Packet layout (44 bytes, matches WledSRServer/AudioSyncPacket.cs AudioSyncPacket_v2):
    char[6]   header             "00002\0"
    uint8[2]  pressure           fixed-point sound pressure (not decoded here)
    float     sampleRaw
    float     sampleSmth
    uint8     samplePeak
    uint8     frameCounter
    uint8[16] fftResult          16 GEQ bins, one byte each
    uint16    zeroCrossingCount
    float     fftMagnitude
    float     fftMajorPeak
"""
import argparse
import csv
import datetime
import os
import signal
import socket
import struct
import sys
from collections import defaultdict

PACKET_FMT = "<6s2sffBB16sHff"
PACKET_SIZE = struct.calcsize(PACKET_FMT)
NUM_BINS = 16


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=11988, help="UDP port to listen on (default: 11988, WLED's default SR port)")
    p.add_argument("--duration", type=float, default=None, help="Stop after this many seconds (default: run until Ctrl+C)")
    p.add_argument("--out-dir", default=None, help="Where to write CSVs/plot (default: ./wled-compare-<timestamp>/)")
    p.add_argument("--label", action="append", default=[], metavar="IP=NAME",
                   help="Override the auto-assigned name for a source IP, e.g. --label 192.168.1.50=wled-2. Repeatable.")
    return p.parse_args()


def get_local_ips():
    """Best-effort set of this machine's own IPv4 addresses, so its own packets
    can be auto-labelled without the caller needing to already know its IP."""
    ips = {"127.0.0.1"}
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except socket.gaierror:
        pass
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))  # no packet is actually sent, just resolves routing
            ips.add(s.getsockname()[0])
    except OSError:
        pass
    return ips


def label_for(ip, labels, local_ips):
    if ip in labels:
        return labels[ip]
    if ip in local_ips:
        return f"local ({ip})"
    return f"wled ({ip})"


def safe_filename(name):
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


def capture(port, duration):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", port))
    sock.settimeout(0.5)

    records = defaultdict(list)
    start = None
    stop_requested = False

    def handle_sigint(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, handle_sigint)

    print(f"Listening on UDP :{port} (packet size {PACKET_SIZE} bytes). Press Ctrl+C to stop.")
    if duration:
        print(f"Will stop automatically after {duration:.1f}s.")

    total = 0
    malformed = 0
    while not stop_requested:
        if duration is not None and start is not None and (datetime.datetime.now() - start).total_seconds() >= duration:
            break
        try:
            data, (ip, _port) = sock.recvfrom(2048)
        except socket.timeout:
            continue

        now = datetime.datetime.now()
        if start is None:
            start = now

        if len(data) != PACKET_SIZE:
            malformed += 1
            continue

        header, _pressure, sample_raw, sample_smth, sample_peak, frame_counter, fft_bins, zcc, fft_mag, fft_major_peak = \
            struct.unpack(PACKET_FMT, data)

        if not header.startswith(b"0000"):
            malformed += 1
            continue

        total += 1
        t = (now - start).total_seconds()
        row = {
            "t": t,
            "wall_time": now.isoformat(timespec="milliseconds"),
            "sample_raw": sample_raw,
            "sample_smth": sample_smth,
            "sample_peak": sample_peak,
            "frame_counter": frame_counter,
            "zero_crossing_count": zcc,
            "fft_magnitude": fft_mag,
            "fft_major_peak": fft_major_peak,
        }
        for i, b in enumerate(fft_bins):
            row[f"bin{i}"] = b
        records[ip].append(row)

        if total % 20 == 0:
            print(f"\r{total} packets from {len(records)} source(s) ({malformed} malformed)   ", end="", flush=True)

    print()
    sock.close()
    return records, malformed


def write_csvs(records, labels, local_ips, out_dir):
    fieldnames = ["t", "wall_time", "sample_raw", "sample_smth", "sample_peak", "frame_counter",
                  "zero_crossing_count", "fft_magnitude", "fft_major_peak"] + [f"bin{i}" for i in range(NUM_BINS)]

    paths = {}
    for ip, rows in records.items():
        name = safe_filename(label_for(ip, labels, local_ips))
        path = os.path.join(out_dir, f"packets_{name}.csv")
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        paths[ip] = path
    return paths


def print_summary(records, labels, local_ips):
    print("\nSummary:")
    for ip, rows in records.items():
        if not rows:
            continue
        duration = rows[-1]["t"] - rows[0]["t"]
        rate = len(rows) / duration if duration > 0 else float("nan")
        print(f"  {label_for(ip, labels, local_ips):20s} ({ip:15s})  {len(rows):5d} packets  ~{rate:5.1f} pkt/s over {duration:.1f}s")


def plot_comparison(records, labels, local_ips, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("\nmatplotlib/numpy not installed - skipping plot (CSVs were still written).")
        print("Install with: pip install matplotlib numpy")
        return

    sources = [ip for ip, rows in records.items() if rows]
    if not sources:
        print("No packets captured - nothing to plot.")
        return

    n_sources = len(sources)
    fig, axes = plt.subplots(2 + n_sources, 1, figsize=(12, 4 + 2.2 * n_sources), sharex=True)
    ax_peak, ax_env = axes[0], axes[1]
    bin_axes = axes[2:]

    colors = plt.cm.tab10.colors
    for i, ip in enumerate(sources):
        rows = records[ip]
        t = [r["t"] for r in rows]
        color = colors[i % len(colors)]
        name = label_for(ip, labels, local_ips)

        ax_peak.plot(t, [r["fft_major_peak"] for r in rows], label=name, color=color, linewidth=1)
        peak_t = [r["t"] for r in rows if r["sample_peak"]]
        if peak_t:
            ax_peak.scatter(peak_t, [0] * len(peak_t), marker="|", color=color, s=200, label=f"{name} beat")

        ax_env.plot(t, [r["sample_smth"] for r in rows], label=name, color=color, linewidth=1)

        bins = np.array([[r[f"bin{b}"] for b in range(NUM_BINS)] for r in rows]).T
        extent = [t[0], t[-1], 0, NUM_BINS]
        bin_axes[i].imshow(bins, aspect="auto", origin="lower", extent=extent, cmap="magma", vmin=0, vmax=255)
        bin_axes[i].set_ylabel(f"{name}\nGEQ bin")

    ax_peak.set_ylabel("FFT major peak (Hz)")
    ax_peak.legend(loc="upper right", fontsize=8)
    ax_peak.set_title("WLED audiosync comparison")

    ax_env.set_ylabel("sampleSmth (envelope)")
    ax_env.legend(loc="upper right", fontsize=8)

    bin_axes[-1].set_xlabel("Time (s)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"\nWrote {out_path}")


def main():
    args = parse_args()
    labels = {}
    for entry in args.label:
        if "=" not in entry:
            print(f"Ignoring malformed --label {entry!r} (expected IP=NAME)", file=sys.stderr)
            continue
        ip, name = entry.split("=", 1)
        labels[ip] = name

    local_ips = get_local_ips()

    out_dir = args.out_dir or f"wled-compare-{datetime.datetime.now():%Y%m%d-%H%M%S}"
    os.makedirs(out_dir, exist_ok=True)

    records, malformed = capture(args.port, args.duration)

    if not records:
        print("No packets received.")
        return

    print_summary(records, labels, local_ips)
    if malformed:
        print(f"  ({malformed} malformed/unrecognised packets ignored)")

    csv_paths = write_csvs(records, labels, local_ips, out_dir)
    for ip, path in csv_paths.items():
        print(f"  {label_for(ip, labels, local_ips):20s} -> {path}")

    plot_comparison(records, labels, local_ips, os.path.join(out_dir, "comparison.png"))


if __name__ == "__main__":
    main()
