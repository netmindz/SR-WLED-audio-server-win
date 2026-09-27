#!/usr/bin/env python3
"""
Runs a battery of synthetic test signals out this machine's line-out (so both
this app's loopback capture and a real WLED device on line-in receive the
same signal), capturing and comparing each one with compare_wled.py.

Prerequisites - both of these should already be running before you start this:
  - The real WLED device: powered on, AudioReactive enabled, in Send mode,
    line-in connected to this machine's line-out.
  - This app's Linux CLI: `dotnet run --project source/WledSRServer.Cli`,
    pointed wherever this script's listener will run.

This script only plays test signals and captures/compares the UDP output for
each one - it doesn't start or stop either sender.

Usage:
    python3 tools/run_test_matrix.py --port 11988 --out-dir results/run1
    python3 tools/run_test_matrix.py --list
    python3 tools/run_test_matrix.py --only tone_1000hz sweep_20_15000hz
"""
import argparse
import os
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import testsignals as sig

SR = 48000

# name, signal generator, nominal duration (s)
TEST_MATRIX = [
    ("silence", lambda: sig.silence(SR, 5), 5),
    ("tone_100hz", lambda: sig.tone(SR, 5, 100), 5),
    ("tone_1000hz", lambda: sig.tone(SR, 5, 1000), 5),
    ("tone_8000hz", lambda: sig.tone(SR, 5, 8000), 5),
    ("two_tone_440_880hz", lambda: sig.two_tone(SR, 5, 440, 880), 5),
    ("sweep_20_15000hz", lambda: sig.log_sweep(SR, 15, 20, 15000), 15),
    ("click_120bpm", lambda: sig.click_train(SR, 10, bpm=120), 10),
    ("tremolo_440hz_4hz", lambda: sig.tremolo(SR, 6, 440, 4), 6),
    ("white_noise", lambda: sig.white_noise(SR, 5), 5),
    ("pink_noise", lambda: sig.pink_noise(SR, 5), 5),
]


def run_one(name, gen_fn, duration, port, out_dir, extra_label_args):
    print(f"\n=== {name} ({duration}s) ===")
    samples = gen_fn()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name
    sig.write_wav(wav_path, samples, SR)

    case_dir = os.path.join(out_dir, name)
    os.makedirs(case_dir, exist_ok=True)

    capture_duration = duration + 2  # trailing buffer for pipeline/network latency
    compare_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "compare_wled.py")
    compare_cmd = [
        sys.executable, compare_script,
        "--port", str(port), "--duration", str(capture_duration), "--out-dir", case_dir,
    ] + extra_label_args

    proc = subprocess.Popen(compare_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    time.sleep(0.5)  # let it bind the socket before audio starts

    try:
        subprocess.run(["paplay", wav_path], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as ex:
        print(f"  Warning: couldn't play {wav_path} with paplay ({ex}); "
              f"capture will just record silence/whatever's already playing.", file=sys.stderr)

    proc.wait()
    print(proc.stdout.read())
    os.unlink(wav_path)
    return case_dir


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=11988, help="UDP port both senders are using (default: 11988)")
    p.add_argument("--out-dir", default=None, help="Where per-test-case results go (default: ./results-<timestamp>/)")
    p.add_argument("--label", action="append", default=[], metavar="IP=NAME",
                   help="Passed through to compare_wled.py. Repeatable.")
    p.add_argument("--only", nargs="*", metavar="NAME", help="Only run these test case(s)")
    p.add_argument("--list", action="store_true", help="List available test cases and exit")
    args = p.parse_args()

    if args.list:
        for name, _gen, duration in TEST_MATRIX:
            print(f"  {name:24s} ({duration}s)")
        return

    matrix = TEST_MATRIX
    if args.only:
        known = {name for name, _gen, _dur in TEST_MATRIX}
        unknown = set(args.only) - known
        if unknown:
            print(f"Unknown test case(s): {', '.join(sorted(unknown))}", file=sys.stderr)
        matrix = [t for t in TEST_MATRIX if t[0] in args.only]

    if not matrix:
        print("Nothing to run.", file=sys.stderr)
        sys.exit(1)

    out_dir = args.out_dir or f"results-{time.strftime('%Y%m%d-%H%M%S')}"
    os.makedirs(out_dir, exist_ok=True)

    extra_label_args = []
    for entry in args.label:
        extra_label_args += ["--label", entry]

    results = []
    for name, gen_fn, duration in matrix:
        case_dir = run_one(name, gen_fn, duration, args.port, out_dir, extra_label_args)
        results.append((name, case_dir))

    print("\n=== Done ===")
    for name, case_dir in results:
        print(f"  {name:24s} -> {case_dir}/comparison.png")


if __name__ == "__main__":
    main()
