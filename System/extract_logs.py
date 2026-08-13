#!/usr/bin/env python3
"""
Extract logs for 6 cameras from 20:00 on 2026-05-26 to 08:00 on 2026-05-27.

Usage:
    python3 extract_logs.py                     # Print to terminal
    python3 extract_logs.py -o output.log       # Save merged output to file
    python3 extract_logs.py --per-cam           # Save one file per camera
"""

import os
import sys
import argparse
from datetime import datetime

# ─── Configuration ────────────────────────────────────────────────────────────
LOG_BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

CAMERAS      = [f"cam_{i}" for i in range(1, 7)]   # cam_1 ... cam_6

START_TIME   = datetime(2026, 5, 26, 20, 0, 0)      # 20:00  26/05
END_TIME     = datetime(2026, 5, 27,  8, 0, 0)      # 08:00  27/05

# Log files that might contain entries in our window
LOG_FILES = [
    ("20260526.log", datetime(2026, 5, 26)),         # evening part
    ("20260527.log", datetime(2026, 5, 27)),         # early morning part
]
# ──────────────────────────────────────────────────────────────────────────────


def parse_log_datetime(line: str):
    """Parse the leading datetime from a log line, e.g. '2026-05-26 20:01:05'."""
    try:
        return datetime.strptime(line[:19], "%Y-%m-%d %H:%M:%S")
    except (ValueError, IndexError):
        return None


def extract_lines(cam: str):
    """Return all log lines for *cam* that fall in [START_TIME, END_TIME]."""
    results = []

    for filename, _ in LOG_FILES:
        filepath = os.path.join(LOG_BASE_DIR, cam, filename)
        if not os.path.isfile(filepath):
            continue

        with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
            for raw_line in fh:
                line = raw_line.rstrip("\n")
                ts = parse_log_datetime(line)
                if ts is None:
                    continue
                if START_TIME <= ts <= END_TIME:
                    results.append((ts, line))

    # Sort chronologically (in case files overlap)
    results.sort(key=lambda x: x[0])
    return results


def run(output_file=None, per_cam=False):
    print(f"[*] Log directory : {LOG_BASE_DIR}")
    print(f"[*] Time window   : {START_TIME}  ->  {END_TIME}")
    print(f"[*] Cameras       : {', '.join(CAMERAS)}")
    print()

    all_lines = []
    stats = {}

    for cam in CAMERAS:
        lines = extract_lines(cam)
        stats[cam] = len(lines)
        all_lines.extend(lines)

        if per_cam:
            cam_out = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                f"extracted_{cam}_0526_20h_to_0527_08h.log"
            )
            if lines:
                with open(cam_out, "w", encoding="utf-8") as fh:
                    fh.write(f"# Extracted: {cam} | {START_TIME} -> {END_TIME}\n")
                    for _, text in lines:
                        fh.write(text + "\n")
                print(f"  [{cam}]  {len(lines):>5} lines  ->  {os.path.basename(cam_out)}")
            else:
                print(f"  [{cam}]      0 lines  (no data in window)")
        else:
            print(f"  [{cam}]  {len(lines):>5} lines")

    total = sum(stats.values())
    print(f"\n[*] Total lines   : {total}")

    if not per_cam:
        # Merge-sort all cameras by timestamp
        all_lines.sort(key=lambda x: x[0])

        if output_file:
            with open(output_file, "w", encoding="utf-8") as fh:
                fh.write(f"# Merged log | {START_TIME} -> {END_TIME}\n")
                fh.write(f"# Cameras: {', '.join(CAMERAS)}\n\n")
                for _, text in all_lines:
                    fh.write(text + "\n")
            print(f"[*] Saved to      : {output_file}")
        else:
            print("\n" + "=" * 80)
            for _, text in all_lines:
                print(text)
            print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Extract camera logs for a specific time window."
    )
    parser.add_argument(
        "-o", "--output",
        metavar="FILE",
        default=None,
        help="Save merged output to FILE instead of printing to terminal."
    )
    parser.add_argument(
        "--per-cam",
        action="store_true",
        help="Save one output file per camera (overrides -o)."
    )
    args = parser.parse_args()

    try:
        run(output_file=args.output, per_cam=args.per_cam)
    except KeyboardInterrupt:
        print("\n[!] Interrupted.")
        sys.exit(0)


if __name__ == "__main__":
    main()
