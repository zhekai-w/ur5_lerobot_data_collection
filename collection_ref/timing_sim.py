#!/usr/bin/env python3
"""
Simulates the timing loop from ur5e_multi_cube_lerobot.py
No ROS, no cameras, no dataset — just measures actual FPS.
"""

import time
import threading
import sys
import argparse


def fake_work(work_ms: float):
    """Simulate work done each frame (ROS reads, camera grab, etc.)"""
    time.sleep(work_ms / 1000.0)


def run_loop(fps: int, work_ms: float, use_fixed_deadline: bool):
    """
    use_fixed_deadline=False: original code  (t_next = t_now + dt each iter)
    use_fixed_deadline=True:  corrected code (t_next = t0 + step_idx * dt)
    """
    dt = 1.0 / float(fps)
    print(f"\n--- {'FIXED' if use_fixed_deadline else 'ORIGINAL'} mode | target={fps} FPS | dt={dt*1000:.2f}ms | fake work={work_ms:.1f}ms ---")
    print("Press ENTER to stop recording...\n")

    stop_event = threading.Event()
    quit_event = threading.Event()

    def key_reader():
        try:
            line = sys.stdin.readline()
        except Exception:
            line = ""
        line = (line or "").strip().lower()
        if line == 'q':
            quit_event.set()
        else:
            stop_event.set()

    t_key = threading.Thread(target=key_reader, daemon=True)
    t_key.start()

    step_idx = 0
    t0 = time.time()
    frame_times = []

    while not stop_event.is_set() and not quit_event.is_set():
        t_now = time.time()
        frame_times.append(t_now)

        # Simulate work
        fake_work(work_ms)

        # Pacing
        if use_fixed_deadline:
            t_next = t0 + (step_idx + 1) * dt      # corrected: no drift
        else:
            t_next = t_now + dt                      # original: per-iter baseline

        while time.time() < t_next and not stop_event.is_set() and not quit_event.is_set():
            time.sleep(0.001)

        step_idx += 1

        # Live status every 20 frames
        if step_idx % 20 == 0:
            elapsed = time.time() - t0
            actual_fps = step_idx / elapsed if elapsed > 0 else 0
            print(f"  step={step_idx:4d} | elapsed={elapsed:.2f}s | actual FPS={actual_fps:.1f}")

    elapsed = time.time() - t0
    total_frames = len(frame_times)
    actual_fps = total_frames / elapsed if elapsed > 0 else 0

    print(f"\nResults:")
    print(f"  Total frames : {total_frames}")
    print(f"  Elapsed      : {elapsed:.3f}s")
    print(f"  Actual FPS   : {actual_fps:.2f}")
    print(f"  Target FPS   : {fps}")
    print(f"  Ratio        : {actual_fps/fps:.3f}x")

    if len(frame_times) > 1:
        intervals = [frame_times[i+1] - frame_times[i] for i in range(len(frame_times)-1)]
        mean_interval = sum(intervals) / len(intervals)
        min_interval = min(intervals)
        max_interval = max(intervals)
        print(f"  Frame interval mean={mean_interval*1000:.2f}ms min={min_interval*1000:.2f}ms max={max_interval*1000:.2f}ms")

    return actual_fps


def main():
    parser = argparse.ArgumentParser(description="FPS timing simulator")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--work_ms", type=float, default=1.0,
                        help="Simulated work per frame in ms (default 1ms)")
    parser.add_argument("--mode", choices=["original", "fixed", "both"], default="both",
                        help="Which pacing mode to test")
    args = parser.parse_args()

    print(f"Timing simulator | fps={args.fps} | work_ms={args.work_ms}")
    print("ENTER=stop, 'q'+ENTER=discard\n")

    if args.mode in ("original", "both"):
        run_loop(args.fps, args.work_ms, use_fixed_deadline=False)

    if args.mode == "both":
        again = input("\nRun fixed-deadline version? (ENTER=yes, q=skip): ").strip().lower()
        if again != 'q':
            run_loop(args.fps, args.work_ms, use_fixed_deadline=True)
    elif args.mode == "fixed":
        run_loop(args.fps, args.work_ms, use_fixed_deadline=True)


if __name__ == "__main__":
    main()
