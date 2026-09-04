#!/usr/bin/env python3
"""Encode per-episode image folders of a LeRobot dataset to videos.

Input is a dataset as written by `data_collect` (dtype "image", frames stored
under images/<cam_key>/episode_XXXXXX/frame_XXXXXX.png). Output is written to
<dataset>_encoded/ next to the source, with dtype "video" and the image byte
columns dropped from the parquet files.

Usage:
    python encode_videos.py <dataset> [--root DIR] [--encoder h264_nvenc|av1_nvenc] [--no-convert-parquet]

<dataset> is either a path, or a folder name resolved under --root
(default: current working directory).

Examples:
    cd all_datasets/3_std_datasets
    python /path/to/scripts/encode_videos.py apple_to_basket
    python /path/to/scripts/encode_videos.py apple_to_basket --root ~/work/all_datasets/3_std_datasets
    python /path/to/scripts/encode_videos.py ~/work/all_datasets/3_std_datasets/apple_to_basket --encoder av1_nvenc

Requires ffmpeg with NVENC support, pandas, pyarrow.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pyarrow

parser = argparse.ArgumentParser()
parser.add_argument("dataset", help="Dataset path, or folder name under --root")
parser.add_argument("--root", type=Path, default=Path.cwd(), help="Directory that contains the dataset folder (default: cwd)")
parser.add_argument("--encoder", default="h264_nvenc", choices=["h264_nvenc", "av1_nvenc"])
parser.add_argument("--convert-parquet", action="store_true", default=True, help="Convert parquet files to remove image bytes columns (default: True)")
parser.add_argument("--no-convert-parquet", dest="convert_parquet", action="store_false", help="Skip parquet conversion")
args = parser.parse_args()

SRC = Path(args.dataset)
if not SRC.is_absolute() and not SRC.exists():
    SRC = args.root / args.dataset
SRC = SRC.resolve()
if not SRC.exists():
    print(f"Error: '{SRC}' not found.")
    sys.exit(1)

DST = SRC.parent / (SRC.name + "_encoded")
PIX_FMT = "yuv420p"
ENCODER = args.encoder

print(f"Source  : {SRC}")
print(f"Output  : {DST}")
print(f"Encoder : {ENCODER}")

# Copy entire dataset to dst (excluding images/ — videos replace them)
if DST.exists():
    print(f"\nOutput dir already exists: {DST}")
    print("Remove it manually if you want a clean re-encode.")
    sys.exit(1)

print("\nCopying non-image data to output dir...")
shutil.copytree(SRC, DST, ignore=shutil.ignore_patterns("images", "data"))
if not args.convert_parquet:
    print("Copying data (parquet) files...")
    shutil.copytree(SRC / "data", DST / "data")

# Load info.json from dst (already copied)
INFO_PATH = DST / "meta" / "info.json"
with open(INFO_PATH) as f:
    info = json.load(f)

fps = info["fps"]
image_keys = [k for k, v in info["features"].items() if v["dtype"] == "image"]
total_episodes = info["total_episodes"]
chunks_size = info["chunks_size"]

print(f"FPS     : {fps}")
print(f"Cameras : {image_keys}")
print(f"Episodes: {total_episodes}\n")

total_videos = 0
errors = []

for ep_idx in range(total_episodes):
    chunk_idx = ep_idx // chunks_size
    ep_str = f"episode_{ep_idx:06d}"
    chunk_str = f"chunk-{chunk_idx:03d}"

    for cam_key in image_keys:
        src_frames = SRC / "images" / cam_key / ep_str
        if not src_frames.exists():
            print(f"  [SKIP] {src_frames} not found")
            continue

        out_dir = DST / "videos" / chunk_str / cam_key
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{ep_str}.mp4"

        if out_file.exists():
            print(f"  [SKIP] {out_file.relative_to(DST)} already exists")
            total_videos += 1
            continue

        frame_pattern = str(src_frames / "frame_%06d.png")
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", frame_pattern,
            "-c:v", ENCODER,
            "-pix_fmt", PIX_FMT,
            "-preset", "p4",
            "-g", "10",
            str(out_file),
        ]

        print(f"  Encoding {cam_key}/{ep_str} ...", end=" ", flush=True)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("FAILED")
            errors.append((cam_key, ep_str, result.stderr[-500:]))
        else:
            print("OK")
            total_videos += 1

if args.convert_parquet:
    print("\nConverting parquet files...")
    parquet_errors = []
    for ep_idx in range(total_episodes):
        chunk_idx = ep_idx // chunks_size
        ep_str = f"episode_{ep_idx:06d}"
        chunk_str = f"chunk-{chunk_idx:03d}"

        src_parquet = SRC / "data" / chunk_str / f"{ep_str}.parquet"
        dst_parquet = DST / "data" / chunk_str / f"{ep_str}.parquet"

        if not src_parquet.exists():
            print(f"  [SKIP] {src_parquet.relative_to(SRC)} not found")
            continue

        if dst_parquet.exists():
            print(f"  [SKIP] {dst_parquet.relative_to(DST)} already exists")
            continue

        print(f"  Converting {ep_str} ...", end=" ", flush=True)
        try:
            df = pd.read_parquet(src_parquet)
            cols_to_drop = [col for col in df.columns if any(img_key in col for img_key in image_keys)]
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)
                print(f"(dropped {len(cols_to_drop)} columns)", end=" ")
            dst_parquet.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(dst_parquet, index=False)
            print("OK")
        except Exception as e:
            print(f"FAILED: {e}")
            parquet_errors.append((ep_str, str(e)))

    if parquet_errors:
        print(f"\n{len(parquet_errors)} parquet conversion error(s):")
        for ep, msg in parquet_errors:
            print(f"  {ep}: {msg}")
        sys.exit(1)

# Update info.json in dst
info["total_videos"] = total_videos
info["video_path"] = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"

codec = "h264" if "h264" in ENCODER else "av1"
for key in image_keys:
    feat = info["features"][key]
    h, w = feat["shape"][0], feat["shape"][1]
    feat["dtype"] = "video"
    feat["info"] = {
        "video.height": h,
        "video.width": w,
        "video.codec": codec,
        "video.pix_fmt": PIX_FMT,
        "video.is_depth_map": "depth" in key,
        "video.fps": fps,
        "video.channels": 3,
        "has_audio": False,
    }

with open(INFO_PATH, "w") as f:
    json.dump(info, f, indent=4)

print(f"\nDone. {total_videos} videos written to {DST}")
if errors:
    print(f"\n{len(errors)} error(s):")
    for cam, ep, msg in errors:
        print(f"  {cam}/{ep}:\n{msg}")
    sys.exit(1)
