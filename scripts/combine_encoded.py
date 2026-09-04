#!/usr/bin/env python3
"""Combine all *_encoded LeRobot datasets (video dtype) in a directory into one merged dataset.

Episode, frame and task indices are remapped globally; videos are copied,
parquets rewritten, meta/*.jsonl and info.json regenerated. All sources must
share the same features and fps (the first one found is used as reference).

Usage:
    python combine_encoded.py [--root DIR] [--output OUTPUT] [--chunks-size N] [--dry-run]

--root is the directory that holds the *_encoded folders (default: current
working directory). --output is created inside --root.

Examples:
    cd all_datasets/3_std_datasets
    python /path/to/scripts/combine_encoded.py --dry-run
    python /path/to/scripts/combine_encoded.py --output 3_combined_encoded
    python /path/to/scripts/combine_encoded.py --root ~/work/all_datasets/3_std_datasets --chunks-size 500
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, default=Path.cwd(), help="Directory containing the *_encoded datasets (default: cwd)")
parser.add_argument("--output", default="combined_encoded", help="Output dataset folder name, created under --root")
parser.add_argument("--chunks-size", type=int, default=1000, help="Episodes per chunk (default: 1000)")
parser.add_argument("--exclude", nargs="*", default=["*combined*"], help="Glob patterns of *_encoded folders to skip (default: *combined*, so a previous merge is not re-merged)")
parser.add_argument("--dry-run", action="store_true", help="Print plan without writing anything")
args = parser.parse_args()

ROOT = args.root.resolve()
DST = ROOT / args.output

# --- discover encoded datasets ---
candidates = sorted(
    p for p in ROOT.glob("*_encoded")
    if p.is_dir() and p != DST and not any(p.match(pat) for pat in args.exclude)
)
sources = []
for p in candidates:
    info_path = p / "meta" / "info.json"
    if not info_path.exists():
        continue
    with open(info_path) as f:
        info = json.load(f)
    image_keys = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
    if not image_keys:
        continue
    sources.append((p, info))

if not sources:
    print("No encoded datasets found.")
    sys.exit(1)

print(f"Found {len(sources)} encoded dataset(s):")
for p, info in sources:
    print(f"  {p.name}: {info['total_episodes']} eps, tasks: ", end="")
    tasks_path = p / "meta" / "tasks.jsonl"
    tasks = [json.loads(l)["task"] for l in tasks_path.read_text().splitlines() if l.strip()]
    print(", ".join(f'"{t}"' for t in tasks))

total_episodes = sum(info["total_episodes"] for _, info in sources)
total_chunks = (total_episodes + args.chunks_size - 1) // args.chunks_size
print(f"\nOutput  : {DST}")
print(f"Episodes: {total_episodes}  Chunks: {total_chunks}  Chunk size: {args.chunks_size}")

if args.dry_run:
    print("\n[dry-run] No files written.")
    sys.exit(0)

if DST.exists():
    print(f"\nOutput dir already exists: {DST}")
    print("Remove it manually if you want a fresh combine.")
    sys.exit(1)

DST.mkdir(parents=True)
(DST / "meta").mkdir()
(DST / "data").mkdir()
(DST / "videos").mkdir()

# --- build global task list (deduplicate, preserve order) ---
global_tasks = []       # list of task strings in insertion order
task_str_to_idx = {}    # task string -> global task_index

def get_or_add_task(task_str: str) -> int:
    if task_str not in task_str_to_idx:
        idx = len(global_tasks)
        global_tasks.append(task_str)
        task_str_to_idx[task_str] = idx
    return task_str_to_idx[task_str]

# --- iterate sources and merge ---
global_ep_idx = 0
global_frame_offset = 0
all_episodes_meta = []      # rows for episodes.jsonl
all_episodes_stats = []     # rows for episodes_stats.jsonl
total_frames = 0
total_videos = 0

# Use features from first source (assumed identical across datasets)
reference_features = sources[0][1]["features"]
fps = sources[0][1]["fps"]
robot_type = sources[0][1]["robot_type"]
codebase_version = sources[0][1]["codebase_version"]
chunks_size_src = sources[0][1]["chunks_size"]
video_keys = [k for k, v in reference_features.items() if v.get("dtype") == "video"]

for src_path, src_info in sources:
    src_eps = src_info["total_episodes"]
    src_chunks_size = src_info["chunks_size"]

    # load source task list
    tasks_path = src_path / "meta" / "tasks.jsonl"
    src_tasks = {}  # local task_index -> task string
    for line in tasks_path.read_text().splitlines():
        if line.strip():
            obj = json.loads(line)
            src_tasks[obj["task_index"]] = obj["task"]

    # load source episodes meta
    eps_meta_path = src_path / "meta" / "episodes.jsonl"
    src_eps_meta = {}
    for line in eps_meta_path.read_text().splitlines():
        if line.strip():
            obj = json.loads(line)
            src_eps_meta[obj["episode_index"]] = obj

    # load source episodes stats
    eps_stats_path = src_path / "meta" / "episodes_stats.jsonl"
    src_eps_stats = {}
    for line in eps_stats_path.read_text().splitlines():
        if line.strip():
            obj = json.loads(line)
            src_eps_stats[obj["episode_index"]] = obj

    print(f"\nMerging {src_path.name} ({src_eps} eps) -> global ep {global_ep_idx}..{global_ep_idx + src_eps - 1}")

    for local_ep in range(src_eps):
        src_chunk = local_ep // src_chunks_size
        dst_chunk = global_ep_idx // args.chunks_size

        # --- parquet ---
        src_parquet = src_path / "data" / f"chunk-{src_chunk:03d}" / f"episode_{local_ep:06d}.parquet"
        dst_parquet_dir = DST / "data" / f"chunk-{dst_chunk:03d}"
        dst_parquet_dir.mkdir(exist_ok=True)
        dst_parquet = dst_parquet_dir / f"episode_{global_ep_idx:06d}.parquet"

        df = pd.read_parquet(src_parquet)
        ep_len = len(df)

        # remap indices
        df["episode_index"] = global_ep_idx
        df["index"] = range(global_frame_offset, global_frame_offset + ep_len)

        # remap task_index
        def remap_task_idx(local_idx):
            task_str = src_tasks.get(int(local_idx), f"unknown_task_{local_idx}")
            return get_or_add_task(task_str)

        df["task_index"] = df["task_index"].apply(remap_task_idx)

        df.to_parquet(dst_parquet, index=False)

        # --- videos ---
        for vk in video_keys:
            src_vid = src_path / "videos" / f"chunk-{src_chunk:03d}" / vk / f"episode_{local_ep:06d}.mp4"
            dst_vid_dir = DST / "videos" / f"chunk-{dst_chunk:03d}" / vk
            dst_vid_dir.mkdir(parents=True, exist_ok=True)
            dst_vid = dst_vid_dir / f"episode_{global_ep_idx:06d}.mp4"
            if src_vid.exists():
                shutil.copy2(src_vid, dst_vid)
                total_videos += 1
            else:
                print(f"  [WARN] missing video: {src_vid.relative_to(src_path)}")

        # --- episodes meta ---
        src_meta = src_eps_meta.get(local_ep, {})
        src_task_strs = src_meta.get("tasks", [src_tasks.get(0, "")])
        global_task_strs = src_task_strs  # tasks are strings here, keep as-is
        all_episodes_meta.append({
            "episode_index": global_ep_idx,
            "tasks": global_task_strs,
            "length": ep_len,
        })

        # --- episodes stats (remap episode_index and index bounds) ---
        src_stat = src_eps_stats.get(local_ep, {})
        if src_stat:
            stat_copy = json.loads(json.dumps(src_stat))
            stat_copy["episode_index"] = global_ep_idx
            # update index stats
            if "stats" in stat_copy and "index" in stat_copy["stats"]:
                idx_stats = stat_copy["stats"]["index"]
                idx_stats["min"] = [global_frame_offset]
                idx_stats["max"] = [global_frame_offset + ep_len - 1]
                idx_stats["mean"] = [global_frame_offset + (ep_len - 1) / 2]
            # update episode_index stats
            if "stats" in stat_copy and "episode_index" in stat_copy["stats"]:
                ep_stats = stat_copy["stats"]["episode_index"]
                ep_stats["min"] = [global_ep_idx]
                ep_stats["max"] = [global_ep_idx]
                ep_stats["mean"] = [float(global_ep_idx)]
            all_episodes_stats.append(stat_copy)

        global_frame_offset += ep_len
        total_frames += ep_len
        global_ep_idx += 1

        if (local_ep + 1) % 10 == 0 or local_ep == src_eps - 1:
            print(f"  ep {local_ep + 1}/{src_eps} done", end="\r")
    print()

# --- write meta files ---
print("Writing meta files...")

# tasks.jsonl
with open(DST / "meta" / "tasks.jsonl", "w") as f:
    for idx, task_str in enumerate(global_tasks):
        f.write(json.dumps({"task_index": idx, "task": task_str}) + "\n")

# episodes.jsonl
with open(DST / "meta" / "episodes.jsonl", "w") as f:
    for row in all_episodes_meta:
        f.write(json.dumps(row) + "\n")

# episodes_stats.jsonl
with open(DST / "meta" / "episodes_stats.jsonl", "w") as f:
    for row in all_episodes_stats:
        f.write(json.dumps(row) + "\n")

# info.json
merged_info = {
    "codebase_version": codebase_version,
    "robot_type": robot_type,
    "total_episodes": total_episodes,
    "total_frames": total_frames,
    "total_tasks": len(global_tasks),
    "total_videos": total_videos,
    "total_chunks": total_chunks,
    "chunks_size": args.chunks_size,
    "fps": fps,
    "splits": {"train": f"0:{total_episodes}"},
    "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    "features": reference_features,
}

with open(DST / "meta" / "info.json", "w") as f:
    json.dump(merged_info, f, indent=4)

print(f"\nDone.")
print(f"  Episodes : {total_episodes}")
print(f"  Frames   : {total_frames}")
print(f"  Videos   : {total_videos}")
print(f"  Tasks    : {len(global_tasks)}")
for i, t in enumerate(global_tasks):
    print(f"    [{i}] {t}")
print(f"  Output   : {DST}")
