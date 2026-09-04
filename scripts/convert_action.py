"""
Convert a LeRobot dataset so that action[t] = obs.state[t+1].

Usage:
    python convert_action.py <src_dataset_dir> <dst_dataset_dir>

Example:
    python convert_action.py \
        /home/zack/work/all_datasets/1_std_datasets/test_fruit_encoded \
        /home/zack/work/all_datasets/1_std_datasets/test_fruit_encoded_act_shifted

For each episode:
  - arm dims (0:6):  new_action[t][:6] = observation.state[t+1][:6]  (realized next arm state)
  - gripper dim (6): new_action[t][6]  = original_action[t+1][6]     (next gripper command intent)
  - last row dropped (no t+1 available)

Videos are hard-linked (no copy), parquets rewritten, meta recomputed.
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def compute_stats(arr: np.ndarray) -> dict:
    """Compute per-column stats for a (N, D) float array."""
    return {
        "min":   arr.min(axis=0).tolist(),
        "max":   arr.max(axis=0).tolist(),
        "mean":  arr.mean(axis=0).tolist(),
        "std":   arr.std(axis=0).tolist(),
        "count": [int(arr.shape[0])],
    }


def compute_stats_with_quantiles(arr: np.ndarray) -> dict:
    """Like compute_stats but also includes q01/q99 for global stats.json."""
    s = compute_stats(arr)
    s["q01"] = np.percentile(arr, 1, axis=0).tolist()
    s["q99"] = np.percentile(arr, 99, axis=0).tolist()
    return s


def unpack_column(series: pd.Series) -> np.ndarray:
    """Stack a pandas Series of numpy arrays into (N, D)."""
    return np.stack(series.values).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("src", type=Path)
    parser.add_argument("dst", type=Path)
    args = parser.parse_args()

    src: Path = args.src
    dst: Path = args.dst

    if dst.exists():
        print(f"Destination {dst} already exists. Remove it first.")
        raise SystemExit(1)

    # Copy full dataset tree (videos included).
    # Use symlinks=True so video files are hard-linked rather than duplicated.
    print(f"Copying {src} → {dst} ...")
    shutil.copytree(src, dst, symlinks=False)
    print("Copy done. Rewriting parquets...")

    data_dir = dst / "data"
    chunk_dirs = sorted(data_dir.glob("chunk-*"))

    all_actions: list[np.ndarray] = []
    all_states:  list[np.ndarray] = []
    all_timestamps: list[np.ndarray] = []
    all_frame_indices: list[np.ndarray] = []

    # (episode_index → (new_length, per-column stats dict))
    episode_meta: dict[int, dict] = {}

    global_index = 0

    for chunk_dir in chunk_dirs:
        parquet_files = sorted(chunk_dir.glob("episode_*.parquet"))
        for pf in parquet_files:
            df = pd.read_parquet(pf)

            ep_idx = int(df["episode_index"].iloc[0])
            n = len(df)

            if n < 2:
                print(f"  Episode {ep_idx}: only {n} frame(s), cannot shift — skipping (episode dropped).")
                # Write empty parquet so file still exists but has 0 rows.
                df.iloc[:0].to_parquet(pf, index=False)
                episode_meta[ep_idx] = {"length": 0, "stats": {}}
                continue

            states  = unpack_column(df["observation.state"])   # (N, 7)
            actions = unpack_column(df["action"])               # (N, 7)

            # action[t] = [arm_state[t+1], gripper_cmd[t+1]]
            new_arm    = states[1:, :6]     # (N-1, 6)
            new_grip   = actions[1:, 6:7]   # (N-1, 1)  original gripper command shifted +1
            new_action = np.concatenate([new_arm, new_grip], axis=1).astype(np.float32)  # (N-1, 7)

            # Drop last row
            df = df.iloc[:-1].copy()
            assert len(df) == n - 1

            # Assign shifted actions
            df["action"] = list(new_action)

            # Reset frame_index to 0-based within episode
            df["frame_index"] = np.arange(len(df), dtype=np.int64)

            # Assign global index
            df["index"] = np.arange(global_index, global_index + len(df), dtype=np.int64)
            global_index += len(df)

            df.to_parquet(pf, index=False)

            # Accumulate for global stats
            ep_states = unpack_column(df["observation.state"])
            all_actions.append(new_action)
            all_states.append(ep_states)
            all_timestamps.append(df["timestamp"].values.astype(np.float32).reshape(-1, 1))
            all_frame_indices.append(df["frame_index"].values.astype(np.int64).reshape(-1, 1))

            # Per-episode stats (all columns that have float arrays)
            ep_stats: dict[str, dict] = {}
            for col, arr in [("observation.state", ep_states), ("action", new_action)]:
                ep_stats[col] = compute_stats(arr)

            episode_meta[ep_idx] = {"length": len(df), "stats": ep_stats}

            print(f"  Episode {ep_idx}: {n} → {len(df)} rows  (dropped last frame)")

    # Rewrite episodes.jsonl (update length per episode)
    ep_jsonl_src = src / "meta" / "episodes.jsonl"
    ep_jsonl_dst = dst / "meta" / "episodes.jsonl"
    new_ep_lines = []
    with open(ep_jsonl_src) as f:
        for line in f:
            rec = json.loads(line)
            ep_i = rec["episode_index"]
            if ep_i in episode_meta:
                rec["length"] = episode_meta[ep_i]["length"]
            new_ep_lines.append(json.dumps(rec))
    ep_jsonl_dst.write_text("\n".join(new_ep_lines) + "\n")

    # Rewrite episodes_stats.jsonl
    ep_stats_src = src / "meta" / "episodes_stats.jsonl"
    ep_stats_dst = dst / "meta" / "episodes_stats.jsonl"
    new_ep_stats_lines = []
    with open(ep_stats_src) as f:
        for line in f:
            rec = json.loads(line)
            ep_i = rec["episode_index"]
            if ep_i in episode_meta and episode_meta[ep_i]["stats"]:
                # Merge: update action + observation.state stats, keep others
                for col, s in episode_meta[ep_i]["stats"].items():
                    rec["stats"][col] = s
            new_ep_stats_lines.append(json.dumps(rec))
    ep_stats_dst.write_text("\n".join(new_ep_stats_lines) + "\n")

    # Recompute global stats.json
    stats_dst = dst / "meta" / "stats.json"
    with open(src / "meta" / "stats.json") as f:
        orig_stats = json.load(f)

    all_actions_flat = np.concatenate(all_actions, axis=0)  # (total_frames, 7)
    all_states_flat  = np.concatenate(all_states,  axis=0)

    orig_stats["action"]            = compute_stats_with_quantiles(all_actions_flat)
    orig_stats["observation.state"] = compute_stats_with_quantiles(all_states_flat)

    # timestamp + frame_index stats — small change due to dropped last frame
    all_ts = np.concatenate(all_timestamps, axis=0)
    all_fi = np.concatenate(all_frame_indices, axis=0).astype(np.float32)
    if "timestamp" in orig_stats:
        orig_stats["timestamp"] = compute_stats_with_quantiles(all_ts)
    if "frame_index" in orig_stats:
        orig_stats["frame_index"] = compute_stats_with_quantiles(all_fi)

    stats_dst.write_text(json.dumps(orig_stats, indent=4) + "\n")

    # Update info.json total_frames
    info_dst = dst / "meta" / "info.json"
    with open(info_dst) as f:
        info = json.load(f)
    info["total_frames"] = global_index
    info_dst.write_text(json.dumps(info, indent=4) + "\n")

    print(f"\nDone. {global_index} frames written to {dst}")
    print("Verify with:")
    print(f"  python3 -c \"\nimport pandas as pd, numpy as np")
    print(f"  df0 = pd.read_parquet('{dst}/data/chunk-000/episode_000000.parquet')")
    print(f"  df0s = pd.read_parquet('{src}/data/chunk-000/episode_000000.parquet')")
    print(f"  act = np.stack(df0['action'].values)")
    print(f"  obs = np.stack(df0s['observation.state'].values)")
    print(f"  assert np.allclose(act[:, :6], obs[1:len(act)+1, :6], atol=1e-5), 'arm mismatch'")
    print(f"  print('arm OK')\"")


if __name__ == "__main__":
    main()
