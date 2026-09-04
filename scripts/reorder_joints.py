#!/usr/bin/env python3
"""
Reorder joint indices in LeRobot datasets recorded with the old joint order.

Current order: [shoulder_lift, elbow, wrist_1, wrist_2, wrist_3, shoulder_pan, gripper]
Target order:  [shoulder_pan,  shoulder_lift, elbow, wrist_1, wrist_2, wrist_3, gripper]
Reindex:       [5, 0, 1, 2, 3, 4, 6]

Copies each dataset to <name>_reordered/ next to it and applies the reordering
there. Originals are left untouched. Datasets already in the target order, or in
an unexpected order, are skipped.

Usage:
    python reorder_joints.py [DATASET ...] [--root DIR] [--dry-run]

With no DATASET arguments, every sub-directory of --root (default: current
working directory) is processed. A DATASET is a path, or a folder name under
--root.

Examples:
    cd all_datasets/old_recordings
    python /path/to/scripts/reorder_joints.py --dry-run
    python /path/to/scripts/reorder_joints.py dataset_small_to_orange
    python /path/to/scripts/reorder_joints.py --root ~/work/all_datasets/old_recordings
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

CURRENT_ORDER = [
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
    "shoulder_pan_joint",
    "gripper_joint",
]

TARGET_ORDER = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
    "gripper_joint",
]

# Build reindex from current → target
REINDEX = [CURRENT_ORDER.index(name) for name in TARGET_ORDER]


def reorder_array_column(series: pd.Series) -> pd.Series:
    return series.apply(lambda arr: np.array(arr, dtype=np.float32)[REINDEX])


def process_dataset(dataset_path: Path, dry_run: bool = False):
    info_path = dataset_path / "meta" / "info.json"
    data_dir = dataset_path / "data"

    if not info_path.exists() or not data_dir.exists():
        print(f"  [SKIP] Missing meta/info.json or data/ in {dataset_path.name}")
        return

    # --- Verify current joint order matches expectation ---
    with open(info_path) as f:
        info = json.load(f)

    current_names = info["features"]["observation.state"]["names"]
    if current_names == TARGET_ORDER:
        print(f"  [SKIP] {dataset_path.name} already in target order")
        return
    if current_names != CURRENT_ORDER:
        print(f"  [WARN] {dataset_path.name} has unexpected order: {current_names}")
        print(f"         Expected: {CURRENT_ORDER}")
        print(f"         Skipping to avoid data corruption.")
        return

    # --- Copy dataset to <name>_reordered/ ---
    dest_path = dataset_path.parent / f"{dataset_path.name}_reordered"
    if dest_path.exists():
        print(f"  [SKIP] {dest_path.name} already exists, skipping copy")
        return
    if dry_run:
        print(f"  [DRY-RUN] would copy to {dest_path.name} and reorder")
        return
    print(f"  Copying to {dest_path.name}...")
    shutil.copytree(dataset_path, dest_path)

    # Work on the copy from here
    info_path = dest_path / "meta" / "info.json"
    data_dir = dest_path / "data"

    # --- Reorder parquet files in the copy ---
    parquet_files = sorted(data_dir.rglob("*.parquet"))
    print(f"  Reordering {len(parquet_files)} parquet files...")
    for pq_path in parquet_files:
        df = pd.read_parquet(pq_path)
        df["observation.state"] = reorder_array_column(df["observation.state"])
        df["action"] = reorder_array_column(df["action"])
        df.to_parquet(pq_path, index=False)

    # --- Update info.json in the copy ---
    with open(info_path) as f:
        info = json.load(f)
    info["features"]["observation.state"]["names"] = TARGET_ORDER
    info["features"]["action"]["names"] = TARGET_ORDER
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    # --- Reorder stats.json ---
    stats_path = dest_path / "meta" / "stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
        for key in ("observation.state", "action"):
            if key in stats:
                for stat in ("min", "max", "mean", "std", "q01", "q99"):
                    if stat in stats[key]:
                        arr = stats[key][stat]
                        if isinstance(arr, list) and len(arr) == len(REINDEX):
                            stats[key][stat] = [arr[i] for i in REINDEX]
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

    # --- Reorder episodes_stats.jsonl ---
    ep_stats_path = dest_path / "meta" / "episodes_stats.jsonl"
    if ep_stats_path.exists():
        lines = ep_stats_path.read_text().splitlines()
        rewritten = []
        for line in lines:
            ep = json.loads(line)
            for key in ("observation.state", "action"):
                if key in ep.get("stats", {}):
                    for stat in ("min", "max", "mean", "std", "q01", "q99"):
                        if stat in ep["stats"][key]:
                            arr = ep["stats"][key][stat]
                            if isinstance(arr, list) and len(arr) == len(REINDEX):
                                ep["stats"][key][stat] = [arr[i] for i in REINDEX]
            rewritten.append(json.dumps(ep))
        ep_stats_path.write_text("\n".join(rewritten) + "\n")

    print(f"  [DONE] {dest_path.name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("datasets", nargs="*", help="Dataset paths or folder names under --root (default: all sub-directories of --root)")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Directory containing the datasets (default: cwd)")
    parser.add_argument("--dry-run", action="store_true", help="Report what would be done without copying or writing")
    args = parser.parse_args()

    root = args.root.resolve()
    if args.datasets:
        datasets = []
        for d in args.datasets:
            p = Path(d)
            if not p.is_absolute() and not p.exists():
                p = root / d
            datasets.append(p.resolve())
    else:
        datasets = sorted(p for p in root.iterdir() if p.is_dir())

    print(f"Reindex mapping: {REINDEX}")
    print(f"  (e.g. new[0]=old[{REINDEX[0]}] = {CURRENT_ORDER[REINDEX[0]]})")
    print(f"Found {len(datasets)} dataset(s) under {root}\n")

    for dataset_path in datasets:
        print(f"Processing: {dataset_path.name}")
        process_dataset(dataset_path, dry_run=args.dry_run)

    print("\nAll done.")


if __name__ == "__main__":
    main()
