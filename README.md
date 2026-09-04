# UR5 LeRobot Data Collection

This is a ROS2 node for collecting LeRobot data using UR5 robot arm.

## Prerequisites

Install the UR ROS2 driver:
```bash
sudo apt-get install ros-humble-ur
```

## Network Setup

I set my PC's IP address to `192.168.1.199`. Your robot ip setup may vary.

For detailed UR robot network setup and configuration, follow the official documentation:
https://docs.ros.org/en/ros2_packages/humble/api/ur_robot_driver/doc/installation/toc.html

## Usage

### Launch Controller for Real Robot

```bash
ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur5 robot_ip:=192.168.1.100
```

Replace `192.168.1.100` with your robot's IP address.

### Launch MoveIt Controller

```bash
ros2 launch ur_moveit_config ur_moveit.launch.py ur_type:=ur5
```

### Use Fake Hardware (for testing without physical robot)

```bash
ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur5e robot_ip:=yyy.yyy.yyy.yyy use_fake_hardware:=true launch_rviz:=true
```

## Dataset post-processing scripts

`scripts/` holds standalone Python tools for datasets written by the
`data_collect` node. They have no ROS dependency; run them in the `lerobot`
conda env (needs `pandas`, `pyarrow`, and `ffmpeg` with NVENC).

### Encode images to video

`data_collect` stores frames as PNG folders (`dtype: image`). Training
expects `dtype: video`. This encodes every episode and camera with NVENC,
drops the image byte columns from the parquet files, and rewrites
`meta/info.json`. Output goes to `<dataset>_encoded/` next to the source.

```bash
cd /path/to/datasets            # directory that contains apple_to_basket/
python /path/to/scripts/encode_videos.py apple_to_basket
python /path/to/scripts/encode_videos.py apple_to_basket --encoder av1_nvenc
python /path/to/scripts/encode_videos.py /abs/path/apple_to_basket   # or a full path
```

`--root DIR` resolves a bare folder name under `DIR` instead of the current
directory. `--no-convert-parquet` keeps the original parquet files.

### Combine encoded datasets

Merges every `*_encoded` folder in a directory into one dataset with globally
remapped episode, frame and task indices. All sources must share features and
fps. Folders matching `*combined*` are skipped by default so an earlier merge
is not merged again.

```bash
cd /path/to/datasets
python /path/to/scripts/combine_encoded.py --dry-run
python /path/to/scripts/combine_encoded.py --output 3_combined_encoded --chunks-size 1000
python /path/to/scripts/combine_encoded.py --root /path/to/datasets --exclude '*combined*' 'test*'
```

### Shift actions to next state

Rewrites a dataset so `action[t] = observation.state[t+1]` for the arm dims
and `action[t][6] = action[t+1][6]` for the gripper (next command intent).
The last frame of every episode is dropped. Videos are hard-linked, parquets
rewritten, meta recomputed.

```bash
python /path/to/scripts/convert_action.py /path/to/apple_to_basket_encoded /path/to/apple_to_basket_encoded_shifted
```

### Reorder joints (old recordings only)

Early datasets stored joints as
`[shoulder_lift, elbow, wrist_1, wrist_2, wrist_3, shoulder_pan, gripper]`.
The current order is `[shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2,
wrist_3, gripper]`. This copies each old dataset to `<name>_reordered/` and
reindexes `observation.state`, `action`, `stats.json` and
`episodes_stats.jsonl`. Datasets already in the new order are skipped.

```bash
cd /path/to/datasets
python /path/to/scripts/reorder_joints.py --dry-run            # every sub-directory
python /path/to/scripts/reorder_joints.py dataset_small_to_orange
python /path/to/scripts/reorder_joints.py --root /path/to/datasets
```
