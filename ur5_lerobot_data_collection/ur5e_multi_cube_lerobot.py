#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re
import sys
import time
import json
import argparse
import threading
from typing import List, Tuple, Optional

import numpy as np
import cv2
import av
import shutil
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, Float32
from control_msgs.action import GripperCommand

# ---- Imports for user's bridges (absolute or package-relative) ----
try:
    from ros2_bridge import Ros2Bridge  # user-provided
except Exception:
    # Package-style fallback (if this file is inside a package)
    from .ros2_bridge import Ros2Bridge  # type: ignore

# If user's ros2_bridge doesn't export CommandTap, define a minimal version
class CommandTap(Node):
    def __init__(self, arm_topic: str, gripper_state_topic: str):
        super().__init__('command_tap')
        self._last = np.zeros(6, dtype=np.float64)
        self._lock = threading.Lock()
        self.create_subscription(Float64MultiArray, arm_topic, self.arm_cb, 10)
        self.get_logger().info(f"[CommandTap] Listening on {arm_topic}")
        self.create_subscription(Float32, gripper_state_topic, self.gripper_cb, 10)
        self.get_logger().info(f"[CommandTap] Listening on {gripper_state_topic}")
    def arm_cb(self, msg: Float64MultiArray):
        arr = np.array(list(msg.data), dtype=np.float64)
        if arr.size < 6:
            pad = np.zeros(6, dtype=np.float64); pad[:arr.size] = arr; arr = pad
        else:
            arr = arr[:6]
        with self._lock:
            self.arm_last = arr
    def gripper_cb(self, msg: Float32):
        arr = np.zeros(6, dtype=np.float64)
        # if arr.size < 6:
        #     # pad = np.zeros(6, dtype=np.float64); pad[:arr.size] = arr; arr = pad
        #     arr = arr*6
        # else:
        #     arr = arr[:6]
        # with self._lock:
        #     self.gripper_last = arr
            
        try:
            val = float(msg.data)  # typical field
        except Exception:
            # Fallback: best-effort 0.0 if unavailable
            val = 0.0
        arr[:6] = val
        with self._lock:
            self.gripper_last = arr

    def get_last_cmd(self) -> np.ndarray:
        with self._lock:
            return self.arm_last.copy(), self.gripper_last.copy()

try:
    from sensor_bridge import MultiRealSense
except Exception:
    from .sensor_bridge import MultiRealSense  # type: ignore

# ---- LeRobot API ----
from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

"""
    export HF_LEROBOT_HOME=/workspace/datasets/lerobot/ur5e_pick_place_multi_cube

    # ./isaaclab.sh -p scripts/environments/state_machine/pickplace_cube_sm_lerobot_joint.py --num_envs 10 --total_data_num 10 --seed 42 --enable_cameras --headless
    ./isaaclab.sh -p scripts/tools/pickplace_multi_cube_ur5e_lerobot_joint.py --num_envs 10 --total_data_num 10 --seed 42 --enable_cameras --target_color blue --place_spot_color red --repo_name robot_sim.PickNPlaceCubeUR5eMulti_10 --headless
    ./isaaclab.sh -p scripts/tools/pickplace_multi_cube_ur5e_lerobot_joint.py --num_envs 1 --total_data_num 10 --seed 42 --enable_cameras --target_color blue --place_spot_color red --repo_name robot_sim.PickNPlaceCubeUR5eMulti_b2r --task_description "pick the blue cube and place to the red spot" --headless
"""

def _ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def build_dataset(repo_name: str,
                  cam_h: int, cam_w: int,
                  joints_num: int, fps: int, enable_depth: bool) -> LeRobotDataset:
    motor_names = [f"motor_{i}" for i in range(joints_num)]

    if enable_depth:
        datasets = LeRobotDataset.create(
            repo_id=repo_name,
            video_backend="torchcodec",
            # video_backend="pyav",
            # video_backend="decord",
            # video_backend="torchvision_av",
            robot_type="ur5e",
            # use_videos=False,
            fps=fps,
            features={
                "observation.images.exterior_image": {
                    "dtype": "video", "shape": (cam_h, cam_w, 3), "names": ["height", "width", "rgb"]
                },
                "observation.images.wrist_image": {
                    "dtype": "video", "shape": (cam_h, cam_w, 3), "names": ["height", "width", "rgb"]
                },
                "observation.images.exterior_depth_image": {
                    "dtype": "video", "shape": (cam_h, cam_w, 3), "names": ["height", "width", "depth"]
                },
                "observation.images.wrist_depth_image": {
                    "dtype": "video", "shape": (cam_h, cam_w, 3), "names": ["height", "width", "depth"]
                },
                "observation.state": {"dtype": "float64", "shape": (joints_num,), "names": motor_names},
                "action": {"dtype": "float64", "shape": (joints_num,), "names": motor_names},
                "annotation.human.action.task_description": {"dtype": "int64", "shape": (1,)},
                "next.reward": {"dtype": "float64", "shape": (1,)},
                "next.done": {"dtype": "bool", "shape": (1,)},
            },
            image_writer_threads=10,
        )
    else:
        datasets = LeRobotDataset.create(
            repo_id=repo_name,
            video_backend="torchcodec",
            # video_backend="pyav",
            # video_backend="decord",
            # video_backend="torchvision_av",
            robot_type="ur5e",
            # use_videos=False,
            fps=fps,
            features={
                "observation.images.exterior_image": {
                    "dtype": "video", "shape": (cam_h, cam_w, 3), "names": ["height", "width", "rgb"]
                },
                # "observation.images.exterior_image": {
                #     "dtype": "image", "shape": (cam_h, cam_w, 3), "names": ["height", "width", "rgb"]
                # },
                # "observation.images.exterior_segmentation_image": {
                #     "dtype": "video", "shape": (cam_h, cam_w, 3), "names": ["height", "width", "rgb"]
                # },
                "observation.images.wrist_image": {
                    "dtype": "video", "shape": (cam_h, cam_w, 3), "names": ["height", "width", "rgb"]
                },
                # "observation.images.wrist_image": {
                #     "dtype": "image", "shape": (cam_h, cam_w, 3), "names": ["height", "width", "rgb"]
                # },
                # "observation.images.wrist_segmentation_image": {
                #     "dtype": "video", "shape": (cam_h, cam_w, 3), "names": ["height", "width", "rgb"]
                # },

                "observation.state": {"dtype": "float64", "shape": (joints_num,), "names": motor_names},
                "action": {"dtype": "float64", "shape": (joints_num,), "names": motor_names},
                "annotation.human.action.task_description": {"dtype": "int64", "shape": (1,)},
                "next.reward": {"dtype": "float64", "shape": (1,)},
                "next.done": {"dtype": "bool", "shape": (1,)},
            },
            image_writer_threads=10,
        )
    print("LeRobot video_backend:", datasets.video_backend)
    return datasets

def create_modality(args):
    # --- NEW: generate modality.json ---c
    REPO_NAME = args.repo_name
    OUTPUT_DIR = HF_LEROBOT_HOME / REPO_NAME

    meta_dir = OUTPUT_DIR / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    modalities = {
        "state": {
            "robot_arm": {"start": 0, "end": 6},
            "gripper": {"start": 6, "end": 12},
        },
        "action": {
            "joint_action": {"start": 0, "end": 6},
            "gripper": {"start": 6, "end": 12},
        },
        "video": {
            "exterior_image": {
                "original_key": "observation.images.exterior_image"
            },
            "wrist_image": {
                "original_key": "observation.images.wrist_image"
            },
        },
        "annotation": {
            "human.action.task_description": {},
        }
    }

    with open(meta_dir / "modality.json", "w") as fp:
        json.dump(modalities, fp, indent=4)

def main():
    parser = argparse.ArgumentParser(description="Collect dataset to LeRobot format (real-world UR5e)")
    parser.add_argument("--repo_name", type=str, default="robot_real.PickNPlaceCubeUR5eMulti",
                        help="HF repo-style dataset name under $HF_LEROBOT_HOME")
    parser.add_argument("--joints_num", type=int, default=12, help="Dimension of state/action (pad/truncate)")
    parser.add_argument("--fps", type=int, default=20, help="Recording FPS")
    parser.add_argument("--task_description", type=str, default="pick the blue cube and place to the red spot")
    parser.add_argument("--validity", type=int, default=1)

    # Cameras
    parser.add_argument("--serials_list", type=list, default = ["117122250090", "043422251770"],
                        help="RealSense serials: first=wrist, second=exterior") #d455 wrist, d455
    parser.add_argument("--camera_width", type=int, default=640)
    parser.add_argument("--camera_height", type=int, default=360)
    parser.add_argument("--camera_fps", type=int, default=30)
    parser.add_argument("--enable_depth", type=bool, default = False)
    # parser.add_argument("--enable_depth", action="store_true")

    # Robot homing
    parser.add_argument("--go_initial", action="store_true", default=True)

    args = parser.parse_args()
    repo_name = args.repo_name

    output_dir = (HF_LEROBOT_HOME / repo_name)

    print(f"[INFO] Writing to dataset: {output_dir}")
    if output_dir.exists():
        shutil.rmtree(output_dir)

    # Dataset construction
    
    dataset = build_dataset(
        repo_name=repo_name,
        cam_h=args.camera_height,
        cam_w=args.camera_width,
        joints_num=args.joints_num,
        fps=args.fps,
        enable_depth=args.enable_depth,
    )


    # ROS bring-up
    rclpy.init()
    ros2_bridge = Ros2Bridge()
    cmd_tap = CommandTap(arm_topic=ros2_bridge.ur_controller_topic_name, 
                         gripper_state_topic=ros2_bridge.gripper_action_state_topic)

    executor = MultiThreadedExecutor()
    executor.add_node(ros2_bridge)
    executor.add_node(cmd_tap)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    # Cameras
    cams = MultiRealSense(args.serials_list, 
                          args.camera_width, 
                          args.camera_height, 
                          args.camera_fps,
                          args.enable_depth,
                          log_fn=ros2_bridge.get_logger().info,
                          warn_fn=ros2_bridge.get_logger().warning)
    cams.start()
    # cams = MultiRealSense(
    #     serials=args.serials_list,
    #     width=args.camera_width,
    #     height=args.camera_height,
    #     fps=args.camera_fps,
    #     enable_depth=args.enable_depth,
    #     log_fn=ros2_bridge.get_logger().info,
    #     warn_fn=ros2_bridge.get_logger().warning,
    # )
    # cams.start()
    time.sleep(1.0)

    # Helpers
    def get_images_pair() -> Tuple[np.ndarray, np.ndarray]:
        # frames = cams.get_rgb_frames_list()
        # if len(frames) == 0:
        #     fake = np.random.randint(0, 255, (args.camera_height, args.camera_width, 3), dtype=np.uint8)
        #     return fake, fake
        # if len(frames) == 1:
        #     img = frames[0]
        #     if img is None:
        #         img = np.zeros((args.camera_height, args.camera_width, 3), dtype=np.uint8)
        #     return img, img
        # wrist = frames[0] if frames[0] is not None else np.zeros((args.camera_height, args.camera_width, 3), dtype=np.uint8)
        # ext = frames[1] if frames[1] is not None else np.zeros((args.camera_height, args.camera_width, 3), dtype=np.uint8)
        # # Ensure shape
        # if wrist.shape[:2] != (args.camera_height, args.camera_width):
        #     wrist = cv2.resize(wrist, (args.camera_width, args.camera_height), interpolation=cv2.INTER_AREA)
        # if ext.shape[:2] != (args.camera_height, args.camera_width):
        #     ext = cv2.resize(ext, (args.camera_width, args.camera_height), interpolation=cv2.INTER_AREA)
        # return wrist, ext
        if cams is None:
            fake = np.random.randint(0, 255, (args.camera_height, args.camera_width, 3), dtype=np.uint8)
            return fake, fake
        frames = cams.get_frames_by_serial()
        imgs = []
        for s in args.serials_list:
            frm = frames.get(s, None)
            if frm is None:
                frm = np.zeros((args.camera_height, args.camera_width, 3), dtype=np.uint8)
            else:
                if frm.shape[0] != args.camera_height or frm.shape[1] != args.camera_width:
                    frm = cv2.resize(frm, (args.camera_width, args.camera_height), interpolation=cv2.INTER_AREA)
            imgs.append(frm)
        if len(imgs) == 0:
            fake = np.random.randint(0, 255, (args.camera_height, args.camera_width, 3), dtype=np.uint8)
            return fake, fake
        if len(imgs) == 1:
            return imgs[0], imgs[0]
        return imgs[0], imgs[1]
    
    def get_depth_pair() -> Tuple[np.ndarray, np.ndarray]:
        if cams is None:
            fake = np.zeros((args.camera_height, args.camera_width), dtype=np.uint16)
            return fake, fake

        frames = cams.get_depth_by_serial()
        imgs = []
        for s in args.serials_list:
            frm = frames.get(s, None)
            if frm is None:
                frm = np.zeros((args.camera_height, args.camera_width), dtype=np.uint16)
            else:
                if frm.shape[0] != args.camera_height or frm.shape[1] != args.camera_width:
                    frm = cv2.resize(
                        frm,
                        (args.camera_width,
                        args.camera_height),
                        interpolation=cv2.INTER_NEAREST,  # important for depth
                    )
            imgs.append(frm)

        if len(imgs) == 0:
            fake = np.zeros((args.camera_height, args.camera_width), dtype=np.uint16)
            return fake, fake
        if len(imgs) == 1:
            return imgs[0], imgs[0]
        return imgs[0], imgs[1]
    
    init_q = np.array([
        0.0/180.0*3.14,
        -90.0/180.0*3.14,
        90.0/180.0*3.14,
        -90.0/180.0*3.14,
        -90.0/180.0*3.14,
        0.0/180.0*3.14,
    ], dtype=np.float64)

    def go_home():
        try:
            print("[INFO] Returning to initial pose...")
            # ros2_bridge.switch_controller(start=['scaled_joint_trajectory_controller'],
            #                                 stop=['forward_position_controller'])
            ros2_bridge.move_to_joint(init_q, duration=3.0)
            # ros2_bridge.switch_controller(start=['forward_position_controller'],
            #                                 stop=['scaled_joint_trajectory_controller'])
            # print("[INFO] Switching controller: start=['scaled_joint_trajectory_controller'] stop=['forward_position_controller']")
            # ros2_bridge.switch_controller(start=['scaled_joint_trajectory_controller'],
            #                       stop=['forward_position_controller'])
        except Exception as e:
            print(f"[WARN] Home/switch error: {e}")

    # def convert_depth_channel(depth_frame, depth_min=0.01, depth_max=10.0):
    #     # depth_min = depth_frame.min()
    #     # depth_max = depth_frame.max()

    #     if depth_max > depth_min:
    #         depth_normalized = (depth_frame - depth_min) / (depth_max - depth_min) * 255.0
    #     else:
    #         depth_normalized = np.full_like(depth_frame, 127.5)

    #     depth_uint8 = np.clip(depth_normalized, 0, 255).astype(np.uint8)
    #     depth_3ch = np.repeat(depth_uint8[..., np.newaxis], 3, axis=-1)

    #     return depth_3ch
    
    def convert_depth_channel(
        depth_frame: np.ndarray,
        depth_scale: float = 0.001,
        depth_min_m: float = 0.01,
        depth_max_m: float = 5.0,
    ) -> np.ndarray:
        """
        depth_frame: uint16 raw depth from RealSense
        depth_scale: sensor.get_depth_scale() (meters per unit)
        returns: 3-channel uint8 grayscale for visualization
        """
        # Convert to meters
        depth_m = depth_frame.astype(np.float32) * depth_scale

        # Invalid depths (0) -> set to max so they appear far / dark (or handle separately)
        invalid_mask = depth_frame == 0
        depth_m[invalid_mask] = depth_max_m

        # Clip to range of interest
        depth_m = np.clip(depth_m, depth_min_m, depth_max_m)

        # Normalize to 0–255
        depth_normalized = (depth_m - depth_min_m) / (depth_max_m - depth_min_m)
        depth_normalized = (depth_normalized * 255.0).astype(np.uint8)

        # If you need 3-channel for video writer (BGR)
        depth_3ch = np.repeat(depth_normalized[..., np.newaxis], 3, axis=-1)

        return depth_3ch

    episode_index = 0

    print("\n=== Interactive Collection (LeRobot) ===")
    print("ENTER: start recording one episode")
    print("During recording: ENTER=SAVE to LeRobot, 'q'+ENTER=DISCARD (no files)")
    print("Robot returns to initial pose & switches controllers after each episode.\n")

    try:
        while rclpy.ok():
            user_in = input("[READY] Press ENTER to start (or 'q'+ENTER to quit): ").strip().lower()
            if user_in == 'q':
                break

            # ---- Record until ENTER (save) or 'q' (discard) ----
            print(f"[INFO] Recording episode {episode_index}. ENTER=SAVE, 'q'+ENTER=DISCARD.")
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

            dt = 1.0 / float(args.fps)
            t0 = time.time()
            step_idx = 0

            # Buffer frames as "frame_data" dicts compatible with LeRobotDataset
            episode_buffer: List[dict] = []

            while not stop_event.is_set() and not quit_event.is_set():
                t_now = time.time()

                # --- Observations ---
                arm_current_state = ros2_bridge.get_arm_state().astype(np.float64)  # (6,)
                gripper_current_state = ros2_bridge.get_gripper_state().astype(np.float64)  # likely (6,) duplicated
                # gpos = float(np.mean(grip_state_full)) if grip_state_full.size > 0 else 0.0

                # 12-dim state/action (UR5e 6 + 1 gripper + padding)
                obs_state = np.zeros(args.joints_num, dtype=np.float64)
                obs_state[:6] = arm_current_state
                obs_state[6:] = gripper_current_state

                arm_cmd, gripper_cmd = cmd_tap.get_last_cmd()
                act = np.zeros(args.joints_num, dtype=np.float64)
                act[:6] = arm_cmd
                act[6:] = gripper_cmd

                wrist_img, ext_img = get_images_pair()
                # rgb_frames = cams.get_rgb_frames_list()
                # wrist_img, ext_img = rgb_frames[0], rgb_frames[1]

                # Fake/blank seg images (optional to change if you have segmentation)
                blank_seg = np.zeros_like(wrist_img, dtype=np.uint8)

                # Build LeRobot frame record
                if args.enable_depth:
                    # depth_frames = cams.get_depth_frames_list()
                    wrist_depth_img, ext_depth_img = get_depth_pair()
                    # wrist_depth, ext_depth = depth_frames[0], depth_frames[1]
                    wrist_depth, ext_depth = convert_depth_channel(wrist_depth_img), convert_depth_channel(ext_depth_img)
                    frame_data = {
                        "observation.images.exterior_image": ext_img[:, :, ::-1],   # convert BGR->RGB if needed by your pipeline
                        "observation.images.wrist_image": wrist_img[:, :, ::-1],
                        # "observation.images.exterior_segmentation_image": blank_seg,
                        # "observation.images.wrist_segmentation_image": blank_seg,
                        "observation.images.exterior_depth_image": ext_depth,
                        "observation.images.wrist_depth_image": wrist_depth,
                        "observation.state": obs_state,
                        "action": act,
                        "annotation.human.action.task_description": np.array([0], dtype=np.int64),  # use an index; or map text->index
                        "next.reward": np.array([0.0], dtype=np.float64),
                        "next.done": np.array([False], dtype=np.bool_),
                        # "task": args.task_description,
                    }
                else:
                    frame_data = {
                        "observation.images.exterior_image": ext_img[:, :, ::-1],   # convert BGR->RGB if needed by your pipeline
                        "observation.images.wrist_image": wrist_img[:, :, ::-1],
                        # "observation.images.exterior_segmentation_image": blank_seg,
                        # "observation.images.wrist_segmentation_image": blank_seg,
                        "observation.state": obs_state,
                        "action": act,
                        "annotation.human.action.task_description": np.array([0], dtype=np.int64),  # use an index; or map text->index
                        "next.reward": np.array([0.0], dtype=np.float64),
                        "next.done": np.array([False], dtype=np.bool_),
                        # "task": args.task_description,
                    }                    
                episode_buffer.append(frame_data)

                # pacing
                t_next = t_now + dt
                while time.time() < t_next and not stop_event.is_set() and not quit_event.is_set():
                    time.sleep(0.001)
                step_idx += 1

            # Decide save vs discard
            if quit_event.is_set():
                print("[INFO] Episode discarded (no data written).")
            else:
                # mark last as terminal
                if len(episode_buffer) > 0:
                    episode_buffer[-1]["next.done"] = np.array([True], dtype=np.bool_)
                    episode_buffer[-1]["next.reward"] = np.array([1.0], dtype=np.float64)
                print(f"[INFO] Saving episode {episode_index} with {len(episode_buffer)} frames to LeRobot...")
                for fd in episode_buffer:
                    dataset.add_frame(fd, task=args.task_description)
                    # dataset.add_frame(fd)
                dataset.save_episode()
                # write_episode(episode_index, wrist_img,c ext_img)
                # wrist_images_dir = os.path.join(output_dir, "images",
                #                         "observation.images.wrist_image",
                #                         f"episode_{episode_index:06d}")
                # wrist_video_path = os.path.join(output_dir, "videos", "chunk-000",
                #                         "observation.images.wrist_image",
                #                         f"episode_{episode_index:06d}.mp4")
                # ext_images_dir = os.path.join(output_dir, "images",
                #                         "observation.images.exterior_image",
                #                         f"episode_{episode_index:06d}")
                # ext_video_path = os.path.join(output_dir, "videos", "chunk-000",
                #                         "observation.images.exterior_image",
                #                         f"episode_{episode_index:06d}.mp4")
                # images_to_h264_video(wrist_images_dir, wrist_video_path)
                # images_to_h264_video(ext_images_dir, ext_video_path)

                episode_index += 1
                # go_home()


            # Cleanup after each episode
            try: t_key.join(timeout=0.1)
            except Exception: pass


    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt")
    finally:
        create_modality(args)
        try:
            cams.stop()
        except Exception:
            pass
        try:
            executor.shutdown()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass
        print("[DONE] Collector finished.")


if __name__ == "__main__":
    main()