#!/usr/bin/env python3

import sys
import time
import threading

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import JointState

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Azure Kinect
import pyk4a
from pyk4a import Config, PyK4A

# Realsense
import pyrealsense2 as rs


class JointStateTap(Node):
    """Minimal ROS2 node that buffers the latest joint state."""
    def __init__(self):
        super().__init__('joint_state_tap')
        self._lock = threading.Lock()
        self._latest = None
        self.create_subscription(JointState, "/joint_states", self._cb, 1)
        self.get_logger().info("[JointStateTap] Listening on /joint_states")

    def _cb(self, msg):
        pos = np.array(list(msg.position) + [0], dtype=np.float32)
        with self._lock:
            self._latest = pos

    def get_latest(self):
        with self._lock:
            return self._latest.copy() if self._latest is not None else None


def main():
    # --- Configuration ---
    joints_name = [
        "shoulder_lift_joint", "elbow_joint", "wrist_1_joint",
        "wrist_2_joint", "wrist_3_joint", "shoulder_pan_joint", "gripper_joint",
    ]
    n_joints = len(joints_name)
    # width, height = 1280, 720
    width, height = 640, 360
    fps = 30
    root_dir = "./dataset"
    task_description = "reach target"
    use_videos = True


    # --- Dataset ---
    if use_videos is True:
        features = {
            "observation.state": {
                "dtype": "float32", "shape": (n_joints,), "names": list(joints_name),
            },
            "observation.images.cam1": {
                "dtype": "video", "shape": (height, width, 3), "names": ["height", "width", "channel"],
            },
            "observation.images.cam2": {
                "dtype": "video", "shape": (height, width, 3), "names": ["height", "width", "channel"],
            },
            "action": {
                "dtype": "float32", "shape": (n_joints,), "names": list(joints_name),
            },
        }
    else:
        features = {
            "observation.state": {
                "dtype": "float32", "shape": (n_joints,), "names": list(joints_name),
            },
            "observation.images.cam1": {
                "dtype": "image", "shape": (height, width, 3), "names": ["height", "width", "channel"],
            },
            "observation.images.cam2": {
                "dtype": "image", "shape": (height, width, 3), "names": ["height", "width", "channel"],
            },
            "action": {
                "dtype": "float32", "shape": (n_joints,), "names": list(joints_name),
            },
        }

    dataset = LeRobotDataset.create(
        repo_id="zhekai-w/ur5_lerobot_dataset",
        fps=fps,
        features=features,
        root=root_dir,
        robot_type="ur5",
        use_videos=use_videos,
        video_backend="torchcodec",
        image_writer_processes=10,
        image_writer_threads=10,
    )

    # --- ROS2 ---
    rclpy.init()
    joint_tap = JointStateTap()
    executor = MultiThreadedExecutor()
    executor.add_node(joint_tap)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    # --- Cameras ---
    # Azure Kinect
    k4a_config = Config()
    k4a_config.color_resolution = pyk4a.ColorResolution.RES_720P
    k4a_config.depth_mode = pyk4a.DepthMode.NFOV_UNBINNED
    k4a_config.camera_fps = pyk4a.FPS.FPS_30
    k4a_config.synchronized_images_only = True
    k4a = PyK4A(k4a_config)
    k4a.start()

    # Realsense
    rs_pipeline = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, 30)
    rs_pipeline.start(rs_config)

    # Async camera capture threads
    latest_k4a = {"img": None, "lock": threading.Lock()}
    latest_rs = {"img": None, "lock": threading.Lock()}
    camera_running = threading.Event()
    camera_running.set()

    def k4a_capture_loop():
        while camera_running.is_set():
            try:
                capture = k4a.get_capture()
                if capture.color is not None:
                    rgb = capture.color[:, :, 2::-1]
                    resized = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
                    with latest_k4a["lock"]:
                        latest_k4a["img"] = resized
            except Exception:
                time.sleep(0.01)

    def rs_capture_loop():
        while camera_running.is_set():
            try:
                frames = rs_pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if color_frame:
                    with latest_rs["lock"]:
                        latest_rs["img"] = np.asanyarray(color_frame.get_data(), dtype=np.uint8)
            except Exception:
                time.sleep(0.01)

    threading.Thread(target=k4a_capture_loop, daemon=True).start()
    threading.Thread(target=rs_capture_loop, daemon=True).start()
    time.sleep(1.0)

    def get_k4a_image():
        with latest_k4a["lock"]:
            return latest_k4a["img"].copy() if latest_k4a["img"] is not None else None

    def get_rs_image():
        with latest_rs["lock"]:
            return latest_rs["img"].copy() if latest_rs["img"] is not None else None

    # --- Interactive collection loop ---
    episode_index = 0
    dt = 1.0 / fps

    print("\n=== Interactive Collection (LeRobot) ===")
    print("ENTER: start recording one episode")
    print("During recording: ENTER=SAVE, 'q'+ENTER=DISCARD\n")

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

            frame_count = 0
            previous_position = None

            while not stop_event.is_set() and not quit_event.is_set():
                t_now = time.time()

                current_position = joint_tap.get_latest()
                if current_position is None:
                    time.sleep(0.001)
                    continue

                if previous_position is not None:
                    k4a_img = get_k4a_image()
                    rs_img = get_rs_image()
                    if k4a_img is None or rs_img is None:
                        time.sleep(0.001)
                        continue

                    frame_data = {
                        "observation.state": previous_position,
                        "observation.images.cam1": k4a_img,
                        "observation.images.cam2": rs_img,
                        "action": current_position,
                    }
                    dataset.add_frame(frame_data, task=task_description)
                    frame_count += 1

                previous_position = current_position

                # Pacing to target FPS
                t_next = t_now + dt
                while time.time() < t_next and not stop_event.is_set() and not quit_event.is_set():
                    time.sleep(0.001)

            # Decide save vs discard
            if quit_event.is_set():
                dataset.clear_episode_buffer()
                print("[INFO] Episode discarded (no data written).")
            else:
                if frame_count > 0:
                    print(f"[INFO] Saving episode {episode_index} with {frame_count} frames...")
                    dataset.save_episode()
                    episode_index += 1
                    print(f"[INFO] Episode saved. Total episodes: {dataset.num_episodes}")
                else:
                    print("[INFO] No frames recorded.")

            try:
                t_key.join(timeout=0.1)
            except Exception:
                pass

    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt")
    finally:
        camera_running.clear()
        try:
            k4a.stop()
        except Exception:
            pass
        try:
            rs_pipeline.stop()
        except Exception:
            pass

        dataset.stop_image_writer()
        if dataset.episodes_since_last_encoding > 0:
            print(f"Encoding {dataset.episodes_since_last_encoding} remaining episode(s) to video...")
            start_ep = dataset.num_episodes - dataset.episodes_since_last_encoding
            dataset.batch_encode_videos(start_ep, dataset.num_episodes)
            print("Video encoding complete.")

        try:
            executor.shutdown()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass
        print("[DONE] Collector finished.")


if __name__ == '__main__':
    main()
