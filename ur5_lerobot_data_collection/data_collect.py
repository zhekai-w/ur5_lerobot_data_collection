import time
from pathlib import Path
import cv2
import numpy as np
from pynput import keyboard
import threading
# ROS2 Library
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Joy

# LeRobot Library
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Azure Kinect
import pyk4a
from pyk4a import Config, PyK4A

# Realsense
import pyrealsense2 as rs

# Dualsense
from dualsense_controller import DualSenseController

width_glob = 640
height_glob = 360
fps_glob = 30

def convert_depth_channel(
    depth_frame: np.ndarray,
    depth_scale: float = 0.001,
    depth_min_m: float = 0.01,
    depth_max_m: float = 5.0,
) -> np.ndarray:
    """Convert uint16 raw depth to 3-channel uint8 for LeRobot storage."""
    depth_m = depth_frame.astype(np.float32) * depth_scale
    depth_m[depth_frame == 0] = depth_max_m
    depth_m = np.clip(depth_m, depth_min_m, depth_max_m)
    depth_normalized = (depth_m - depth_min_m) / (depth_max_m - depth_min_m)
    depth_uint8 = (depth_normalized * 255.0).astype(np.uint8)
    return np.repeat(depth_uint8[..., np.newaxis], 3, axis=-1)


class DataCollector(Node):
    def __init__(self, dataset):
        super().__init__('joint_state_subscriber')

        self.declare_parameter('task', 'task description')

        self.dataset = dataset
        self.is_recording = False
        self.frame_count = 0
        self.previous_position = None
        self.task = self.get_parameter('task').value
        self.lock = threading.Lock()
        self.dataset_lock = threading.Lock()

        self.should_quit = False

        # Recording fps
        self.target_fps = fps_glob

        # Buffer for latest joint state
        self.latest_joint_position = None
        self.joint_lock = threading.Lock()

        # Subscribe to joint states (just for buffering)
        self.subscription = self.create_subscription(
            JointState,
            "/joint_states",
            self.jointstate_callback,
            10)
        
        # Buffer for latest gripper state
        self.latest_gripper_position = 0.0
        self.gripper_lock = threading.Lock()

        # Subscriber to gripper joint states
        self.gripper_subscription = self.create_subscription(
            JointState, 
            "/gripper/joint_states",
            self.gripper_callback,
            10)

        # Timer for recording at exactly 20fps
        timer_period = 1.0 / self.target_fps
        self.recording_timer = self.create_timer(timer_period, self.recording_callback)

        # Dualsense controller
        try:
            device_infos = DualSenseController.enumerate_devices()
            if len(device_infos) > 0:
                self.controller = DualSenseController(device_index_or_device_info=device_infos[0])
                self.controller.btn_create.on_down(self.on_btn_start_record)
                self.controller.btn_cross.on_down(self.on_btn_save_episode)
                self.controller.btn_circle.on_down(self.on_btn_discard)
                self.controller.btn_triangle.on_down(self.on_btn_quit)
                self.controller.activate()
            else:
                self.controller = None
                print("\033[93mNo DualSense controller found, using keyboard only.\033[0m")
        except Exception as e:
            self.controller = None
            print(f"\033[93mFailed to initialize DualSense controller: {e}\033[0m")
            print("Fall back to keyboard.")

        # Initialize Azure Kinect
        config = Config()
        config.color_resolution = pyk4a.ColorResolution.RES_1080P
        config.depth_mode = pyk4a.DepthMode.NFOV_UNBINNED
        config.camera_fps = pyk4a.FPS.FPS_30
        config.synchronized_images_only = True

        self.k4a = PyK4A(config)
        self.k4a.start()

        # Initialize Realsense
        if fps_glob > 30:
            realsense_fps = 60
        else:
            realsense_fps = 30
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, width_glob, height_glob, rs.format.rgb8, realsense_fps)
        config.enable_stream(rs.stream.depth, width_glob, height_glob, rs.format.z16, realsense_fps)
        self.pipeline.start(config)


        # Async camera capture variables
        # Azure color + depth
        self.latest_k4a_image = None
        self.latest_k4a_depth = None
        # Save previous frame if not frame
        self.last_k4a_image = None
        self.last_k4a_depth = None
        self.k4a_lock = threading.Lock()   # protects both latest_k4a_image + latest_k4a_depth
        # Realsense color + depth
        self.latest_rs_image = None
        self.latest_rs_depth = None
        self.rs_lock = threading.Lock()    # protects both latest_rs_image + latest_rs_depth

        self.camera_running = True

        self.rs_thread = threading.Thread(target=self._rs_capture_loop, daemon=True)
        self.k4a_thread = threading.Thread(target=self._k4a_capture_loop, daemon=True)
        self.rs_thread.start()
        self.k4a_thread.start()

        # Start keyboard listener
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()
        print("\033[36m Remember to change task description before recording.\033[0m")
        print(f"Collecting data at {self.target_fps}hz, realsense running at {realsense_fps}")
        print("Press 's' to start recording, 'e' to end episode, " \
                "'q' to quit, 'e' to delete episode")
        print("Press 'create' to start recording, 'cross' to end episode, " \
                "'triangle' to quit, 'circle' to delete episode")

    def _k4a_capture_loop(self):
        """Continuously capture images from Azure Kinect in a separate thread"""
        while self.camera_running:
            try:
                capture = self.k4a.get_capture()
                if capture is None:
                    continue
                if capture.color is None or capture.transformed_depth is None:
                    continue

                # Process outside the lock
                rgb = capture.color[:, :, 2::-1]
                rgb = cv2.resize(rgb, (width_glob, height_glob), interpolation=cv2.INTER_AREA)
                color_image = np.array(rgb, dtype=np.uint8)

                depth = capture.transformed_depth
                depth = cv2.resize(depth, (width_glob, height_glob), interpolation=cv2.INTER_NEAREST)

                # Only write under the lock
                with self.k4a_lock:
                    self.latest_k4a_image = color_image
                    self.latest_k4a_depth = depth

            except Exception as e:
                self.get_logger().error(f"Azure Kinect capture error: {e}")
                time.sleep(0.01)

    def _rs_capture_loop(self):
        """Continuously capture images from RealSense in a separate thread"""
        while self.camera_running:
            try:
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue

                # Process outside the lock
                color_image = np.asanyarray(color_frame.get_data(), dtype=np.uint8)
                depth_image = np.asanyarray(depth_frame.get_data(), dtype=np.uint16)

                # Only write under the lock
                with self.rs_lock:
                    self.latest_rs_image = color_image
                    self.latest_rs_depth = depth_image
            except RuntimeError as e:
                self.get_logger().error(f"Realsense capture error: {e}")
                time.sleep(0.01)

    def get_latest_k4a(self):
        """Get the most recent synchronized k4a color + depth (same capture)"""
        with self.k4a_lock:
            if self.latest_k4a_image is not None and self.latest_k4a_depth is not None:
                return self.latest_k4a_image.copy(), self.latest_k4a_depth.copy()
        return None, None

    def get_latest_rs(self):
        """Get the most recent synchronized rs color + depth (same capture)"""
        with self.rs_lock:
            if self.latest_rs_image is not None and self.latest_rs_depth is not None:
                return self.latest_rs_image.copy(), self.latest_rs_depth.copy()
        return None, None

    def get_latest_joint_position(self):
        """Get the most recent joint position"""
        with self.joint_lock:
            if self.latest_joint_position is not None:
                return self.latest_joint_position.copy()
        return None

    def get_latest_gripper_position(self):
        with self.gripper_lock:
            return self.latest_gripper_position

    def on_btn_start_record(self):
        with self.lock:
            self.is_recording = True
            self.frame_count = 0
            self.previous_position = None
            print("\nStarted recording")

    def on_btn_save_episode(self):
        should_save = False
        frame_count_to_report = 0
        with self.lock:
            if self.is_recording and self.frame_count > 0:
                self.is_recording = False
                should_save = True
                frame_count_to_report = self.frame_count
                self.previous_position = None
                self.frame_count = 0
            else:
                print("\nNo frames recorded yet")
        if should_save:
            try:
                print("\nSaving episode...")
                with self.dataset_lock:
                    self.dataset.save_episode()
                print(f"Episode saved ({frame_count_to_report} frames)")
                print(f"Total episodes: {self.dataset.num_episodes}")
                print(f"Task description: {self.task}")
                print("Press 'create' to start recording, 'cross' to end episode, " \
                    "'triangle' to quit, 'circle' to delete episode")
            except Exception as e:
                print(f"\nError saving episode: {e}")

    def on_btn_quit(self):
        print("\nQuitting...")
        # raise SystemExit()
        self.should_quit = True

    def on_btn_discard(self):
        with self.lock:
            self.is_recording = False
            self.frame_count = 0
            self.previous_position = None
        time.sleep(0.1) # let camera writes finish
        with self.dataset_lock:
            self.dataset.clear_episode_buffer()
        print("\nEpisode discarded")
        print("Press 'create' to start recording, 'cross' to end episode, " \
            "'triangle' to quit, 'circle' to delete episode")

    def on_press(self, key):
        try:
            if key.char == 's':
                with self.lock:
                    self.is_recording = True
                    self.frame_count = 0
                    self.previous_position = None
                    print("\nStarted recording")
            elif key.char == 'e':
                # Check and update state inside lock, but save outside to avoid blocking
                should_save = False
                frame_count_to_report = 0
                with self.lock:
                    if self.is_recording and self.frame_count > 0:
                        self.is_recording = False
                        should_save = True
                        frame_count_to_report = self.frame_count
                        self.previous_position = None
                        self.frame_count = 0
                    else:
                        print("\nNo frames recorded yet")

                # Save episode outside the lock to avoid blocking callbacks
                if should_save:
                    try:
                        print("\nSaving episode...")
                        with self.dataset_lock:
                            self.dataset.save_episode()
                        print(f"Episode saved ({frame_count_to_report} frames)")
                        print(f"Total episodes: {self.dataset.num_episodes}")
                        print("Press 's' to start recording, 'e' to end episode, " \
                                "'q' to quit, 'e' to delete episode")

                    except Exception as e:
                        print(f"\nError saving episode: {e}")
            elif key.char == 'd':
                with self.lock:
                    self.is_recording = False
                    self.frame_count = 0
                    self.previous_position = None
                with self.dataset_lock:
                    self.dataset.clear_episode_buffer()
                print("\nEpisode not saved")
                print("Press 's' to start recording, 'e' to end episode, " \
                        "'q' to quit, 'e' to delete episode")
                
            elif key.char == 'q':
                print("\nQuitting...")
                # raise SystemExit()
                self.should_quit = True
        except AttributeError:
            pass

    def jointstate_callback(self, msg):
        """Just buffer the latest joint state - no recording logic here"""
        # Get current position with gripper (0 for closed)
        current_position = np.array(list(msg.position), dtype=np.float32)
        with self.joint_lock:
            self.latest_joint_position = current_position
    
    def gripper_callback(self, msg):
        with self.gripper_lock:
            self.latest_gripper_position = msg.position[0]
            # print(f"GRIPPER POSITION: {self.latest_gripper_position}")

    def recording_callback(self):
        """Timer callback at exactly 20fps for recording frames"""
        with self.lock:
            if not self.is_recording:
                return

            # Get latest joint position
            arm_position = self.get_latest_joint_position()
            gripper_position = self.get_latest_gripper_position()
            current_position = np.append(arm_position, gripper_position).astype(np.float32)
            if current_position is None:
                self.get_logger().warn("No joint state available yet, skipping frame")
                return

            # Only record if we have a previous position (for action)
            if self.previous_position is not None:
                # Get the latest cached image from async capture thread
                k4a_image, k4a_depth = self.get_latest_k4a()
                rs_image, rs_depth = self.get_latest_rs()

                if k4a_image is None or k4a_depth is None:
                    print("No Azure Kinect frame available yet, skipping frame")
                    # self.get_logger().warn("No Azure Kinect frame available yet, skipping frame")
                    return
                if rs_image is None or rs_depth is None:
                    self.get_logger().warn("No RealSense frame available, skipping frame")
                    return
                
                frame = {
                    "observation.state": self.previous_position,
                    "observation.images.cam1": k4a_image,
                    "observation.images.cam2": rs_image,
                    "observation.images.cam1_depth": convert_depth_channel(k4a_depth),
                    "observation.images.cam2_depth": convert_depth_channel(rs_depth),
                    "action": current_position,
                }

                with self.dataset_lock:
                    self.dataset.add_frame(frame, task=self.task)
                self.frame_count += 1

            # Update previous position for next frame
            self.previous_position = current_position

    def stop_camera(self):
        """Stop the camera capture thread and Azure Kinect"""
        self.camera_running = False
        for thread in (self.k4a_thread, self.rs_thread):
            if thread.is_alive():
                thread.join(timeout=1.0)
        try:
            self.k4a.stop()
            self.get_logger().info("Azure Kinect stopped")
        except Exception as e:
            print(f"Error stopping Azure Kinect: {e}")
        try:
            self.pipeline.stop()
            self.get_logger().info("Realsense pipeline stopped")
        except Exception as e:
            print(f"Error stopping Realsense: {e}")
        try:
            if self.controller is not None:
                self.controller.deactivate()
                self.get_logger().info("Dualsense controller deactivated")
        except Exception as e:
            print(f"Error deactivating Dualsense controller: {e}")


def main():
    # Robot configuration
    joints_name = ["shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint",
        "wrist_3_joint", "shoulder_pan_joint", "gripper_joint"]
    # joints_name = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"]
    n_joints = len(joints_name)

    # Camera configuration (Azure Kinect 1080p)
    width = width_glob
    height = height_glob
    rgb_channel = 3
    # 3channel depth because lerobot does not support native uint16 1 channel depth
    depth_channel = 3
    root_dir = './All_Datasets/30hz/just_medium'
    # TODO: Change this to actual task disscription
    use_videos = False
    if use_videos is True:
        cam_dtype = "video"
    else:        
        cam_dtype = "image"

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (n_joints,),
            "names": list(joints_name),
        },
        "observation.images.cam1": {
            "dtype": cam_dtype,
            "shape": (height, width, rgb_channel),
            "names": ["height", "width", "channel"],
        },
        "observation.images.cam2": {
            "dtype": cam_dtype,
            "shape": (height, width, rgb_channel),
            "names": ["height", "width", "channel"],
        },
        "observation.images.cam1_depth": {
            "dtype": cam_dtype,
            "shape": (height, width, depth_channel),
            "names": ["height", "width", "channel"],
        },
        "observation.images.cam2_depth": {
            "dtype": cam_dtype,
            "shape": (height, width, depth_channel),
            "names": ["height", "width", "channel"],
        },
        "action": {
            "dtype": "float32",
            "shape": (n_joints,),
            "names": list(joints_name),
        },
    }

    repo_id = "zhekai-w/ur5_lerobot_dataset"
    root_path = Path(root_dir)

    if root_path.exists() and (root_path / "meta" / "info.json").exists():
        print(f"[INFO] Found existing dataset at {root_dir}, resuming collection...")
        dataset = LeRobotDataset(
            repo_id=repo_id,
            root=root_dir,
            video_backend="torchcodec",
            batch_encoding_size=64,
        )
        dataset.start_image_writer(num_processes=20, num_threads=20)
        dataset.episode_buffer = dataset.create_episode_buffer()
        print(f"[INFO] Existing dataset has {dataset.num_episodes} episodes, {dataset.meta.total_frames} frames")
    else:
        print(f"[INFO] Creating new dataset at {root_dir}...")
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps_glob,
            features=features,
            root=root_dir,
            robot_type="ur5",
            use_videos=use_videos,
            video_backend="torchcodec",
            image_writer_processes=20,
            image_writer_threads=20,
            batch_encoding_size=1,
        )

    rclpy.init()
    data_collector = DataCollector(dataset)

    try:
        # rclpy.spin(data_collector)
        while rclpy.ok() and not data_collector.should_quit:
            rclpy.spin_once(data_collector)
            # print("AAAAAAAAAAAAa")
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        data_collector.stop_camera()
        # Stop the async image writer before shutting down
        dataset.stop_image_writer()

        # Encode any remaining episodes that haven't been encoded yet
        if dataset.episodes_since_last_encoding > 0:
            print(f"Encoding {dataset.episodes_since_last_encoding} remaining episode(s) to video...")
            start_ep = dataset.num_episodes - dataset.episodes_since_last_encoding
            dataset.batch_encode_videos(start_ep, dataset.num_episodes)
            print("Video encoding complete.")

        data_collector.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
