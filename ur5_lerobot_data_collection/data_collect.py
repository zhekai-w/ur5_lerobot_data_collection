import time
from pathlib import Path
import cv2
import numpy as np
from pynput import keyboard
import threading
# ROS2 Library
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32, Float64MultiArray

# LeRobot Library
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Azure Kinect
import pyk4a
from pyk4a import Config, PyK4A

# Dualsense
from dualsense_controller import DualSenseController

width_glob = 640
height_glob = 360
fps_glob = 30

WFOV_DEVICE = 0


def convert_depth_channel(
    depth_frame: np.ndarray,
    depth_scale: float = 0.001,
    depth_min_m: float = 0.01,
    depth_max_m: float = 5.0,
) -> np.ndarray:
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
        self.task = self.get_parameter('task').value
        self.lock = threading.Lock()
        self.dataset_lock = threading.Lock()

        self.should_quit = False

        self.target_fps = fps_glob

        # Joint state feedback (observation.state)
        self.latest_joint_position = None
        self.joint_lock = threading.Lock()

        self.subscription = self.create_subscription(
            JointState,
            "/joint_states",
            self.jointstate_callback,
            10)

        self.latest_gripper_position = 0.0
        self.gripper_lock = threading.Lock()

        self.gripper_subscription = self.create_subscription(
            JointState,
            "/gripper/joint_states",
            self.gripper_callback,
            10)

        self.latest_gripper_cmd = 0.0
        self.gripper_cmd_lock = threading.Lock()

        self.gripper_cmd_subscription = self.create_subscription(
            Float32,
            "/gripper/commands",
            self.gripper_cmd_callback,
            10)

        # Arm command tap (action) — actual commands sent to controller
        self.latest_arm_cmd = None
        self.arm_cmd_lock = threading.Lock()

        self.arm_cmd_subscription = self.create_subscription(
            Float64MultiArray,
            "/forward_position_controller/commands",
            self.arm_cmd_callback,
            10)

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

        # Azure Kinect
        config = Config()
        config.color_resolution = pyk4a.ColorResolution.RES_1080P
        config.depth_mode = pyk4a.DepthMode.OFF
        config.camera_fps = pyk4a.FPS.FPS_30
        config.synchronized_images_only = False
        self.k4a = PyK4A(config)
        self.k4a.start()

        # WFOV USB camera
        self.wfov_cap = cv2.VideoCapture(WFOV_DEVICE)
        self.wfov_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.wfov_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 9999)
        self.wfov_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 9999)
        if not self.wfov_cap.isOpened():
            raise RuntimeError(f"Failed to open WFOV camera at /dev/video{WFOV_DEVICE}")

        # Camera buffers — depth already converted to 8-bit RGB in capture threads
        self.latest_k4a_image = None
        self.k4a_lock = threading.Lock()

        self.latest_wfov_image = None
        self.wfov_lock = threading.Lock()

        self.camera_running = True

        self.k4a_thread = threading.Thread(target=self._k4a_capture_loop, daemon=True)
        self.wfov_thread = threading.Thread(target=self._wfov_capture_loop, daemon=True)
        self.k4a_thread.start()
        self.wfov_thread.start()

        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()
        print("\033[36m Remember to change task description before recording.\033[0m")
        print(f"Collecting data at {self.target_fps}hz")
        print("Press 's' to start recording, 'e' to end episode, 'q' to quit, 'd' to discard")
        print("Press 'create' to start recording, 'cross' to end episode, "
              "'triangle' to quit, 'circle' to discard episode")

    # ------------------------------------------------------------------
    # Capture loops — depth converted here, off the 30Hz timer hot path
    # ------------------------------------------------------------------

    def _k4a_capture_loop(self):
        while self.camera_running:
            try:
                capture = self.k4a.get_capture()
                if capture is None:
                    continue
                if capture.color is None:
                    continue

                rgb = capture.color[:, :, 2::-1]
                rgb = cv2.resize(rgb, (width_glob, height_glob), interpolation=cv2.INTER_AREA)
                color_image = np.array(rgb, dtype=np.uint8)

                with self.k4a_lock:
                    self.latest_k4a_image = color_image

            except Exception as e:
                self.get_logger().error(f"Azure Kinect capture error: {e}")
                time.sleep(0.01)

    def _wfov_capture_loop(self):
        while self.camera_running:
            try:
                ret, frame = self.wfov_cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue

                h, w = frame.shape[:2]
                target_ratio = width_glob / height_glob
                if w / h > target_ratio:
                    crop_w = int(h * target_ratio)
                    x = (w - crop_w) // 2
                    frame = frame[:, x:x + crop_w]
                else:
                    crop_h = int(w / target_ratio)
                    y = (h - crop_h) // 2
                    frame = frame[y:y + crop_h, :]
                frame = cv2.resize(frame, (width_glob, height_glob), interpolation=cv2.INTER_AREA)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                with self.wfov_lock:
                    self.latest_wfov_image = frame

            except Exception as e:
                self.get_logger().error(f"WFOV capture error: {e}")
                time.sleep(0.01)

    # ------------------------------------------------------------------
    # Buffer getters
    # ------------------------------------------------------------------

    def get_latest_k4a(self):
        with self.k4a_lock:
            if self.latest_k4a_image is not None:
                return self.latest_k4a_image.copy()
        return None

    def get_latest_wfov(self):
        with self.wfov_lock:
            if self.latest_wfov_image is not None:
                return self.latest_wfov_image.copy()
        return None

    def get_latest_joint_position(self):
        with self.joint_lock:
            if self.latest_joint_position is not None:
                return self.latest_joint_position.copy()
        return None

    def get_latest_gripper_position(self):
        with self.gripper_lock:
            return self.latest_gripper_position

    # ------------------------------------------------------------------
    # Button / keyboard handlers
    # ------------------------------------------------------------------

    def on_btn_start_record(self):
        with self.lock:
            self.is_recording = True
            self.frame_count = 0
            print("\nStarted recording")

    def on_btn_save_episode(self):
        should_save = False
        frame_count_to_report = 0
        with self.lock:
            if self.is_recording and self.frame_count > 0:
                self.is_recording = False
                should_save = True
                frame_count_to_report = self.frame_count
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
                print("Press 'create' to start recording, 'cross' to end episode, "
                      "'triangle' to quit, 'circle' to discard episode")
            except Exception as e:
                print(f"\nError saving episode: {e}")

    def on_btn_quit(self):
        print("\nQuitting...")
        self.should_quit = True

    def on_btn_discard(self):
        with self.lock:
            self.is_recording = False
            self.frame_count = 0
        time.sleep(0.1)
        with self.dataset_lock:
            self.dataset.clear_episode_buffer()
        print("\nEpisode discarded")
        print("Press 'create' to start recording, 'cross' to end episode, "
              "'triangle' to quit, 'circle' to discard episode")

    def on_press(self, key):
        try:
            if key.char == 's':
                with self.lock:
                    self.is_recording = True
                    self.frame_count = 0
                    print("\nStarted recording")
            elif key.char == 'e':
                should_save = False
                frame_count_to_report = 0
                with self.lock:
                    if self.is_recording and self.frame_count > 0:
                        self.is_recording = False
                        should_save = True
                        frame_count_to_report = self.frame_count
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
                        print("Press 's' to start recording, 'e' to end episode, "
                              "'q' to quit, 'd' to discard")
                    except Exception as e:
                        print(f"\nError saving episode: {e}")
            elif key.char == 'd':
                with self.lock:
                    self.is_recording = False
                    self.frame_count = 0
                with self.dataset_lock:
                    self.dataset.clear_episode_buffer()
                print("\nEpisode not saved")
                print("Press 's' to start recording, 'e' to end episode, "
                      "'q' to quit, 'd' to discard")
            elif key.char == 'q':
                print("\nQuitting...")
                self.should_quit = True
        except AttributeError:
            pass

    # ------------------------------------------------------------------
    # Joint state callbacks
    # ------------------------------------------------------------------

    JOINT_ORDER = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]

    def jointstate_callback(self, msg):
        name_to_pos = dict(zip(msg.name, msg.position))
        current_position = np.array(
            [name_to_pos[j] for j in self.JOINT_ORDER], dtype=np.float32
        )
        with self.joint_lock:
            self.latest_joint_position = current_position

    def gripper_callback(self, msg):
        with self.gripper_lock:
            self.latest_gripper_position = msg.position[0]

    def gripper_cmd_callback(self, msg: Float32):
        with self.gripper_cmd_lock:
            self.latest_gripper_cmd = msg.data

    def arm_cmd_callback(self, msg: Float64MultiArray):
        arr = np.array(list(msg.data), dtype=np.float32)
        with self.arm_cmd_lock:
            self.latest_arm_cmd = arr

    def get_latest_arm_cmd(self):
        with self.arm_cmd_lock:
            if self.latest_arm_cmd is not None:
                return self.latest_arm_cmd.copy()
        return None

    # ------------------------------------------------------------------
    # Recording timer
    # ------------------------------------------------------------------

    def recording_callback(self):
        with self.lock:
            if not self.is_recording:
                return

            arm_position = self.get_latest_joint_position()
            gripper_position = self.get_latest_gripper_position()
            if arm_position is None:
                self.get_logger().warn("No joint state available yet, skipping frame")
                return
            obs_state = np.append(arm_position, gripper_position).astype(np.float32)

            arm_cmd = self.get_latest_arm_cmd()
            if arm_cmd is None:
                self.get_logger().warn("No arm command available yet, skipping frame")
                return
            with self.gripper_cmd_lock:
                gripper_cmd = self.latest_gripper_cmd
            action = np.append(arm_cmd[:len(arm_position)], gripper_cmd).astype(np.float32)

            k4a_image = self.get_latest_k4a()
            wfov_image = self.get_latest_wfov()

            if k4a_image is None:
                print("No Azure Kinect frame available yet, skipping frame")
                return
            if wfov_image is None:
                self.get_logger().warn("No WFOV frame available, skipping frame")
                return

            frame = {
                "observation.state":            obs_state,
                "observation.images.cam1":       k4a_image,
                "observation.images.cam2":       wfov_image,
                "action":                        action,
            }

            with self.dataset_lock:
                self.dataset.add_frame(frame, task=self.task)
            self.frame_count += 1

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def stop_camera(self):
        self.camera_running = False
        for thread in (self.k4a_thread, self.wfov_thread):
            if thread.is_alive():
                thread.join(timeout=1.0)
        try:
            self.k4a.stop()
            self.get_logger().info("Azure Kinect stopped")
        except Exception as e:
            print(f"Error stopping Azure Kinect: {e}")
        try:
            self.wfov_cap.release()
            self.get_logger().info("WFOV camera released")
        except Exception as e:
            print(f"Error releasing WFOV camera: {e}")
        try:
            if self.controller is not None:
                self.controller.deactivate()
                self.get_logger().info("Dualsense controller deactivated")
        except Exception as e:
            print(f"Error deactivating Dualsense controller: {e}")


def main():
    joints_name = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint",
                   "wrist_2_joint", "wrist_3_joint", "gripper_joint"]
    n_joints = len(joints_name)

    width = width_glob
    height = height_glob
    rgb_channel = 3
    root_dir = './all_datasets/2_std_datasets/test'
    use_videos = False
    cam_dtype = "video" if use_videos else "image"

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
        dataset.start_image_writer(num_processes=4, num_threads=4)
        dataset.episode_buffer = dataset.create_episode_buffer()
        print(f"[INFO] Existing dataset has {dataset.num_episodes} episodes, "
              f"{dataset.meta.total_frames} frames")
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
            image_writer_processes=4,
            image_writer_threads=4,
            batch_encoding_size=1,
        )

    rclpy.init()
    data_collector = DataCollector(dataset)

    try:
        while rclpy.ok() and not data_collector.should_quit:
            rclpy.spin_once(data_collector, timeout_sec=0.001)

            # k4a_img = data_collector.get_latest_k4a()
            # if k4a_img is not None:
            #     cv2.imshow("Azure Kinect", cv2.cvtColor(k4a_img, cv2.COLOR_RGB2BGR))

            # wfov_img = data_collector.get_latest_wfov()
            # if wfov_img is not None:
            #     cv2.imshow("WFOV", cv2.cvtColor(wfov_img, cv2.COLOR_RGB2BGR))

            # cv2.waitKey(1)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        # cv2.destroyAllWindows()
        data_collector.stop_camera()
        dataset.stop_image_writer()

        if dataset.episodes_since_last_encoding > 0:
            print(f"Encoding {dataset.episodes_since_last_encoding} remaining episode(s) to video...")
            start_ep = dataset.num_episodes - dataset.episodes_since_last_encoding
            dataset.batch_encode_videos(start_ep, dataset.num_episodes)
            print("Video encoding complete.")

        data_collector.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
