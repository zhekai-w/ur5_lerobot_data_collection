import time
from pathlib import Path
import numpy as np
from pynput import keyboard
import threading
# ROS2 Library
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState, Image
from cv_bridge import CvBridge

# LeRobot Library
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Dualsense
from dualsense_controller import DualSenseController

width_glob  = 640
height_glob = 360
fps_glob    = 30


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

        self.target_fps = fps_glob
        self.bridge = CvBridge()

        # --- Joint state buffers ---
        self.latest_joint_position = None
        self.joint_lock = threading.Lock()

        self.subscription = self.create_subscription(
            JointState, '/joint_states', self.jointstate_callback, 10)

        self.latest_gripper_position = 0.0
        self.gripper_lock = threading.Lock()

        self.gripper_subscription = self.create_subscription(
            JointState, '/gripper/joint_states', self.gripper_callback, 10)

        # --- Camera buffers ---
        self.latest_k4a_image = None
        self.latest_k4a_depth = None
        self.k4a_lock = threading.Lock()

        self.latest_rs_image = None
        self.latest_rs_depth = None
        self.rs_lock = threading.Lock()

        qos = qos_profile_sensor_data
        self.create_subscription(Image, '/camera/azure/color',      self._cb_azure_color, qos)
        self.create_subscription(Image, '/camera/azure/depth',      self._cb_azure_depth, qos)
        self.create_subscription(Image, '/camera/realsense/color',  self._cb_rs_color,    qos)
        self.create_subscription(Image, '/camera/realsense/depth',  self._cb_rs_depth,    qos)

        # --- Recording timer ---
        self.recording_timer = self.create_timer(1.0 / self.target_fps, self.recording_callback)

        # --- DualSense controller ---
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

        # --- Keyboard listener ---
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()

        print("\033[36m Remember to change task description before recording.\033[0m")
        print(f"Collecting data at {self.target_fps}hz")
        print("Press 's' to start recording, 'e' to end episode, 'q' to quit, 'd' to discard")
        print("Press 'create' to start recording, 'cross' to end episode, "
              "'triangle' to quit, 'circle' to discard episode")

    # ------------------------------------------------------------------
    # Camera subscribers
    # ------------------------------------------------------------------

    def _cb_azure_color(self, msg: Image):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        with self.k4a_lock:
            self.latest_k4a_image = img

    def _cb_azure_depth(self, msg: Image):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        with self.k4a_lock:
            self.latest_k4a_depth = img

    def _cb_rs_color(self, msg: Image):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        with self.rs_lock:
            self.latest_rs_image = img

    def _cb_rs_depth(self, msg: Image):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        with self.rs_lock:
            self.latest_rs_depth = img

    # ------------------------------------------------------------------
    # Camera getters
    # ------------------------------------------------------------------

    def get_latest_k4a(self):
        with self.k4a_lock:
            if self.latest_k4a_image is not None and self.latest_k4a_depth is not None:
                return self.latest_k4a_image.copy(), self.latest_k4a_depth.copy()
        return None, None

    def get_latest_rs(self):
        with self.rs_lock:
            if self.latest_rs_image is not None and self.latest_rs_depth is not None:
                return self.latest_rs_image.copy(), self.latest_rs_depth.copy()
        return None, None

    # ------------------------------------------------------------------
    # Joint state
    # ------------------------------------------------------------------

    def get_latest_joint_position(self):
        with self.joint_lock:
            if self.latest_joint_position is not None:
                return self.latest_joint_position.copy()
        return None

    def get_latest_gripper_position(self):
        with self.gripper_lock:
            return self.latest_gripper_position

    def jointstate_callback(self, msg):
        current_position = np.array(list(msg.position), dtype=np.float32)
        with self.joint_lock:
            self.latest_joint_position = current_position

    def gripper_callback(self, msg):
        with self.gripper_lock:
            self.latest_gripper_position = msg.position[0]

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def recording_callback(self):
        with self.lock:
            if not self.is_recording:
                return

            arm_position = self.get_latest_joint_position()
            gripper_position = self.get_latest_gripper_position()
            current_position = np.append(arm_position, gripper_position).astype(np.float32)
            if current_position is None:
                self.get_logger().warn("No joint state available yet, skipping frame")
                return

            if self.previous_position is not None:
                k4a_image, k4a_depth = self.get_latest_k4a()
                rs_image, rs_depth = self.get_latest_rs()

                if k4a_image is None or k4a_depth is None:
                    print("No Azure Kinect frame available yet, skipping frame")
                    return
                if rs_image is None or rs_depth is None:
                    self.get_logger().warn("No RealSense frame available, skipping frame")
                    return

                frame = {
                    "observation.state":            self.previous_position,
                    "observation.images.cam1":       k4a_image,
                    "observation.images.cam2":       rs_image,
                    "observation.images.cam1_depth": k4a_depth,
                    "observation.images.cam2_depth": rs_depth,
                    "action":                        current_position,
                }

                with self.dataset_lock:
                    self.dataset.add_frame(frame, task=self.task)
                self.frame_count += 1

            self.previous_position = current_position

    # ------------------------------------------------------------------
    # Button / keyboard handlers
    # ------------------------------------------------------------------

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
            self.previous_position = None
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
                    self.previous_position = None
                    print("\nStarted recording")
            elif key.char == 'e':
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
                        print("Press 's' to start recording, 'e' to end episode, "
                              "'q' to quit, 'd' to discard")
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
                print("Press 's' to start recording, 'e' to end episode, "
                      "'q' to quit, 'd' to discard")
            elif key.char == 'q':
                print("\nQuitting...")
                self.should_quit = True
        except AttributeError:
            pass

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def stop_camera(self):
        try:
            if self.controller is not None:
                self.controller.deactivate()
                self.get_logger().info("DualSense controller deactivated")
        except Exception as e:
            print(f"Error deactivating DualSense controller: {e}")


def main():
    joints_name = ["shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint",
                   "wrist_3_joint", "shoulder_pan_joint", "gripper_joint"]
    n_joints = len(joints_name)

    width       = width_glob
    height      = height_glob
    rgb_channel = 3
    depth_channel = 3
    root_dir    = './All_Datasets/30hz/test_attention'
    use_videos  = False
    cam_dtype   = "video" if use_videos else "image"

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

    repo_id   = "zhekai-w/ur5_lerobot_dataset"
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
            image_writer_processes=20,
            image_writer_threads=20,
            batch_encoding_size=1,
        )

    rclpy.init()
    data_collector = DataCollector(dataset)

    try:
        while rclpy.ok() and not data_collector.should_quit:
            rclpy.spin_once(data_collector)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
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
