import time
import numpy as np
from pynput import keyboard
import threading
# ROS2 Library
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

# LeRobot Library
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Azure Kinect
import pyk4a
from pyk4a import Config, PyK4A

# Realsense
import pyrealsense2 as rs


class DataCollector(Node):

    def __init__(self, dataset):
        super().__init__('joint_state_subscriber')
        self.dataset = dataset
        self.is_recording = False
        self.frame_count = 0
        self.previous_position = None
        self.lock = threading.Lock()

        # Recording fps
        self.target_fps = 30

        # Buffer for latest joint state
        self.latest_joint_position = None
        self.joint_lock = threading.Lock()

        # Subscribe to joint states (just for buffering)
        self.subscription = self.create_subscription(
            JointState,
            "/joint_states",
            self.jointstate_callback,
            1)

        # Timer for recording at exactly 30fps
        timer_period = 1.0 / self.target_fps
        self.recording_timer = self.create_timer(timer_period, self.recording_callback)

        # Initialize Azure Kinect
        config = Config()
        config.color_resolution = pyk4a.ColorResolution.RES_1080P
        config.depth_mode = pyk4a.DepthMode.NFOV_UNBINNED
        config.camera_fps = pyk4a.FPS.FPS_30
        config.synchronized_images_only = True

        self.k4a = PyK4A(config)
        self.k4a.start()

        # Initialize Realsense
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 1920, 1080, rs.format.rgb8, 30)
        self.pipeline.start(config)


        # Async camera capture variables
        # Azure
        self.latest_k4a_image = None
        self.k4a_image_lock = threading.Lock()
        # Realsense
        self.lastest_rs_image = None
        self.rs_image_lock = threading.Lock()

        self.camera_running = True

        self.rs_thread = threading.Thread(target=self._rs_capture_loop, daemon=True)
        self.k4a_thread = threading.Thread(target=self._k4a_capture_loop, daemon=True)
        self.rs_thread.start()
        self.k4a_thread.start()

        # Start keyboard listener
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()

        print("Press 's' to start recording, 'e' to end episode, 'q' to quit")

    def _k4a_capture_loop(self):
        """Continuously capture images from Azure Kinect in a separate thread"""
        while self.camera_running:
            try:
                capture = self.k4a.get_capture()
                if capture.color is not None:
                    # Get BGR image (1920x1080x3)
                    # bgr = capture.color[:, :, :3]
                    rgb = capture.color[:, :, 2::-1]
                    color_image = np.array(rgb, dtype=np.uint8)
                    with self.k4a_image_lock:
                        self.latest_k4a_image = color_image
            except Exception as e:
                self.get_logger().error(f"Azure Kinect capture error: {e}")
                time.sleep(0.01)  # Brief sleep on error to avoid tight loop

    def _rs_capture_loop(self):
        """Continuously capture images from Azure Kinect in a separate thread"""
        while self.camera_running:
            try:
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if color_frame:
                    # pyrealsense2 already delivers rgb8 as requested in config
                    color_image = np.asanyarray(color_frame.get_data(), dtype=np.uint8)
                    with self.rs_image_lock:
                        self.latest_rs_image = color_image
            except RuntimeError as e:
                self.get_logger().error(f"Realsense capture error: {e}")
                time.sleep(0.01)  # Brief sleep on error to avoid tight loop

    def get_latest_k4a_image(self):
        """Get the most recent captured image"""
        with self.k4a_image_lock:
            if self.latest_k4a_image is not None:
                return self.latest_k4a_image.copy()
        return None

    def get_latest_rs_image(self):
        """Get the most recent captured image"""
        with self.rs_image_lock:
            if self.latest_rs_image is not None:
                return self.latest_rs_image.copy()
        return None

    def get_latest_joint_position(self):
        """Get the most recent joint position"""
        with self.joint_lock:
            if self.latest_joint_position is not None:
                return self.latest_joint_position.copy()
        return None

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
                        print("No frames recorded yet")

                # Save episode outside the lock to avoid blocking callbacks
                if should_save:
                    try:
                        print("Saving episode...")
                        self.dataset.save_episode()
                        print(f"Episode saved ({frame_count_to_report} frames)")
                        print(f"Total episodes: {self.dataset.num_episodes}")
                    except Exception as e:
                        print(f"Error saving episode: {e}")
            elif key.char == 'q':
                print("Quitting...")
                raise SystemExit()
        except AttributeError:
            pass

    def jointstate_callback(self, msg):
        """Just buffer the latest joint state - no recording logic here"""
        # Get current position with gripper (0 for closed)
        current_position = np.array(list(msg.position) + [0], dtype=np.float32)
        with self.joint_lock:
            self.latest_joint_position = current_position

    def recording_callback(self):
        """Timer callback at exactly 30fps for recording frames"""
        with self.lock:
            if not self.is_recording:
                return

            # Get latest joint position
            current_position = self.get_latest_joint_position()
            if current_position is None:
                self.get_logger().warn("No joint state available yet, skipping frame")
                return

            # Only record if we have a previous position (for action)
            if self.previous_position is not None:
                # Get the latest cached image from async capture thread
                k4a_image = self.get_latest_k4a_image()
                rs_image = self.get_latest_rs_image()

                if k4a_image is None:
                    self.get_logger().warn("No Azure Kinect image available yet, skipping frame")
                    return
                if rs_image is None:
                    self.get_logger().warn("No RealSense image available, skipping frame")
                    return

                frame = {
                    "observation.state": self.previous_position,
                    "observation.images.cam1": k4a_image,
                    "observation.images.cam2": rs_image,
                    "action": current_position,  # Current position is the action
                }

                self.dataset.add_frame(frame, task="reach target")
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


def main():
    # Robot configuration
    joints_name = ["shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint",
        "wrist_3_joint", "shoulder_pan_joint", "gripper_joint"]
    # joints_name = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"]
    n_joints = len(joints_name)

    # Camera configuration (Azure Kinect 1080p)
    width = 1920
    height = 1080
    channel = 3
    cam_dtype = "video"
    root_dir = "./dataset"
    # TODO: Change this to actual task disscription
    task = "test"

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (n_joints,),
            "names": list(joints_name),
        },
        "observation.images.cam1": {
            "dtype": cam_dtype,
            "shape": (height, width, channel),
            "names": ["height", "width", "channel"],
        },
        "observation.images.cam2": {
            "dtype": cam_dtype,
            "shape": (height, width, channel),
            "names": ["height", "width", "channel"],
        },
        "action": {
            "dtype": "float32",
            "shape": (n_joints,),
            "names": list(joints_name),
        },
    }

    dataset = LeRobotDataset.create(
        repo_id="zhekai-w/ur5_lerobot_dataset",
        fps=20,
        features=features,
        root=root_dir,
        robot_type="ur5",
        use_videos=True,
        video_backend="pyav",
        image_writer_processes=20,
        image_writer_threads=20,  # Async image writing for better fps
        # batch_encoding_size=20,
    )

    rclpy.init()
    data_collector = DataCollector(dataset)

    try:
        rclpy.spin(data_collector)
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
