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

class DataCollector(Node):

    def __init__(self, dataset):
        super().__init__('joint_state_subscriber')
        self.dataset = dataset
        self.is_recording = False
        self.frame_count = 0
        self.previous_position = None
        self.lock = threading.Lock()

        # Rate limiting for 30 fps recording
        self.target_fps = 30
        self.frame_interval = 1.0 / self.target_fps
        self.last_record_time = 0.0

        self.subscription = self.create_subscription(
            JointState,
            "/joint_states",
            self.jointstate_callback,
            1)

        # Initialize Azure Kinect
        config = Config()
        config.color_resolution = pyk4a.ColorResolution.RES_1080P
        config.depth_mode = pyk4a.DepthMode.NFOV_UNBINNED
        config.camera_fps = pyk4a.FPS.FPS_30
        config.synchronized_images_only = True

        self.k4a = PyK4A(config)
        self.k4a.start()

        # Start keyboard listener
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()

        print("Press 's' to start recording, 'e' to end episode, 'q' to quit")

    def on_press(self, key):
        try:
            if key.char == 's':
                with self.lock:
                    self.is_recording = True
                    self.frame_count = 0
                    self.previous_position = None
                    self.last_record_time = 0.0
                    print("\nStarted recording")
            elif key.char == 'e':
                with self.lock:
                    if self.is_recording and self.frame_count > 0:
                        self.is_recording = False
                        try:
                            self.dataset.save_episode()
                            print(f"Episode saved ({self.frame_count} frames)")
                            print(f"Total episodes: {self.dataset.num_episodes}")
                        except Exception as e:
                            print(f"Error saving episode: {e}")
                        finally:
                            self.previous_position = None
                            self.frame_count = 0
                    else:
                        print("No frames recorded yet")
            elif key.char == 'q':
                print("Quitting...")
                rclpy.shutdown()
        except AttributeError:
            pass

    def capture_azure_image(self):
        """Capture image from Azure Kinect camera"""
        try:
            capture = self.k4a.get_capture()
            if capture.color is not None:
                # Get BGR image (1920x1080x3)
                bgr = capture.color[:, :, :3]
                color_image = np.array(bgr, dtype=np.uint8)
                return color_image
        except Exception as e:
            self.get_logger().error(f"Failed to capture image: {e}")
        return None

    # TODO: change joints order from [shoulder_lift, elbow, wrist_1, wrist_2, wrist_3, shoulder_pen]
    # to [shoulder_pen, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3]
    def jointstate_callback(self, msg):
        with self.lock:
            if not self.is_recording:
                return

            # Rate limiting: skip if not enough time has passed since last frame
            current_time = time.time()
            if current_time - self.last_record_time < self.frame_interval:
                return

            # Get current position with gripper (0 for closed)
            current_position = np.array(list(msg.position) + [0], dtype=np.float32)

            # Only record if we have a previous position (for action)
            if self.previous_position is not None:
                # Capture image from Azure Kinect camera
                image = self.capture_azure_image()

                if image is None:
                    self.get_logger().warn("Failed to capture image, skipping frame")
                    return

                frame = {
                    "observation.state": self.previous_position,
                    "observation.images.cam1": image,
                    "action": current_position,  # Current position is the action
                }

                self.dataset.add_frame(frame, task="reach target")
                self.frame_count += 1
                self.last_record_time = current_time

            # Update previous position for next frame
            self.previous_position = current_position


def main():
    # Robot configuration
    n_joints = 7
    # joints_name = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3", "gripper"]
    joints_name = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"]
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
        "action": {
            "dtype": "float32",
            "shape": (n_joints,),
            "names": list(joints_name),
        },
    }

    dataset = LeRobotDataset.create(
        repo_id="zhekai-w/ur5_lerobot_dataset",
        fps=30,
        features=features,
        root=root_dir,
        robot_type="ur5",
        use_videos=True,
        batch_encoding_size=1,
    )

    rclpy.init()
    data_collector = DataCollector(dataset)

    try:
        rclpy.spin(data_collector)
    except KeyboardInterrupt:
        pass
    finally:
        # Stop Azure Kinect camera
        try:
            data_collector.k4a.stop()
            data_collector.get_logger().info("Azure Kinect camera stopped")
        except Exception as e:
            print(f"Error stopping camera: {e}")

        data_collector.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
