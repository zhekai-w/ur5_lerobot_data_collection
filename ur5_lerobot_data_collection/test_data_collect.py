import numpy as np
from pynput import keyboard
# ROS2 and UR Library
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

# LeRobot Library
from lerobot.datasets.lerobot_dataset import LeRobotDataset

def record():
    # Robot configuration
    n_joints = 7
    # joints_name = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3", "gripper"]
    joints_name = ["0", "1", "2", "3", "4", "5", "6", "7"]

    # Camera configuration
    width = 640
    height = 480
    channel = 3
    # Cam dtype should be "video" or "image"
    cam_dtype = "video"

    # Root diractory where the dataset will be stored 
    root_dir = "./dataset"

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(n_joints),), 
            "names": list(joints_name),
        },
        "observation.images": {
            "dtype": cam_dtype, 
            "shape": (height, width, channel),
            "names": ["height", "width", "channel"],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(n_joints),), 
            "names": list(joints_name),
        },
    }

    dataset = LeRobotDataset.create(
        # cls,
        repo_id="zhekai-w/ur5_lerobot_dataset",
        fps=30,
        features=features,
        root=root_dir,
        robot_type="ur5",
        use_videos=True,
        batch_encoding_size=1,
    )
    # 3. Record episodes
    num_episodes = 10
    frames_per_episode = 100

    for ep_idx in range(num_episodes):
        print(f"Recording episode {ep_idx}...")
        
        for frame_idx in range(frames_per_episode):
            # Simulate data (replace with your actual robot interface)
            state = np.random.randn(7).astype(np.float32)
            image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            action = np.random.randn(7).astype(np.float32)
            
            frame = {
                "observation.state": state,
                "observation.images.cam1": image,
                "action": action,
            }
            
            dataset.add_frame(frame, task="reach target")
        
        # Save episode to disk
        dataset.save_episode()
        print(f"✓ Episode {ep_idx} saved")

    print("Recording complete!")

    # 4. Check dataset properties
    print(f"Total episodes: {dataset.num_episodes}")
    print(f"Total frames: {dataset.num_frames}")
    print(f"FPS: {dataset.fps}")



class DataCollector(Node):

    def __init__(self):
        super().__init__('joint_state_subscriber')
        self.subscription = self.create_subscription(
            JointState, 
            "/joint_states", 
            self.jointstate_callback, 
            1)
    
    def jointstate_callback(self, msg):
        print("Joint State(Position):", msg.position)
        print("Data Type:", type(msg.position))
        print("size of data:", len(msg.position))

def main():
    rclpy.init()
    data_collector = DataCollector()
    rclpy.spin(data_collector)

    data_collector.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
    