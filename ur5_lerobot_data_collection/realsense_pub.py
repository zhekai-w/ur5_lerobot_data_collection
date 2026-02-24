import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import pyrealsense2 as rs
import cv2
import numpy as np
import os
import configparser

def read_camera_config(filepath):
    camera_matrix = None
    dist_coeffs = None
    config = configparser.ConfigParser()
    config.read(filepath)
    try:
        # Read camera matrix
        cm = config['Intrinsic']
        camera_matrix = np.array([
            [float(cm['0_0']), float(cm['0_1']), float(cm['0_2'])],
            [float(cm['1_0']), float(cm['1_1']), float(cm['1_2'])],
            [0, 0, 1]
        ])

        # Read distortion coefficient
        dc = config['Distortion']
        dist_coeffs = np.array(
            [float(dc['k1']), float(dc['k2']), float(dc['t1']), float(dc['t2']), float(dc['k3'])]
        )
    except configparser.Error as e:
        print(e)
    return camera_matrix, dist_coeffs


def undistort_image(image, camera_matrix, dist_coeffs):
    """
    Undistort an image using camera calibration parameters
    param image: Input image
    param camera_matrix: Camera matrix
    param dist_coeffs: Distortion ceofficients
    return: Undistorted image
    """
    h, w = image.shape[:2]

    new_camera_matrix,  roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    undistorted = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)

    return undistorted

def save_realsense_intrinsics(filepath, intrinsics):
    """Save RealSense color camera intrinsics and distortion to an ini file."""
    # Build camera matrix from intrinsics
    camera_matrix = np.array([
        [intrinsics.fx, 0.0, intrinsics.ppx],
        [0.0, intrinsics.fy, intrinsics.ppy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    # Flatten distortion coefficients
    dist_coeffs = np.array(intrinsics.coeffs, dtype=np.float64).flatten()
    # Prepare config sections
    config = configparser.ConfigParser()
    config['Intrinsic'] = {
        '0_0': f"{camera_matrix[0,0]:.6f}",
        '0_1': f"{camera_matrix[0,1]:.6f}",
        '0_2': f"{camera_matrix[0,2]:.6f}",
        '1_0': f"{camera_matrix[1,0]:.6f}",
        '1_1': f"{camera_matrix[1,1]:.6f}",
        '1_2': f"{camera_matrix[1,2]:.6f}"
    }
    config['Distortion'] = {
        'k1': f"{dist_coeffs[0]:.6f}",
        'k2': f"{dist_coeffs[1]:.6f}",
        't1': f"{dist_coeffs[2]:.6f}",
        't2': f"{dist_coeffs[3]:.6f}",
        'k3': f"{dist_coeffs[4]:.6f}"
    }
    # Ensure directory exists and write file
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        config.write(f)

class ImagePublisher(Node):
    def __init__(self):
        super().__init__('image_publisher')  # Node name

        # Create the publisher
        self.publisher_ = self.create_publisher(Image, '/camera/color/for_shelf', 10)
        timer_period = 0.03  # Publish at 30Hz
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # Initialize the RealSense pipeline
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
        # Start streaming
        self.pipeline.start(config)
        # Get the intrinsics after starting the pipeline
        profile = self.pipeline.get_active_profile()
        color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
        color_intrinsics = color_profile.get_intrinsics()

        # self.camera_matrix = np.array([
        #     [color_intrinsics.fx, 0.0, color_intrinsics.ppx],
        #     [0.0, color_intrinsics.fy, color_intrinsics.ppy],
        #     [0.0, 0.0, 1.0]
        # ], dtype=np.float64)
        # # Flatten distortion coefficients
        # self.dist_coeffs = np.array(color_intrinsics.coeffs, dtype=np.float64).flatten()

        # Get RealSense color intrinsics and export to ini
        ini_path = os.path.join(os.path.dirname(__file__), 'camera_calibration.ini')
        save_realsense_intrinsics(ini_path, color_intrinsics)

        # Initialize CvBridge
        self.bridge = CvBridge()

        self.get_logger().info("Camera Node Initialized and Running")
        self.get_logger().info("Use \"ros2 topic list\" to see all topics")


    def timer_callback(self):
        # Get the frames
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        # Check if a frame is available
        if not color_frame:
            return

        # Convert to numpy array and resize
        color_image = np.asanyarray(color_frame.get_data())

        # Undistort image
        # undistort = undistort_image(color_image, self.camera_matrix, self.dist_coeffs)

        # Create and publish the image message
        msg = self.bridge.cv2_to_imgmsg(color_image, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    image_publisher = ImagePublisher()
    rclpy.spin(image_publisher)
    # Stop the RealSense pipeline after spinning
    image_publisher.pipeline.stop()
    image_publisher.destroy_node()  # Clean up before exiting
    rclpy.shutdown()

if __name__ == '__main__':
    main()
