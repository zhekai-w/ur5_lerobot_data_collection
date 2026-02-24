import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import configparser
import pyk4a
from pyk4a import Config, PyK4A
from rclpy.qos import qos_profile_sensor_data

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

def get_azure_intrinsics():
    config = Config()
    config.color_resolution = pyk4a.ColorResolution.RES_1080P
    config.depth_mode = pyk4a.DepthMode.NFOV_UNBINNED
    config.camera_fps = pyk4a.FPS.FPS_30

    k4a = PyK4A(config)
    k4a.open()

    # Get calibration data
    calibration = k4a.calibration

    # Color camera intrinsics
    camera_matrix = calibration.get_camera_matrix(pyk4a.CalibrationType.COLOR)
    print("Color Camera Matrix:")
    print(camera_matrix)
    dist_coeffs = calibration.get_distortion_coefficients(pyk4a.CalibrationType.COLOR)
    print("\nColor distortion:")
    print(dist_coeffs)
    k4a.close()

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
    output_file = "./src/shelf/shelf_pose_est/shelf_pose_est/config/azure_camera_calibration.ini"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        config.write(f)



class ImagePublisher(Node):
    def __init__(self):
        super().__init__('image_publisher')  # Node name
        get_azure_intrinsics()
        self.camera_matrix, self.dist_coeffs = read_camera_config("./src/shelf/shelf_pose_est/shelf_pose_est/config/azure_camera_calibration.ini")

        # Create the publisher
        self.color_publisher = self.create_publisher(Image, '/camera/color/azure_image', 1)
        self.depth_publisher = self.create_publisher(Image, '/camera/depth/azure_depth', 1)
        timer_period = 0.033  # Publish at 30Hz
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # Configure azure kinect dk
        config = Config()
        config.color_resolution = pyk4a.ColorResolution.RES_1080P
        config.depth_mode = pyk4a.DepthMode.NFOV_UNBINNED
        config.camera_fps = pyk4a.FPS.FPS_30
        config.synchronized_images_only = True
        # Start streaming
        self.k4a = PyK4A(config)
        self.k4a.start()
        self.calibration = self.k4a.calibration
        # Initialize CvBridge
        self.bridge = CvBridge()

        self.get_logger().info("Camera Node Initialized and Running")
        self.get_logger().info("Publishing to: /camera/color/azure_image and /camera/depth/azure_depth")
        self.get_logger().info("Use \"ros2 topic list\" to see all topics")


    def timer_callback(self):
        try:
            # Get the frames
            capture = self.k4a.get_capture()

            # Publish color image
            if capture.color is not None:
                bgr = capture.color[:, :, :3]
                color_image = np.array(bgr, dtype=np.uint8)

                # Undistort image
                # undistort = undistort_image(color_image, self.camera_matrix, self.dist_coeffs)

                # Create and publish the image message
                color_msg = self.bridge.cv2_to_imgmsg(color_image, encoding="bgr8")
                color_msg.header.stamp = self.get_clock().now().to_msg()
                color_msg.header.frame_id = "azure_color_frame"
                self.color_publisher.publish(color_msg)

            # Publish depth image
            # if capture.transformed_depth is not None:
            #     depth_image = capture.transformed_depth
            #     # encoding="passthrough" same data type
            #     depth_msg = self.bridge.cv2_to_imgmsg(depth_image, encoding="16UC1")
            #     depth_msg.header.stamp = self.get_clock().now().to_msg()
            #     depth_msg.header.frame_id = "azure_depth_frame"
            #     self.depth_publisher.publish(depth_msg)

        except Exception as e:
            self.get_logger().error(f"Timer callback error:{str(e)}")
            self.k4a.stop()
            self.destroy_node()  # Clean up before exiting
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    image_publisher = ImagePublisher()
    rclpy.spin(image_publisher)
    # Stop the Azure Kinect DK pipeline after spinning
    image_publisher.k4a.stop()
    image_publisher.destroy_node()  # Clean up before exiting
    rclpy.shutdown()

if __name__ == '__main__':
    main()
