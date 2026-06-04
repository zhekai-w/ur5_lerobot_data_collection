import threading

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
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
        cm = config['Intrinsic']
        camera_matrix = np.array([
            [float(cm['0_0']), float(cm['0_1']), float(cm['0_2'])],
            [float(cm['1_0']), float(cm['1_1']), float(cm['1_2'])],
            [0, 0, 1]
        ])

        dc = config['Distortion']
        dist_coeffs = np.array(
            [float(dc['k1']), float(dc['k2']), float(dc['t1']), float(dc['t2']), float(dc['k3'])]
        )
    except configparser.Error as e:
        print(e)
    return camera_matrix, dist_coeffs


def undistort_image(image, camera_matrix, dist_coeffs):
    h, w = image.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    return cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)


def get_azure_intrinsics():
    config = Config()
    config.color_resolution = pyk4a.ColorResolution.RES_1080P
    config.depth_mode = pyk4a.DepthMode.NFOV_UNBINNED
    config.camera_fps = pyk4a.FPS.FPS_30

    k4a = PyK4A(config)
    k4a.open()

    calibration = k4a.calibration
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
    output_file = "./src/ur5_lerobot_data_collection/ur5_lerobot_data_collection/config/azure_camera_calibration.ini"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        config.write(f)


class ImagePublisher(Node):
    def __init__(self):
        super().__init__('image_publisher')
        get_azure_intrinsics()
        self.camera_matrix, self.dist_coeffs = read_camera_config(
            "./src/ur5_lerobot_data_collection/ur5_lerobot_data_collection/config/azure_camera_calibration.ini"
        )

        self.color_publisher = self.create_publisher(CompressedImage, '/camera/color/azure_image/compressed', 1)
        self.depth_publisher = self.create_publisher(Image, '/camera/depth/azure_depth', 1)
        self.declare_parameter('jpeg_quality', 90)
        self.jpeg_quality = self.get_parameter('jpeg_quality').get_parameter_value().integer_value

        config = Config()
        config.color_resolution = pyk4a.ColorResolution.RES_1080P
        config.depth_mode = pyk4a.DepthMode.NFOV_UNBINNED
        config.camera_fps = pyk4a.FPS.FPS_30
        config.synchronized_images_only = True
        self.k4a = PyK4A(config)
        self.k4a.start()
        self.calibration = self.k4a.calibration
        self.bridge = CvBridge()

        self.declare_parameter('display', True)
        self.display = self.get_parameter('display').get_parameter_value().bool_value

        self._stop_event = threading.Event()
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()

        self.get_logger().info("Camera Node Initialized and Running")
        self.get_logger().info("Publishing to: /camera/color/azure_image/compressed and /camera/depth/azure_depth")
        self.get_logger().info(f"JPEG quality: {self.jpeg_quality}  Display: {self.display}")

    def _capture_loop(self):
        # get_capture() blocks until the hardware delivers the next frame at hardware rate (30fps).
        # Running in a thread avoids timer/clock drift that caused inconsistent publish rate.
        while not self._stop_event.is_set():
            try:
                capture = self.k4a.get_capture()

                if capture.color is not None:
                    bgr = capture.color[:, :, :3]
                    color_image = np.array(bgr, dtype=np.uint8)
                    # color_image = cv2.rotate(color_image, cv2.ROTATE_180)

                    # undistort = undistort_image(color_image, self.camera_matrix, self.dist_coeffs)

                    ok, encoded = cv2.imencode(
                        '.jpg', color_image,
                        [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
                    )
                    if ok:
                        color_msg = CompressedImage()
                        color_msg.header.stamp = self.get_clock().now().to_msg()
                        color_msg.header.frame_id = "azure_color_frame"
                        color_msg.format = "jpeg"
                        color_msg.data = encoded.tobytes()
                        self.color_publisher.publish(color_msg)

                if capture.transformed_depth is not None:
                    depth_image = capture.transformed_depth
                    depth_image = cv2.rotate(depth_image, cv2.ROTATE_180)
                    depth_msg = self.bridge.cv2_to_imgmsg(depth_image, encoding="16UC1")
                    depth_msg.header.stamp = self.get_clock().now().to_msg()
                    depth_msg.header.frame_id = "azure_depth_frame"
                    self.depth_publisher.publish(depth_msg)

                if self.display and capture.color is not None:
                    cv2.namedWindow("Azure Camera", cv2.WINDOW_NORMAL)
                    cv2.imshow("Azure Camera", color_image)
                    cv2.waitKey(1)

            except Exception as e:
                self.get_logger().error(f"Capture error: {str(e)}")
                break

        self.k4a.stop()

    def destroy_node(self):
        self._stop_event.set()
        self._capture_thread.join(timeout=2.0)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    image_publisher = ImagePublisher()
    try:
        rclpy.spin(image_publisher)
    finally:
        image_publisher.destroy_node()
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
