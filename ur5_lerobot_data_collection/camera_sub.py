import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import numpy as np
import cv2


class AzureImageViewer(Node):
    def __init__(self):
        super().__init__('azure_image_viewer')

        self.subscription = self.create_subscription(
            CompressedImage,
            '/camera/color/azure_image/compressed',
            self.image_callback,
            1,
        )

        self.frame_count = 0
        self.last_frame_time = None
        self.latest_frame = None

        self.declare_parameter('display', True)
        self.display = self.get_parameter('display').get_parameter_value().bool_value
        if self.display:
            cv2.namedWindow('Azure Kinect Color Stream', cv2.WINDOW_NORMAL)

        self.get_logger().info(f'Display: {self.display}')
        self.get_logger().info("Azure Image Viewer Node Started")
        self.get_logger().info("Subscribing to: /camera/color/azure_image/compressed")

    def image_callback(self, msg):
        try:
            buf = np.frombuffer(msg.data, dtype=np.uint8)
            cv_image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if cv_image is None:
                self.get_logger().warn("Failed to decode JPEG frame")
                return

            self.frame_count += 1
            height, width = cv_image.shape[:2]

            current_time = self.get_clock().now()
            fps = 0.0
            if self.last_frame_time is not None:
                time_diff = (current_time - self.last_frame_time).nanoseconds / 1e9
                fps = 1.0 / time_diff if time_diff > 0 else 0.0
            self.last_frame_time = current_time

            print(
                f"\rFPS: {fps:5.1f}  Frame: {self.frame_count}  {width}x{height}",
                end='',
                flush=True,
            )

            if self.display:
                self.latest_frame = cv_image

        except Exception as e:
            self.get_logger().error(f"Error processing image: {str(e)}")


def main(args=None):
    rclpy.init(args=args)
    viewer = AzureImageViewer()

    try:
        while rclpy.ok():
            rclpy.spin_once(viewer, timeout_sec=0.01)
            if viewer.display and viewer.latest_frame is not None:
                cv2.imshow('Azure Kinect Color Stream', viewer.latest_frame)
                cv2.waitKey(1)
    except KeyboardInterrupt:
        pass
    finally:
        print()
        cv2.destroyAllWindows()
        viewer.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()