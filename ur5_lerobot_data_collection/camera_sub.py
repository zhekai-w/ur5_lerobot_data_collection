import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.qos import qos_profile_sensor_data

class AzureImageViewer(Node):
    def __init__(self):
        super().__init__('azure_image_viewer')

        # QoS profile for video streaming
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1  # Only keep the latest frame
        )

        # Create subscriber
        self.subscription = self.create_subscription(
            Image,
            '/camera/color/azure_image',
            # '/camera/camera/color/image_raw',
            self.image_callback,
            1
        )
        # Initialize CvBridge
        self.bridge = CvBridge()

        # Counter for statistics
        self.frame_count = 0
        self.last_frame_time = None

        cv2.namedWindow('Azure Kinect Color Stream', cv2.WINDOW_NORMAL)

        self.get_logger().info(f'OpenCV version: {cv2.__version__}')
        self.get_logger().info("Azure Image Viewer Node Started")
        self.get_logger().info("Subscribing to: /camera/color/azure_image")
        self.get_logger().info("Press 'q' to quit, 's' to save screenshot")

    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Update frame counter
            self.frame_count += 1

            # Add frame info overlay
            height, width = cv_image.shape[:2]
            cv2.putText(cv_image, f"Frame: {self.frame_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(cv_image, f"Size: {width}x{height}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Calculate and display FPS
            current_time = self.get_clock().now()
            if self.last_frame_time is not None:
                time_diff = (current_time - self.last_frame_time).nanoseconds / 1e9
                fps = 1.0 / time_diff if time_diff > 0 else 0
                cv2.putText(cv_image, f"FPS: {fps:.1f}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            self.last_frame_time = current_time

            # Display the image

            cv2.imshow('Azure Kinect Color Stream', cv_image)
            # cv2.setWindowProperty('Azure Kinect Color Stream',
            #                     cv2.WND_PROP_FULLSCREEN,
            #                     cv2.WINDOW_NORMAL)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.get_logger().info("User pressed 'q' - shutting down")
                self.cleanup_and_shutdown()
            elif key == ord('s'):
                # Save screenshot
                filename = f"./data/azure_frame_{self.frame_count}.jpg"
                self.frame_count += 1
                cv2.imwrite(filename, cv_image)
                self.get_logger().info(f"Screenshot saved as {filename}")
            elif key == ord('f'):
                # Toggle fullscreen (if supported)
                cv2.setWindowProperty('Azure Kinect Color Stream',
                                    cv2.WND_PROP_FULLSCREEN,
                                    cv2.WINDOW_FULLSCREEN)
            elif key == ord('n'):
                # Normal window
                cv2.setWindowProperty('Azure Kinect Color Stream',
                                    cv2.WND_PROP_FULLSCREEN,
                                    cv2.WINDOW_NORMAL)

        except Exception as e:
            self.get_logger().error(f"Error processing image: {str(e)}")

    def cleanup_and_shutdown(self):
        """Clean up resources and shutdown"""
        cv2.destroyAllWindows()
        self.get_logger().info("Cleanup completed")
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)

    try:
        viewer = AzureImageViewer()
        rclpy.spin(viewer)
    except KeyboardInterrupt:
        print("Keyboard interrupt received")
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        # Cleanup
        cv2.destroyAllWindows()
        if 'viewer' in locals():
            viewer.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
