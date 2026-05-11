import threading
import time

import cv2
import numpy as np
import pyk4a
import pyrealsense2 as rs
import rclpy
from cv_bridge import CvBridge
from pyk4a import Config, PyK4A
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

WIDTH = 640
HEIGHT = 360
FPS = 30


def convert_depth_to_rgb(
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


class AllCamPub(Node):
    def __init__(self):
        super().__init__("all_cam_pub")

        self.bridge = CvBridge()

        # --- Publishers ---
        qos = qos_profile_sensor_data
        self.pub_azure_color = self.create_publisher(Image, "/camera/azure/color", qos)
        self.pub_azure_depth = self.create_publisher(Image, "/camera/azure/depth", qos)
        self.pub_rs_color = self.create_publisher(Image, "/camera/realsense/color", qos)
        self.pub_rs_depth = self.create_publisher(Image, "/camera/realsense/depth", qos)

        # --- Azure Kinect ---
        k4a_cfg = Config()
        k4a_cfg.color_resolution = pyk4a.ColorResolution.RES_1080P
        k4a_cfg.depth_mode = pyk4a.DepthMode.NFOV_UNBINNED
        k4a_cfg.camera_fps = pyk4a.FPS.FPS_30
        k4a_cfg.synchronized_images_only = True
        self.k4a = PyK4A(k4a_cfg)
        self.k4a.start()

        self.latest_azure_color = None
        self.latest_azure_depth = None
        self.azure_lock = threading.Lock()

        # --- RealSense ---
        rs_cfg = rs.config()
        rs_cfg.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.rgb8, FPS)
        rs_cfg.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
        self.rs_pipeline = rs.pipeline()
        self.rs_pipeline.start(rs_cfg)

        self.latest_rs_color = None
        self.latest_rs_depth = None
        self.rs_lock = threading.Lock()

        # --- Capture threads ---
        self.running = True
        self.azure_thread = threading.Thread(target=self._azure_loop, daemon=True)
        self.rs_thread = threading.Thread(target=self._rs_loop, daemon=True)
        self.azure_thread.start()
        self.rs_thread.start()

        # --- Publish timer ---
        self.timer = self.create_timer(1.0 / FPS, self._publish)

        # Local preview — reads direct from capture buffers, no ROS serialization
        # cv2.namedWindow("AllCamPub Preview", cv2.WINDOW_NORMAL)
        # self.display_timer = self.create_timer(1.0 / FPS, self._display)

        self.get_logger().info("AllCamPub started")
        self.get_logger().info("  /camera/azure/color   bgr8  640x360")
        self.get_logger().info("  /camera/azure/depth   16UC1 640x360")
        self.get_logger().info("  /camera/realsense/color rgb8 640x360")
        self.get_logger().info("  /camera/realsense/depth 16UC1 640x360")

    # ------------------------------------------------------------------
    # Capture loops
    # ------------------------------------------------------------------

    def _azure_loop(self):
        while self.running:
            try:
                capture = self.k4a.get_capture()
                if capture is None:
                    continue
                if capture.color is None or capture.transformed_depth is None:
                    continue

                bgr = np.ascontiguousarray(capture.color[:, :, :3], dtype=np.uint8)
                bgr = cv2.resize(bgr, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
                depth = cv2.resize(
                    capture.transformed_depth,
                    (WIDTH, HEIGHT),
                    interpolation=cv2.INTER_NEAREST,
                )

                with self.azure_lock:
                    self.latest_azure_color = bgr
                    self.latest_azure_depth = depth
            except Exception as e:
                self.get_logger().error(f"Azure capture error: {e}")
                time.sleep(0.01)

    def _rs_loop(self):
        while self.running:
            try:
                frames = self.rs_pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue

                color = np.asanyarray(color_frame.get_data(), dtype=np.uint8)
                depth = np.asanyarray(depth_frame.get_data(), dtype=np.uint16)

                with self.rs_lock:
                    self.latest_rs_color = color
                    self.latest_rs_depth = depth
            except RuntimeError as e:
                self.get_logger().error(f"RealSense capture error: {e}")
                time.sleep(0.01)

    # ------------------------------------------------------------------
    # Publish
    # ------------------------------------------------------------------

    def _publish(self):
        now = self.get_clock().now().to_msg()

        with self.azure_lock:
            azure_color = self.latest_azure_color
            azure_depth = self.latest_azure_depth

        with self.rs_lock:
            rs_color = self.latest_rs_color
            rs_depth = self.latest_rs_depth

        if azure_color is not None:
            msg = self.bridge.cv2_to_imgmsg(azure_color, encoding="bgr8")
            msg.header.stamp = now
            msg.header.frame_id = "azure_color_frame"
            self.pub_azure_color.publish(msg)

        if azure_depth is not None:
            msg = self.bridge.cv2_to_imgmsg(
                convert_depth_to_rgb(azure_depth), encoding="rgb8"
            )
            msg.header.stamp = now
            msg.header.frame_id = "azure_color_frame"
            self.pub_azure_depth.publish(msg)

        if rs_color is not None:
            msg = self.bridge.cv2_to_imgmsg(rs_color, encoding="rgb8")
            msg.header.stamp = now
            msg.header.frame_id = "realsense_color_frame"
            self.pub_rs_color.publish(msg)

        if rs_depth is not None:
            msg = self.bridge.cv2_to_imgmsg(
                convert_depth_to_rgb(rs_depth), encoding="rgb8"
            )
            msg.header.stamp = now
            msg.header.frame_id = "realsense_color_frame"
            self.pub_rs_depth.publish(msg)

    # ------------------------------------------------------------------
    # Display (local, no ROS)
    # ------------------------------------------------------------------

    def _display(self):
        blank = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

        with self.azure_lock:
            ac = self.latest_azure_color
            ad = self.latest_azure_depth

        with self.rs_lock:
            rc = self.latest_rs_color
            rd = self.latest_rs_depth

        def prep(img, label, convert_rgb=False):
            tile = img.copy() if img is not None else blank.copy()
            if convert_rgb and img is not None:
                tile = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
            cv2.putText(
                tile, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            return tile

        tiles = [
            prep(ac, "Azure Color"),
            prep(convert_depth_to_rgb(ad) if ad is not None else None, "Azure Depth"),
            prep(rc, "RS Color", convert_rgb=True),
            prep(convert_depth_to_rgb(rd) if rd is not None else None, "RS Depth"),
        ]

        grid = cv2.vconcat([cv2.hconcat(tiles[:2]), cv2.hconcat(tiles[2:])])
        cv2.imshow("AllCamPub Preview", grid)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self):
        self.running = False
        self.azure_thread.join(timeout=2.0)
        self.rs_thread.join(timeout=2.0)
        self.k4a.stop()
        self.rs_pipeline.stop()


def main(args=None):
    rclpy.init(args=args)
    node = AllCamPub()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
