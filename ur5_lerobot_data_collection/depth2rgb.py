import cv2
import numpy as np
import pyk4a
from pyk4a import Config, PyK4A


def depth_to_bgr_lossless(depth_frame: np.ndarray) -> np.ndarray:
    """Convert uint16 depth to BGR with overlapping middle channel.
    B: bits 15-8 (high byte), G: bits 11-4 (middle, overlaps 4 bits with B and R), R: bits 7-0 (low byte)."""
    b = (depth_frame >> 8).astype(np.uint8)       # bits 15-8
    g = ((depth_frame >> 4) & 0xFF).astype(np.uint8)  # bits 11-4
    r = (depth_frame & 0xFF).astype(np.uint8)      # bits 7-0
    return cv2.merge([r, g, b])

def depth_to_color(depth, min_depth=0, max_depth=5000):
    depth_clipped = np.clip(depth, min_depth, max_depth)
    depth_norm = ((depth_clipped - min_depth)/(max_depth - min_depth) * 255).astype(np.uint8)
    # depth_norm = cv2.normalize(depth_clipped, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
    return colored



def main():
    config = Config()
    config.color_resolution = pyk4a.ColorResolution.RES_1080P
    config.depth_mode = pyk4a.DepthMode.NFOV_UNBINNED
    config.camera_fps = pyk4a.FPS.FPS_30
    config.synchronized_images_only = True

    k4a = PyK4A(config)
    k4a.start()

    try:
        while True:
            capture = k4a.get_capture()
            if capture.transformed_depth is not None:
                depth = capture.transformed_depth
                rgb = depth_to_bgr_lossless(depth)
                color = depth_to_color(depth)
                cv2.imshow("Depth RGB", rgb)
                cv2.imshow("Color RBG", color)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        k4a.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
