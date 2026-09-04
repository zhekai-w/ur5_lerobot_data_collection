import cv2
import time
import subprocess

DEVICE = 2
WIN = "Camera Feed  |  Controls"


DEVICE_PATH = "/dev/video" + str(DEVICE)

def v4l2_set(name, value):
    subprocess.run(["v4l2-ctl", "-d", DEVICE_PATH, "-c", f"{name}={value}"],
                   capture_output=True)


# tame exposure so image not blown out and fps not capped
v4l2_set("auto_exposure", 3)              # 3 = auto
v4l2_set("exposure_dynamic_framerate", 0)
# if still overexposed, switch to manual:
# v4l2_set("auto_exposure", 1)
# v4l2_set("exposure_time_absolute", 157)

# WFOV USB camera
wfov_cap = cv2.VideoCapture(DEVICE)
wfov_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
wfov_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
wfov_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
wfov_cap.set(cv2.CAP_PROP_FPS, 30)

cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

while True:
    try:
        ret, frame = wfov_cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        # frame = cv2.resize(frame, (width_glob, height_glob), interpolation=cv2.INTER_AREA)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imshow(WIN, frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break


    except Exception as e:
        print(f"WFOV capture error: {e}")
        time.sleep(0.01)

wfov_cap.release()
cv2.destroyAllWindows()
