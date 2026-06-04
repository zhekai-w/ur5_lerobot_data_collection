import cv2

DEVICE = "/dev/video0"
WIN = "Camera Feed  |  Controls"


# WFOV USB camera
wfov_cap = cv2.VideoCapture(DEVICE)
wfov_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
wfov_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
wfov_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while True:
    try:
        ret, frame = wfov_cap.read()

        # frame = cv2.resize(frame, (width_glob, height_glob), interpolation=cv2.INTER_AREA)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.imshow(WIN, frame)
        cv2.waitKey(1)

    except Exception as e:
        print(f"WFOV capture error: {e}")
        time.sleep(0.01)
