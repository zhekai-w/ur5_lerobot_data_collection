import cv2
import subprocess
from datetime import datetime
from pathlib import Path

DEVICE = "/dev/video6"
WIN = "Camera Feed  |  Controls"
SAVE_DIR = Path.home() / "work/videos"

# (label, v4l2_name, real_min, real_max, default, type)
CONTROLS = [
    ("Brightness [-64..64]",   "brightness",              -64, 64,   0,   "int"),
    ("Contrast",               "contrast",                  0, 64,  32,   "int"),
    ("Saturation",             "saturation",                0, 128, 64,   "int"),
    ("Hue [-40..40]",          "hue",                     -40, 40,   0,   "int"),
    ("WB Auto  0=off 1=on",    "white_balance_automatic",   0,  1,   1,   "bool"),
    ("Gamma",                  "gamma",                    72, 500, 100,   "int"),
    ("Gain",                   "gain",                      0, 100,  0,   "int"),
    ("Power Line 0/1Hz/2Hz",   "power_line_frequency",      0,  2,   1,   "menu"),
    ("WB Temperature",         "white_balance_temperature",2800,6500,4600,"int"),
    ("Sharpness",              "sharpness",                 0,  6,   2,   "int"),
    ("Backlight Comp",         "backlight_compensation",    0,  2,   1,   "int"),
    ("Auto Exp 1=Man 3=Auto",  "auto_exposure",             0,  3,   3,   "menu"),
    ("Exposure Time",          "exposure_time_absolute",    1, 5000,157,  "int"),
    ("Dyn Framerate  0=off",   "exposure_dynamic_framerate",0,  1,   0,  "bool"),
]

current_values = {}   # what we last wrote (drives trackbars)
hw_values = {}        # what the camera reports back (for display)
writer = None
recording = False
record_path = ""

_ALL_CTRL_NAMES = ",".join(name for _, name, *_ in CONTROLS)

def v4l2_set(name, value):
    subprocess.run(
        ["v4l2-ctl", "-d", DEVICE, "-c", f"{name}={value}"],
        capture_output=True
    )

def v4l2_read_all():
    """Read all control values from the hardware and update hw_values."""
    result = subprocess.run(
        ["v4l2-ctl", "-d", DEVICE, "-C", _ALL_CTRL_NAMES],
        capture_output=True, text=True
    )
    for line in result.stdout.splitlines():
        if ":" in line:
            key, _, val = line.partition(":")
            try:
                hw_values[key.strip()] = int(val.strip())
            except ValueError:
                pass

def make_callback(name, real_min):
    def cb(trackbar_val):
        real = trackbar_val + real_min
        current_values[name] = real
        v4l2_set(name, real)
    return cb

def init_camera():
    for _, name, _, _, default, _ in CONTROLS:
        v4l2_set(name, default)

def start_recording(w, h, fps=30.0):
    global writer, recording, record_path
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    record_path = str(SAVE_DIR / f"capture_{ts}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(record_path, fourcc, fps, (w, h))
    recording = True
    print(f"Recording → {record_path}")

def stop_recording():
    global writer, recording
    if writer:
        writer.release()
        writer = None
    recording = False
    print(f"Saved: {record_path}")

DISPLAY_LABELS = [
    ("Brightness",  "brightness"),
    ("Contrast",    "contrast"),
    ("Saturation",  "saturation"),
    ("Hue",         "hue"),
    ("WB Auto",     "white_balance_automatic"),
    ("WB Temp",     "white_balance_temperature"),
    ("Gamma",       "gamma"),
    ("Gain",        "gain"),
    ("Sharpness",   "sharpness"),
    ("Backlight",   "backlight_compensation"),
    ("Auto Exp",    "auto_exposure"),
    ("Exposure",    "exposure_time_absolute"),
    ("Dyn FPS",     "exposure_dynamic_framerate"),
    ("Power Line",  "power_line_frequency"),
]

def draw_overlay(frame):
    h, w = frame.shape[:2]

    # ── left panel ────────────────────────────────────────────────────────────
    panel_w = 175
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    line_h = 16
    y0 = 14
    for i, (label, key) in enumerate(DISPLAY_LABELS):
        val = hw_values.get(key, "?")
        # Human-readable for menu controls
        if isinstance(val, int):
            if key == "auto_exposure":
                val = "Manual" if val == 1 else "Auto"
            elif key == "white_balance_automatic":
                val = "On" if val == 1 else "Off"
            elif key == "exposure_dynamic_framerate":
                val = "On" if val == 1 else "Off"
            elif key == "power_line_frequency":
                val = ["Off", "50 Hz", "60 Hz"][val]
        text = f"{label}: {val}"
        cv2.putText(frame, text, (6, y0 + i * line_h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 230, 180), 1)

    # ── bottom bar ────────────────────────────────────────────────────────────
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - 26), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay2, 0.55, frame, 0.45, 0, frame)
    rec_hint = "[r] stop" if recording else "[r] record"
    cv2.putText(frame, f"{rec_hint}   [q] quit", (8, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # ── REC indicator ─────────────────────────────────────────────────────────
    if recording:
        cv2.circle(frame, (w - 18, h - 13), 7, (0, 0, 220), -1)
        cv2.putText(frame, "REC", (w - 52, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 220), 1)
    return frame

# ── init ──────────────────────────────────────────────────────────────────────
init_camera()

cap = cv2.VideoCapture(DEVICE)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN, 700, 820)

for label, name, real_min, real_max, default, _ in CONTROLS:
    trackbar_max = real_max - real_min
    trackbar_default = default - real_min
    current_values[name] = default
    cv2.createTrackbar(label, WIN, trackbar_default, trackbar_max, make_callback(name, real_min))

# ── main loop ─────────────────────────────────────────────────────────────────
frame_count = 0
v4l2_read_all()   # initial read before first frame

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 30 == 0:   # refresh hw values ~once per second
        v4l2_read_all()

    if recording and writer:
        writer.write(frame)

    frame = draw_overlay(frame)
    cv2.imshow(WIN, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        if recording:
            stop_recording()
        else:
            start_recording(frame_w, frame_h)

if recording:
    stop_recording()
cap.release()
cv2.destroyAllWindows()
