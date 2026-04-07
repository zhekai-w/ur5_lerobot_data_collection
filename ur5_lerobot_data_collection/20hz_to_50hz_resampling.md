# 20 Hz → 50 Hz Data Resampling: Options and Trade-offs

## Current Setup (from `data_collect.py`)

| Component | Detail |
|---|---|
| Recording rate | 20 Hz (timer period = 50 ms) |
| Camera hardware rate | 30 FPS (Azure Kinect + RealSense) |
| Visual storage | MP4 / H.264 via `torchcodec` |
| State/action storage | Parquet (float32, 7-DOF) |
| Resolution | 640 × 360 RGB + depth |

**Key constraint:** your cameras only run at 30 FPS. You cannot get real 50 Hz visual data
from 30 FPS cameras without synthesis. For true 50 Hz you would need cameras running at ≥50 FPS
(e.g. 60 FPS) and collect at 50 Hz.

---

## Upscaling 20 Hz → 50 Hz

The ratio is 5:2 — for every 2 original frames you need 5 output frames (3 are synthesized).

### Robot State / Action (Parquet)

Interpolation is clean and physically valid since joints move continuously.

| Method | Quality | Notes |
|---|---|---|
| Linear interpolation | Good | Fast, sufficient for smooth motions |
| Cubic spline | Better | Preserves velocity continuity |
| PCHIP | Best | Monotone-preserving, no overshoots |

```python
import numpy as np
from scipy.interpolate import PchipInterpolator

t_20hz = np.arange(len(states)) / 20.0          # original timestamps
t_50hz = np.arange(0, t_20hz[-1], 1/50.0)       # target timestamps
interp = PchipInterpolator(t_20hz, states)       # states: (N, 7)
states_50hz = interp(t_50hz)                     # (M, 7)
```

### Visual Data

This is the hard part. You have three realistic options:

#### Option A — AI Frame Interpolation (RIFE / FILM)
- Synthesize intermediate frames using optical flow-based deep interpolation
- Works on extracted video frames or saved images
- **RIFE** (Real-time Intermediate Flow Estimation) is fast and GPU-accelerated
- Quality is high for smooth, predictable robot motions
- Introduces hallucinated frames — not ground truth

```bash
# Extract frames from video first, then run RIFE interpolation
# pip install rife-ncnn-vulkan  (or use the PyTorch version)
```

#### Option B — Simple Frame Duplication / Nearest-neighbor
- Each 20 Hz frame is repeated to fill 50 Hz slots
- 20 Hz → 50 Hz: frame[i] repeats for slots at t+0 ms, t+20 ms; frame[i+1] for t+40 ms, etc.
- No hallucination, but jerky/stepped visual appearance
- Simplest to implement

#### Option C — Collect at Native Camera Rate (Recommended for new data)

Capture all camera frames at 30 FPS as **images**, record robot state at 30 Hz or higher,
then post-process to 50 Hz using interpolation. Even better: switch to 60 FPS cameras and
collect at 50 Hz — all data is real, no synthesis needed.

---

## Video vs. Images for Resampling

| Factor | Video (current) | Images |
|---|---|---|
| Storage | Compact (H.264 compression) | Large (3-5× more disk) |
| Frame access | Requires decode (torchcodec handles this) | Direct read |
| Interpolation quality | Compression artifacts degrade RIFE/FILM | Cleaner pixel values |
| Flexibility | Re-encoding needed after interpolation | Easy to resample then encode |
| Depth data | Lossy compression degrades depth | Lossless PNG preserves depth |
| Workflow complexity | Extra decode step | Simple |

### Recommendation for Resampling

**Save as images if you plan to resample.** Reasons:

1. H.264 is lossy — compression artifacts accumulate when you re-encode after interpolation
2. Depth data is especially sensitive to compression (you already convert uint16→uint8, losing precision)
3. Image sequences allow frame-accurate access without seeking in video
4. Easier to mix interpolated and real frames during post-processing

If storage is a concern, use **lossless PNG** for images, then encode to video as the final step
after all resampling is done.

---

## Practical Path to 50 Hz

### Short-term: post-process existing 20 Hz video data

1. Decode video frames using torchcodec (already in your stack)
2. Interpolate robot state with PCHIP to 50 Hz
3. Use RIFE to generate synthetic visual frames between each 20 Hz pair
4. Re-encode as a new LeRobot dataset at 50 Hz

### Long-term: collect at 50 Hz from the start (cleanest)

1. Switch cameras to 60 FPS mode (both Azure Kinect and RealSense support this)
2. Set `self.target_fps = 50` in [data_collect.py](src/ur5_lerobot_data_collection/ur5_lerobot_data_collection/data_collect.py) (line 60)
3. Robot state at 50 Hz — real data, no synthesis
4. Visual data sub-sampled from 60 FPS to 50 FPS — also real data

This gives you:
- Real (not hallucinated) visual observations
- True 50 Hz robot state transitions
- No resampling artifacts

---

## Code Change Required in `data_collect.py`

For 50 Hz collection, two lines need to change:

```python
# Line 60 — change target fps
self.target_fps = 50   # was 20

# LeRobotDataset.create() call — update fps parameter
dataset = LeRobotDataset.create(
    ...
    fps=50,            # was 20
    ...
)
```

Camera threads already run asynchronously at native rate (30–60 FPS); the recording
timer just picks up the latest buffered frame. At 50 Hz, the timer fires every 20 ms.
With 60 FPS cameras (16.7 ms per frame), a fresh frame is always available.

---

## Summary

| Goal | Best approach |
|---|---|
| Quick upscale of existing data | PCHIP for state, RIFE for visuals |
| Clean 50 Hz going forward | 60 FPS cameras + `target_fps=50` |
| Best resampling quality | Save as images (PNG), not video |
| Depth data fidelity | Always save depth as lossless images |
