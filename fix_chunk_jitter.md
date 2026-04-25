# Plan: Fix chunk-mode jitter in data_replay.py

## Context

`send_chunk_action` sends all frames as one `FollowJointTrajectory` goal, which should be smooth — but it only sets `positions`, omitting `velocities`. The controller must infer velocities via its own spline fit; without boundary constraints it overshoots/undershoots at each waypoint, causing visible jitter especially on direction reversals.

Fix: compute per-frame velocities via central finite differences and pass them explicitly. The controller then uses a cubic spline with exact velocity constraints → smooth motion.

## Critical file

`src/ur5_lerobot_data_collection/ur5_lerobot_data_collection/data_replay.py`

---

## Change: `send_chunk_action` — add velocity hints

**Location:** lines 140–149

**Before:**
```python
def send_chunk_action(self, arm_actions, dt: float):
    """Send all frames as one trajectory goal; controller spline-interpolates."""
    goal = FollowJointTrajectory.Goal()
    goal.trajectory.joint_names = self.scaled_joint_names
    for i, positions in enumerate(arm_actions):
        point = JointTrajectoryPoint()
        point.positions = np.atleast_1d(positions).tolist()
        t = (i + 1) * dt
        point.time_from_start = Duration(sec=int(t), nanosec=int((t % 1) * 1e9))
        goal.trajectory.points.append(point)
```

**After:**
```python
def send_chunk_action(self, arm_actions, dt: float):
    """Send all frames as one trajectory goal; controller spline-interpolates."""
    arm_actions = np.asarray(arm_actions)
    # Central-difference velocities; clamp endpoints to zero
    velocities = np.zeros_like(arm_actions)
    velocities[1:-1] = (arm_actions[2:] - arm_actions[:-2]) / (2.0 * dt)

    goal = FollowJointTrajectory.Goal()
    goal.trajectory.joint_names = self.scaled_joint_names
    for i, (positions, vels) in enumerate(zip(arm_actions, velocities)):
        point = JointTrajectoryPoint()
        point.positions = positions.tolist()
        point.velocities = vels.tolist()
        t = (i + 1) * dt
        point.time_from_start = Duration(sec=int(t), nanosec=int((t % 1) * 1e9))
        goal.trajectory.points.append(point)
```

**Why central differences:** `v[i] = (pos[i+1] - pos[i-1]) / (2*dt)` — matches velocity the robot was actually moving at when that frame was recorded. Zero at endpoints ensures clean start/stop.

---

## Also save plan file

Copy this plan to `src/ur5_lerobot_data_collection/fix_chunk_jitter.md` for repo reference.

## Verification

```bash
python data_replay.py --dataset-path /path/to/dataset --send-mode chunk
```

Compare motion smoothness vs before. If direction-reversal joints still overshoot, optionally also add `accelerations` (second finite difference), but velocities alone should be sufficient.
