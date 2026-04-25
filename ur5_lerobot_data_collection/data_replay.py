"""
Replay recorded joint trajectories from a LeRobot dataset onto a real UR5 + Robotiq 2F-85.

Reads joint data from parquet files (observation.state or action columns) and sends
each frame to the robot via ROS2, faithfully reproducing the recorded motion.

Joint ordering in dataset (indices 0–6):
    [shoulder_lift, elbow, wrist_1, wrist_2, wrist_3, shoulder_pan, gripper]
    indices 0-5 match scaled_joint_names order; index 6 is gripper position (meters).

Example:
    python data_replay.py --dataset-path /path/to/All_Datasets/dataset_small_to_orange
    python data_replay.py --dataset-path /path/to/dataset --episode 2 --use-state
    python data_replay.py --dataset-path /path/to/dataset --send-mode chunk
    python data_replay.py --dataset-path /path/to/dataset --controller passthrough
"""

import os
import time
import threading
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import tyro
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from control_msgs.action import FollowJointTrajectory, GripperCommand
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration


CONTROLLER_ACTIONS = {
    "scaled": "/scaled_joint_trajectory_controller/follow_joint_trajectory",
    "passthrough": "/passthrough_trajectory_controller/follow_joint_trajectory",
}


@dataclass
class ArgsConfig:
    dataset_path: str = "."
    """Path to the LeRobot dataset directory."""
    episode: int = 0
    """Episode index to replay."""
    use_state: bool = False
    """If True, replay observation.state (actual recorded positions) instead of action."""
    send_mode: Literal["single", "chunk", "chunked16"] = "single"
    """single: send frame-by-frame (paced by trajectory dt); chunk: one trajectory for all frames;
    chunked16: simulate inference pattern — process in groups of 16, first 14 wait=False, last 2 wait=True."""
    dt: float = 0.05
    """Time between frames in seconds (default 0.05 = 20 Hz, matching collection rate)."""
    move_to_start: bool = True
    """Move robot to the first frame's position before replaying."""
    move_to_start_duration: float = 3.0
    """Duration in seconds to reach the starting position."""
    gripper_max_effort: float = 50.0
    """Maximum gripper effort in Newtons."""
    start_frame: int = 0
    """Skip this many leading frames."""
    end_frame: int = -1
    """Stop at this frame index (-1 = replay all frames)."""
    controller: Literal["scaled", "passthrough"] = "scaled"
    """'scaled' = scaled_joint_trajectory_controller, 'passthrough' = passthrough_trajectory_controller."""


class DataReplayNode(Node):
    """ROS2 node for replaying joint trajectories on a UR5 + Robotiq 2F-85."""

    def __init__(self, controller: str = "scaled"):
        super().__init__("data_replay")

        # Action client: UR5 arm trajectory
        action_topic = CONTROLLER_ACTIONS[controller]
        self._traj_action = ActionClient(self, FollowJointTrajectory, action_topic)

        # Action client: Robotiq gripper
        self._gripper_action = ActionClient(
            self,
            GripperCommand,
            "/gripper/robotiq_gripper_controller/gripper_cmd",
        )

        # Publisher: forward_position_controller (available for fire-and-forget use)
        self._fwd_pos_pub = self.create_publisher(
            Float64MultiArray,
            "/forward_position_controller/commands",
            1,
        )

        # Joint order matching the dataset and eval_policy_hardware.py
        self.scaled_joint_names = [
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
            "shoulder_pan_joint",
        ]

    def send_single_action_scaled_joint(self, arm_positions, dt: float, wait: bool = False,
                                         timeout_sec: float = 0.3, velocities=None):
        """Send a single joint position as a 1-point trajectory.

        With wait=True the call blocks until the controller finishes executing the
        point, so time_from_start=dt naturally paces replay at the correct rate.
        Pass velocities (6-element array) to avoid stop-and-go jitter between frames.
        """

        if not self._traj_action.wait_for_server(timeout_sec=1.0):
            self.get_logger().warning(f"UR action server not ready: {self.ur_action_name}")
            return False

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = self.scaled_joint_names
        point = JointTrajectoryPoint()
        point.positions = np.atleast_1d(arm_positions).tolist()
        if velocities is not None:
            point.velocities = np.atleast_1d(velocities).tolist()
        point.time_from_start = Duration(sec=int(dt), nanosec=int((dt % 1) * 1e9))
        goal.trajectory.points.append(point)

        send_future = self._traj_action.send_goal_async(goal)

        if not wait:
            return True

        rclpy.spin_until_future_complete(self, send_future, timeout_sec=dt+timeout_sec)
        if not send_future.done():
            self.get_logger().error("send_goal_async timed out")
            return False

        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("Trajectory goal rejected")
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=dt+timeout_sec)
        if not result_future.done():
            self.get_logger().error("get_result_async timed out")
            return False

        result = result_future.result().result
        if result.error_code == FollowJointTrajectory.Result.SUCCESSFUL:
            return True
        else:
            self.get_logger().warn(
                f"Trajectory finished with error_code={result.error_code}, "
                f"error_string={result.error_string}"
            )
            return False

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

        send_future = self._traj_action.send_goal_async(goal)
        timeout = len(arm_actions) * dt + 5.0

        rclpy.spin_until_future_complete(self, send_future, timeout_sec=timeout)
        if not send_future.done():
            self.get_logger().error("send_chunk_action: send_goal_async timed out")
            return

        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("Trajectory chunk goal rejected")
            return

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=timeout)
        if not result_future.done():
            self.get_logger().error("send_chunk_action: get_result_async timed out")

    def send_gripper_command(self, position: float, max_effort: float = 50.0):
        """Send a gripper position command (fire-and-forget)."""
        goal = GripperCommand.Goal()
        goal.command.position = float(position)
        goal.command.max_effort = float(max_effort)
        self._gripper_action.send_goal_async(goal)

    def move_to_start(self, arm_positions, duration: float):
        """Move to the starting position using a single long-duration trajectory point."""
        self.get_logger().info(f"Moving to start position over {duration:.1f}s ...")
        self.send_single_action_scaled_joint(arm_positions, dt=duration, wait=True)
        self.get_logger().info("Reached start position.")


def load_episode(dataset_path: str, episode_index: int) -> pd.DataFrame:
    """Load a single episode's parquet file and return rows sorted by frame_index."""
    chunk_id = episode_index // 1000
    parquet_path = os.path.join(
        dataset_path,
        f"data/chunk-{chunk_id:03d}/episode_{episode_index:06d}.parquet",
    )
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    df = df.sort_values("frame_index").reset_index(drop=True)
    return df


def main(args: ArgsConfig):
    # --- Load episode ---
    print(f"Loading episode {args.episode} from: {args.dataset_path}")
    df = load_episode(args.dataset_path, args.episode)
    print(f"Episode length: {len(df)} frames ({len(df) * args.dt:.1f}s at {1/args.dt:.0f} Hz)")

    col = "observation.state" if args.use_state else "action"
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found. Available: {list(df.columns)}")
    print(f"Replaying column: '{col}'")

    # Slice frames
    end = args.end_frame if args.end_frame >= 0 else len(df)
    df = df.iloc[args.start_frame:end].reset_index(drop=True)
    print(f"Replaying frames {args.start_frame}–{end - 1} ({len(df)} frames)")

    # Extract arm (0:6) and gripper (6) from each frame
    joint_data = np.stack(df[col].values)   # (N, 7)
    arm_data = joint_data[:, :6]             # (N, 6)
    gripper_data = joint_data[:, 6]          # (N,)

    # --- Init ROS2 ---
    rclpy.init()
    ur5 = DataReplayNode(controller=args.controller)

    executor = MultiThreadedExecutor()
    executor.add_node(ur5)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        # Wait for action servers
        print("Waiting for action servers ...")
        ur5._traj_action.wait_for_server()
        ur5._gripper_action.wait_for_server()
        print("Action servers ready.")

        # Move to first frame's position
        if args.move_to_start:
            ur5.move_to_start(arm_data[0], duration=args.move_to_start_duration)
            ur5.send_gripper_command(gripper_data[0], max_effort=args.gripper_max_effort)
            time.sleep(0.5)  # brief settle before replay

        print(f"Starting replay in '{args.send_mode}' mode ...")
        t_start = time.perf_counter()

        # Only send a new gripper goal when position changes enough to avoid
        # flooding the Robotiq action server (which queues goals and lags at >~20 Hz).
        GRIPPER_THRESHOLD = 0.005  # metres (~5 mm dead-band)

        if args.send_mode == "chunk":
            # Send gripper commands in background thread (paced by dt) while arm chunk runs
            def _send_gripper_seq():
                last_sent = gripper_data[0] - 2 * GRIPPER_THRESHOLD  # force first send
                for grip_pos in gripper_data:
                    if abs(grip_pos - last_sent) > GRIPPER_THRESHOLD:
                        ur5.send_gripper_command(grip_pos, max_effort=args.gripper_max_effort)
                        last_sent = grip_pos
                    time.sleep(args.dt)

            gripper_thread = threading.Thread(target=_send_gripper_seq, daemon=True)
            gripper_thread.start()
            ur5.send_chunk_action(arm_data, args.dt)
            gripper_thread.join()

        elif args.send_mode == "chunked16":
            # Simulate inference pattern: process frames in groups of 16.
            # Within each chunk, frames 0-13 (1-14) fire-and-forget; frames 14-15 (15-16) block.
            # The blocking tail paces chunk-to-chunk, mimicking inference compute delay.
            CHUNK_SIZE = 16
            WAIT_FROM = 14  # frame index within chunk where wait flips to True
            last_grip_sent = gripper_data[0] - 2 * GRIPPER_THRESHOLD
            for i, (arm_pos, grip_pos) in enumerate(zip(arm_data, gripper_data)):
                pos_in_chunk = i % CHUNK_SIZE
                is_last_in_chunk = (pos_in_chunk >= WAIT_FROM) or (i == len(arm_data) - 1)
                if abs(grip_pos - last_grip_sent) > GRIPPER_THRESHOLD:
                    ur5.send_gripper_command(grip_pos, max_effort=args.gripper_max_effort)
                    last_grip_sent = grip_pos
                if not is_last_in_chunk:
                    ur5.send_single_action_scaled_joint(arm_pos, dt=args.dt, wait=False)
                else:
                    ur5.send_single_action_scaled_joint(arm_pos, dt=args.dt+0.3, wait=True)

        else:  # single
            # Finite-difference velocities: controller passes through each point without stopping.
            # Zero velocity at first and last frame so robot starts/ends at rest.
            vels = np.zeros_like(arm_data)
            vels[1:-1] = (arm_data[2:] - arm_data[:-2]) / (2 * args.dt)

            last_grip_sent = gripper_data[0] - 2 * GRIPPER_THRESHOLD  # force first send
            for i, (arm_pos, grip_pos) in enumerate(zip(arm_data, gripper_data)):
                t0 = time.perf_counter()
                if abs(grip_pos - last_grip_sent) > GRIPPER_THRESHOLD:
                    ur5.send_gripper_command(grip_pos, max_effort=args.gripper_max_effort)
                    last_grip_sent = grip_pos
                ur5.send_single_action_scaled_joint(arm_pos, dt=args.dt, wait=False, velocities=vels[i])
                time.sleep(max(0.0, args.dt - (time.perf_counter() - t0)))


        #         if (i + 1) % 20 == 0:
        #             elapsed = time.perf_counter() - t_start
        #             print(f"  Frame {i + 1}/{len(arm_data)} — elapsed {elapsed:.1f}s")

        # total_elapsed = time.perf_counter() - t_start
        # print(f"\nReplay complete. {len(arm_data)} frames in {total_elapsed:.2f}s "
        #       f"(expected {len(arm_data) * args.dt:.2f}s)")

    finally:
        executor.shutdown()
        ur5.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    args = tyro.cli(ArgsConfig)
    main(args)
