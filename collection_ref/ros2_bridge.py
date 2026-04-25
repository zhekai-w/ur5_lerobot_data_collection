# -*- coding: utf-8 -*-
from typing import List

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory, GripperCommand

from rclpy.executors import MultiThreadedExecutor

from std_msgs.msg import Float64MultiArray

from controller_manager_msgs.srv import SwitchController

import time
class Ros2Bridge(Node):
    """

    """

    def __init__(self):
        super().__init__("gr00t_ros2_bridge")

        # ---------- Parameters ----------

        # UR & Gripper
        self.declare_parameter("ur.follow_joint_traj_action",
                               "/scaled_joint_trajectory_controller/follow_joint_trajectory")
        # self.declare_parameter("ur.follow_joint_traj_action",
        #                        "/joint_trajectory_controller/follow_joint_trajectory")
        # self.declare_parameter("ur.follow_joint_traj_action",
        #                        "/passthrough_trajectory_controller/follow_joint_trajectory")
        self.declare_parameter("ur.forward_position_controller_topic",
                               "/forward_position_controller/commands")
        
        self.declare_parameter("ur.joint_names", [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ])
        self.declare_parameter("gripper.action_name", "/gripper/robotiq_gripper_controller/gripper_cmd")
        self.declare_parameter("gripper.joint_name", "robotiq_85_left_knuckle_joint")
        self.declare_parameter("gripper.opening_range_m", 0.085)

        # JointStates
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("gripper_joint_states_topic", "/gripper/joint_states")

        # ---------- Read some params locally ----------
        self.ur_action_name = self.get_parameter("ur.follow_joint_traj_action").get_parameter_value().string_value
        self.ur_joint_names = list(self.get_parameter("ur.joint_names").get_parameter_value().string_array_value)
        self.gripper_action_name = self.get_parameter("gripper.action_name").get_parameter_value().string_value
        self.gripper_joint_name = self.get_parameter("gripper.joint_name").get_parameter_value().string_value
        self.gripper_opening_range_m = float(self.get_parameter("gripper.opening_range_m").get_parameter_value().double_value)
        self.joint_states_topic = self.get_parameter("joint_states_topic").get_parameter_value().string_value
        self.gripper_joint_states_topic = self.get_parameter("gripper_joint_states_topic").get_parameter_value().string_value

        self.ur_controller_topic_name = self.get_parameter("ur.forward_position_controller_topic").get_parameter_value().string_value
        self.ur_pos_pub = self.create_publisher(Float64MultiArray, self.ur_controller_topic_name, 10)
        # self.get_logger().info(f"Using forward position controller topic: {topic}")
        self.switch_controller_client = self.create_client(SwitchController, '/controller_manager/switch_controller')
        while not self.switch_controller_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info('Waiting for /controller_manager/switch_controller service...')


        # ---------- State buffers ----------
        self._last_joint_names: List[str] = []
        self._last_joint_pos: List[float] = []
        self._last_gripper_joint_names: List[str] = []
        self._last_gripper_joint_pos: List[float] = []

        # ---------- ROS I/O ----------
        self.create_subscription(JointState, self.joint_states_topic, self._joint_state_cb, 10)
        self.create_subscription(JointState, self.gripper_joint_states_topic, self._gripper_joint_state_cb, 10)
        self.ur_action_client = ActionClient(self, FollowJointTrajectory, self.ur_action_name)
        self.gripper_action_client = ActionClient(self, GripperCommand, self.gripper_action_name)

        self.get_logger().info("Ros2Bridge node ready.")

    def switch_controller(self, start=[], stop=[], strictness=2, timeout=5.0, start_asap=False):
            req = SwitchController.Request()
            req.start_controllers = start
            req.stop_controllers = stop
            req.strictness = strictness       # 1 = BEST_EFFORT, 2 = STRICT
            req.start_asap = start_asap
            # req.timeout = float(timeout)

            self.get_logger().info(f"Switching controllers: start={start}, stop={stop}")
            future = self.switch_controller_client.call_async(req)
            rclpy.spin_until_future_complete(self, future)

            if future.result() and future.result().ok:
                self.get_logger().info("Controller switch successful")
            else:
                self.get_logger().error("Controller switch failed")

    def move_to_joint(self, joint_positions, duration=2.0, rate_hz=100):
        start_time = time.time()
        current = self.get_arm_state()
        joint_positions = np.array(joint_positions, dtype=np.float64)
        steps = int(duration * rate_hz)
        rate = 1.0 / rate_hz

        for i in range(steps + 1):
            alpha = i / steps
            cmd = (1 - alpha) * current + alpha * joint_positions
            msg = Float64MultiArray(data=cmd.tolist())
            self.ur_pos_pub.publish(msg)
            time.sleep(rate)

        self.get_logger().info(f"Reached target: {joint_positions}")


    # ---------------------------
    # Parameter getters
    # ---------------------------
    def get_param_str(self, name: str, default: str) -> str:
        if self.has_parameter(name):
            return self.get_parameter(name).get_parameter_value().string_value or default
        return default

    def get_param_int(self, name: str, default: int) -> int:
        if self.has_parameter(name):
            return int(self.get_parameter(name).get_parameter_value().integer_value or default)
        return default

    # ---------------------------
    # Joint state handling
    # ---------------------------
    def _joint_state_cb(self, msg: JointState):
        self._last_joint_names = list(msg.name)
        self._last_joint_pos = list(msg.position) if msg.position else []

    def _gripper_joint_state_cb(self, msg: JointState):
        self._last_gripper_joint_names = list(msg.name)
        self._last_gripper_joint_pos = list(msg.position) if msg.position else []

    def get_arm_state(self) -> np.ndarray:
        """依宣告的 UR joint 名稱順序回傳 positions(np.float32[6])，若缺資料則 zeros。"""
        if not self._last_joint_names or not self._last_joint_pos:
            return np.zeros(6, dtype=np.float32)
        name_to_pos = {n: p for n, p in zip(self._last_joint_names, self._last_joint_pos)}
        vals = [float(name_to_pos.get(n, 0.0)) for n in self.ur_joint_names]
        return np.array(vals, dtype=np.float32)

    def get_gripper_state(self) -> np.ndarray:
        """回傳夾爪關節位置 (單值 np.array)；若沒有則 0.0。"""
        if not self._last_gripper_joint_names or not self._last_gripper_joint_pos:
            # return np.array([0.0], dtype=np.float32)
            return np.zeros(6, dtype=np.float32)
        last_gripper_joint_pos = (self._last_gripper_joint_pos)*6
        # name_to_pos = {n: p for n, p in zip(self._last_gripper_joint_names, last_gripper_joint_pos)}
        # return np.array([float(name_to_pos.get(self.gripper_joint_name, 0.0))], dtype=np.float32)
        return np.array(last_gripper_joint_pos, dtype=np.float32)
    # ---------------------------
    # Arm & Gripper command
    # ---------------------------
    # def send_arm_trajectory(self, target_positions: np.ndarray, duration: float = 1.0, wait: bool = True,):
    #     if target_positions.shape[0] != 6:
    #         self.get_logger().error(f"Expected 6-DoF, got {target_positions.shape}")
    #         return
    #     if not self.ur_action_client.wait_for_server(timeout_sec=0.0):
    #         self.get_logger().warning(f"UR action server not ready: {self.ur_action_name}")
    #         return

    #     traj = JointTrajectory()
    #     traj.joint_names = self.ur_joint_names

    #     pt = JointTrajectoryPoint()
    #     pt.positions = target_positions.tolist()
    #     pt.time_from_start.sec = int(duration)
    #     pt.time_from_start.nanosec = int((duration - int(duration)) * 1e9)
    #     traj.points.append(pt)

    #     goal = FollowJointTrajectory.Goal()
    #     goal.trajectory = traj
    #     self.ur_action_client.send_goal_async(goal)
    def send_arm_trajectory(self, target_positions: np.ndarray, duration: float = 0.3, wait: bool = False, timeout_sec: float=0.3):
        if target_positions.shape[0] != 6:
                    self.get_logger().error(f"Expected 6-DoF, got {target_positions.shape}")
                    return False

        if not self.ur_action_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warning(f"UR action server not ready: {self.ur_action_name}")
            return False

        traj = JointTrajectory()
        traj.joint_names = self.ur_joint_names

        pt = JointTrajectoryPoint()
        pt.positions = target_positions.tolist()
        pt.time_from_start.sec = int(duration)
        pt.time_from_start.nanosec = int((duration - int(duration)) * 1e9)
        traj.points.append(pt)

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        send_goal_future = self.ur_action_client.send_goal_async(
            goal,
        )

        if not wait:
            # 非同步模式：只丟出去不等結果
            return True

        # 等待 goal 是否被接受
        rclpy.spin_until_future_complete(self, send_goal_future, timeout_sec=timeout_sec)
        if not send_goal_future.done():
            self.get_logger().error("send_goal_async timed out")
            return False

        goal_handle = send_goal_future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Trajectory goal rejected by server")
            return False

        # 等待控制器完成軌跡執行
        get_result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, get_result_future, timeout_sec=timeout_sec)
        if not get_result_future.done():
            self.get_logger().error("get_result_async timed out")
            return False

        result = get_result_future.result().result  # type: FollowJointTrajectory.Result
        if result.error_code == FollowJointTrajectory.Result.SUCCESSFUL:
            # self.get_logger().info("Trajectory finished successfully")
            return True
        else:
            self.get_logger().warn(
                f"Trajectory finished with error_code={result.error_code}, "
                f"error_string={result.error_string}"
            )
            return False

    def send_gripper_command(self, gripper_scalar: np.ndarray, max_effort: float = 50.0, wait: bool = False, timeout_sec: float=0.3):
        """
        gripper_scalar: [0,1]；0=關、1=開
        轉為 2F-85 的開度（公尺）給 position。
        """
        if not self.gripper_action_client.wait_for_server(timeout_sec=0.0):
            self.get_logger().warning(f"Gripper action server not ready: {self.gripper_action_name}")
            return

        # val = float(np.clip(float(gripper_scalar), 0.0, 1.0))
        # position_m = val * self.gripper_opening_range_m
        position_m = float(gripper_scalar)*1.0

        goal = GripperCommand.Goal()
        goal.command.position = position_m
        goal.command.max_effort = float(max_effort)
        send_goal_future = self.gripper_action_client.send_goal_async(goal)
        if not wait:
            return True
        # 等待 goal 是否被接受
        rclpy.spin_until_future_complete(self, send_goal_future, timeout_sec=timeout_sec)
        if not send_goal_future.done():
            self.get_logger().error("gripper send_goal_async timed out")
            return False

        goal_handle = send_goal_future.result()
        if not goal_handle.accepted:
            self.get_logger().error("gripper goal rejected by server")
            return False
            

    # ---------------------------
    # Clean shutdown
    # ---------------------------
    def shutdown(self, executor: MultiThreadedExecutor):
        try:
            executor.remove_node(self)
        except Exception:
            pass
        try:
            self.destroy_node()
        except Exception:
            pass

