#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import threading
from typing import Optional, Dict, List, Tuple

import argparse
import sys,os
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor

from .ros2_bridge import Ros2Bridge
from .sensor_bridge import MultiRealSense
import cv2
import torch

import queue

keyboard_queue = queue.Queue()


INSTRUCTION_MAP = {
    "1": "pick the blue cube and place to the blue spot",
    "2": "pick the blue cube and place to the green spot",
    "3": "pick the blue cube and place to the red spot",
    "4": "pick the blue cube and place to the yellow spot",
    "q": "pick the green cube and place to the blue spot",
    "w": "pick the green cube and place to the green spot",
    "e": "pick the green cube and place to the red spot",
    "r": "pick the green cube and place to the yellow spot",
    "a": "pick the red cube and place to the blue spot",
    "s": "pick the red cube and place to the green spot",
    "d": "pick the red cube and place to the red spot",
    "f": "pick the red cube and place to the yellow spot",
    "z": "pick the yellow cube and place to the blue spot",
    "x": "pick the yellow cube and place to the green spot",
    "c": "pick the yellow cube and place to the red spot",
    "v": "pick the yellow cube and place to the yellow spot",
}

def keyboard_listener():
    while True:
        try:
            key = input().strip().lower()
            keyboard_queue.put(key)
        except EOFError:
            break

def main():
    parser = argparse.ArgumentParser(description="Replay demonstrations in real environments.")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--use_policy", type=bool, default=True)
    parser.add_argument("--action_horizon", type=int, default=4)
    parser.add_argument("--actions_to_execute", type=int, default=50)
    parser.add_argument("--language_instruction", type=str, default="pick the blue cube and place to the green spot")
    # parser.add_argument("--language_instruction", type=str, default="do nothing")
    #parser.add_argument("--language_instruction", type=str, default="pick the red part")
    parser.add_argument("--task_duration", type=float, default=60.0, help="Maximum task duration in seconds")
    # parser.add_argument("--serials_list", type=list, default = ["908212070937", "123622270534"]) #d435i, d405 
    # parser.add_argument("--serials_list", type=list, default = ["908212070937", "844212070148"]) #d435i, d435i 
    parser.add_argument("--serials_list", type=list, default = ["117122250090", "043422251770"]) #d455 wrist, d455
    # parser.add_argument("--camera_width", type=int, default = 640)
    # parser.add_argument("--camera_height", type=int, default = 480)
    parser.add_argument("--camera_width", type=int, default = 640)
    parser.add_argument("--camera_height", type=int, default = 360)
    parser.add_argument("--camera_fps", type=int, default = 30)
    parser.add_argument("--enable_depth", type=bool, default = False)
    # parser.add_argument("--enable_depth", action="store_true", default=True, help="Enable Depth.")
    parser.add_argument("--go_initial", type=bool, default = True)



    # parse the arguments
    args_cli = parser.parse_args()


    if args_cli.use_policy:
        sys.path.append(os.path.expanduser("/workspace/Isaac-GR00T/gr00t/eval/"))
        from service import ExternalRobotInferenceClient
        gr00t_policy = ExternalRobotInferenceClient(host=args_cli.host, port=args_cli.port)
        
    rclpy.init()

    ros2_bridge = Ros2Bridge()

    # run ros callback in background
    executor = MultiThreadedExecutor()
    executor.add_node(ros2_bridge)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    # ros2_bridge.switch_controller(start=['forward_position_controller'],
    #                                 stop=['scaled_joint_trajectory_controller'])


    kb_thread = threading.Thread(target=keyboard_listener, daemon=True)
    kb_thread.start()

    ros2_bridge.get_logger().info("Keyboard control: press ENTER to continue, 'r' to reset robot")
    # Open RealSense
    cams = MultiRealSense(args_cli.serials_list, 
                          args_cli.camera_width, 
                          args_cli.camera_height, 
                          args_cli.camera_fps,
                          args_cli.enable_depth,
                          log_fn=ros2_bridge.get_logger().info,
                          warn_fn=ros2_bridge.get_logger().warning)
    cams.start()


    if args_cli.go_initial == True:
        ros2_bridge.get_logger().info("Resetting robot to initial pose...")
        ros2_bridge.send_arm_trajectory(np.deg2rad([0, -90, 90, -90, -90, 0]), duration=0.5, wait=True, timeout_sec=5.0)
        # ros2_bridge.move_to_joint(np.array([0.0/180*3.14,
        #                                         -90.0/180*3.14,
        #                                         90.0/180*3.14,
        #                                         -90.0/180*3.14,
        #                                         -90.0/180*3.14,
        #                                         0.0/180*3.14], 
        #                                         ), duration=3.0)
        # ros2_bridge.switch_controller(start=['scaled_joint_trajectory_controller'],
        #                                 stop=['forward_position_controller'])

    # Inference Loop
    #period = 1.0 / max(1, inference_hz)

    def get_images_pair() -> Tuple[np.ndarray, np.ndarray]:
        if cams is None:
            fake = np.random.randint(0, 255, (args_cli.camera_height, args_cli.camera_width, 3), dtype=np.uint8)
            return fake, fake
        frames = cams.get_frames_by_serial()
        imgs = []
        for s in args_cli.serials_list:
            frm = frames.get(s, None)
            if frm is None:
                frm = np.zeros((args_cli.camera_height, args_cli.camera_width, 3), dtype=np.uint8)
            else:
                if frm.shape[0] != args_cli.camera_height or frm.shape[1] != args_cli.camera_width:
                    frm = cv2.resize(frm, (args_cli.camera_width, args_cli.camera_height), interpolation=cv2.INTER_AREA)
            imgs.append(frm)
        if len(imgs) == 0:
            fake = np.random.randint(0, 255, (args_cli.camera_height, args_cli.camera_width, 3), dtype=np.uint8)
            return fake, fake
        if len(imgs) == 1:
            return imgs[0], imgs[0]
        return imgs[0], imgs[1]

    period = 0.01
    paused = False 
    ros2_bridge.get_logger().info("GR00T main loop started.")
    language_instruction = args_cli.language_instruction
    try:
        while rclpy.ok():
            rgb_frames_list = cams.get_rgb_frames_list()
            # wrist_rgb, table_rgb = cv2.resize(rgb_frames_list[0], (256, 256)), cv2.resize(rgb_frames_list[1], (256, 256))
            # wrist_rgb, table_rgb = cv2.resize(rgb_frames_list[0], (640, 360)), cv2.resize(rgb_frames_list[1], (640, 360))
            
            wrist_rgb, table_rgb = rgb_frames_list[0][:, :, ::-1], rgb_frames_list[1][:, :, ::-1]

            # wrist_rgb, table_rgb = get_images_pair()
            # table_rgb, wrist_rgb = get_images_pair()
            if args_cli.enable_depth:
                depth_frames_list = cams.get_depth_frames_list()
                wrist_depth, table_depth = depth_frames_list[0], depth_frames_list[1]
            if wrist_rgb is None or table_rgb is None:
                ros2_bridge.get_logger().info("No camera rgb")
                time.sleep(0.01)
                continue
            ###########test images ###################
            # else:
            #     rgb_images = np.hstack((rgb_frames_list[0], rgb_frames_list[1]))

            #     # Show images from both cameras
            #     cv2.namedWindow('RealSense', cv2.WINDOW_NORMAL)
            #     cv2.imshow('RealSense', rgb_images)
            #     cv2.waitKey(1)

                # if args_cli.enable_depth:
                #     depth_frames_list = cams.get_depth_frames_list()
                #     depth_images = np.hstack((wrist_depth, table_depth,))

                #     # Show images from both cameras
                #     cv2.namedWindow('RealSense_depth', cv2.WINDOW_NORMAL)
                #     cv2.imshow('RealSense_depth', depth_images)
                #     cv2.waitKey(1)
            ##############################
            # # 若你的模型期望 RGB，將 BGR 轉 RGB： img = img[:, :, ::-1]
            # arm_state = ros2_bridge.get_arm_state().astype(np.float32)  # (6,)
            # grip_state = ros2_bridge.get_gripper_state().astype(np.float32)  # (6,)
            arm_state = ros2_bridge.get_arm_state()  # (6,)
            grip_state = ros2_bridge.get_gripper_state()# (6,)
            # print(arm_state[None, :])
            # print(type(arm_state[None, :]))

            state = {
                "video.exterior_image": np.expand_dims(table_rgb, 0).astype(np.uint8),
                "video.wrist_image": np.expand_dims(wrist_rgb, 0).astype(np.uint8),
                # "state.robot_arm": ros2_bridge.get_arm_state(),     # (6,)
                # "state.gripper": ros2_bridge.get_gripper_state(),   # (6,)
                "state.robot_arm": arm_state[np.newaxis, :].astype(np.float64),
                "state.gripper": grip_state[np.newaxis, :].astype(np.float64),
                # "state.robot_arm": arm_state[None, :],
                # "state.gripper": grip_state[None, :],
                "annotation.human.action.task_description": [language_instruction],
            }

            # print(ros2_bridge.get_arm_state())
            # print(ros2_bridge.get_gripper_state())
            try:
                action = gr00t_policy.get_action(state)
                arm_actions = np.array([action["action.joint_action"]], dtype=np.float32)
                gripper_actions = np.array([action["action.gripper"]], dtype=np.float32)
            except Exception as e:
                ros2_bridge.get_logger().error(f"GR00T inference failed: {e}")
                time.sleep(period)
                continue

            # # 寫回 UR 與 Gripper
            a = 0
            num_actions = len(arm_actions[0])
            for i in range(num_actions):
                if i !=15 and i !=16:
                # if i !=num_actions-2 and i !=num_actions-1 and i !=num_actions:
                    ros2_bridge.send_arm_trajectory(arm_actions[0][i], duration=0.3)
                    # ros2_bridge.move_to_joint(arm_actions[0][i], duration=0.1, rate_hz=200)
                    ros2_bridge.send_gripper_command(gripper_actions[0][i][2], max_effort=100.0)
                else :
                    ros2_bridge.send_arm_trajectory(arm_actions[0][i], duration=0.2, wait=True, timeout_sec=1.0)
                    # ros2_bridge.move_to_joint(arm_actions[0][i], duration=0.2, rate_hz=200)
                    # time.sleep(0.5)
                    ros2_bridge.send_gripper_command(gripper_actions[0][i][2], max_effort=100.0)
                    # time.sleep(1.0)
                # print(arm_actions[0][i])
                # print(gripper_actions[0][i])
                # print(a)
            # ros2_bridge.send_gripper_command(0.5, max_effort=50.0)
            # 
            try:
                while not keyboard_queue.empty():
                    k = keyboard_queue.get()
                    # ENTER = continue (do nothing)


                    # r = reset to initial pose
                    if k == "t":
                        ros2_bridge.get_logger().info("Keyboard: Resetting robot to initial pose...")
                        ros2_bridge.send_arm_trajectory(np.deg2rad([0, -90, 90, -90, -90, 0]), duration=0.5, wait=True, timeout_sec=5.0)
                        paused = True
                        if paused:
                            ros2_bridge.get_logger().info("ENTER pressed → continuing inference")
                            input()
                            paused = False
                        # input()
                    if k in INSTRUCTION_MAP:
                        language_instruction = INSTRUCTION_MAP[k]
                        ros2_bridge.get_logger().info(f"Language instruction changed → '{language_instruction}'")
                        continue
                    else:
                        pass
            except Exception as e:
                ros2_bridge.get_logger().error(f"reset failed: {e}")  
            time.sleep(period)

    except KeyboardInterrupt:
        pass
    finally:
        cams.stop()
        ros2_bridge.shutdown(executor)
        rclpy.shutdown()


if __name__ == "__main__":
    main()

