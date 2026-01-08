# UR5 LeRobot Data Collection

This is a ROS2 node for collecting LeRobot data using UR5 robot arm.

## Prerequisites

Install the UR ROS2 driver:
```bash
sudo apt-get install ros-humble-ur
```

## Network Setup

Set your PC's IP address to `192.168.1.199`.

For detailed UR robot network setup and configuration, follow the official documentation:
https://docs.ros.org/en/ros2_packages/humble/api/ur_robot_driver/doc/installation/toc.html

## Usage

### Launch Controller for Real Robot

```bash
ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur5 robot_ip:=192.168.1.100
```

Replace `192.168.1.100` with your robot's IP address.

### Launch MoveIt Controller

```bash
ros2 launch ur_moveit_config ur_moveit.launch.py ur_type:=ur5
```

### Use Fake Hardware (for testing without physical robot)

```bash
ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur5e robot_ip:=yyy.yyy.yyy.yyy use_fake_hardware:=true launch_rviz:=true
```
