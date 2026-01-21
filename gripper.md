###test gripper
```
git clone -b humble https://github.com/PickNikRobotics/ros2_robotiq_gripper.git
git clone -b ros2 https://github.com/tylerjw/serial.git

colcon build 

sudo chmod 777 /dev/ttyUSB0
ros2 launch robotiq_description robotiq_control.launch.py

ros2 action send_goal /robotiq_gripper_controller/gripper_cmd   control_msgs/action/GripperCommand   "{command: {position: 0.0, max_effort: 50.0}}"

```


