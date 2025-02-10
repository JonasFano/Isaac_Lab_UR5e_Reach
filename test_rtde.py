import rtde_control
import rtde_receive
import numpy as np
import random

# Replace with the IP address of your robot
ROBOT_IP = "192.168.1.100"

# Initialize RTDE Control and Receive Interfaces
rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)

def move_robot_to_home():
    """Move robot to home pose specified with joint positions"""
    # home_pose = np.array((1.3, -2.0, 2.0, -1.5, -1.5, 3.14, 0.0, 0.0))
    home_pose = np.array([ 1.30899694, -1.83259571,  1.65806279,  4.79965544,  4.71238898, 0. ])
    speed = 0.1
    acceleration = 0.5
    rtde_c.moveJ(home_pose, speed=speed, acceleration=acceleration)


# Random Pose Sampling Function
def sample_random_pose():
    """Randomly sample a target pose within specified bounds."""
    pos_x = random.uniform(-0.1, 0.1)
    # pos_y = random.uniform(0.35, 0.55)
    pos_y = random.uniform(-0.4, -0.2)
    pos_z = random.uniform(0.25, 0.35)
    roll = 0.0
    pitch = np.pi  # End-effector pointing down
    yaw = random.uniform(-np.pi/2, np.pi/2) # 90 degrees
    return [pos_x, pos_y, pos_z, roll, pitch, yaw]


# Function to get the robot's current state
def get_robot_state():
    tcp_pose = rtde_r.getActualTCPPose()
    return tcp_pose


def main():
    move_robot_to_home()

    print("Home Pose: ", get_robot_state())

    target_pose = sample_random_pose()  # Generate a random target pose

    print("Target Pose: ", target_pose)

    speed = 0.1
    acceleration = 0.5
    rtde_c.moveL(target_pose, speed=speed, acceleration=acceleration)

    print("End Pose: ", get_robot_state())


# Run the main function
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Stopping robot execution.")
    finally:
        rtde_c.stopScript()
