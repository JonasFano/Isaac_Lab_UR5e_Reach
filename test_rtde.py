import rtde_control
import rtde_receive
import numpy as np
import random
from scipy.spatial.transform import Rotation as R

# Replace with the IP address of your robot
ROBOT_IP = "192.168.1.100"

print("start")
# Initialize RTDE Control and Receive Interfaces
rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
print("Connected to Control Interface")
rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
print("Conntected to Receive Interface")

def move_robot_to_home():
    """Move robot to home pose specified with joint positions"""
    home_pose = np.array([1.3, -2.0, 2.0, -1.5, -1.5, 3.14])
    # home_pose = np.array([ 1.30899694, -1.83259571,  1.65806279,  4.79965544,  4.71238898, 0. ])
    speed = 0.7
    acceleration = 0.5
    rtde_c.moveJ(home_pose, speed=speed, acceleration=acceleration)


# Random Pose Sampling Function
def sample_random_pose():
    """Randomly sample a target pose within specified bounds."""
    # pos_x = random.uniform(-0.1, 0.1)
    # # pos_y = random.uniform(0.35, 0.55)
    # pos_y = random.uniform(-0.4, -0.2)
    # pos_z = random.uniform(0.25, 0.35)
    # roll = 0.0
    # pitch = np.pi  # End-effector pointing down
    # yaw = random.uniform(-np.pi/2, np.pi/2) # 90 degrees

    pos_x = random.uniform(-0.2, 0.2)
    pos_y = random.uniform(-0.4, -0.3)
    # pos_y = random.uniform(0.3, 0.4)
    pos_z = random.uniform(0.2, 0.5)
    roll = 0.0
    pitch = np.pi
    yaw = np.pi

    # Convert Roll-Pitch-Yaw to Quaternion [w, x, y, z] format
    r = R.from_euler("xyz", [roll, pitch, yaw], degrees=False)
    quat_wxyz = r.as_quat(scalar_first=True)  # Default format is [x, y, z, w]
    # Normalize
    quat_wxyz /= np.linalg.norm(quat_wxyz)

    r = R.from_euler('xyz', [roll, pitch, yaw], degrees=False)  # RPY to rotation vector
    rx, ry, rz = r.as_rotvec()

    print(quat_wxyz)

    return [pos_x, pos_y, pos_z, rx, ry, rz]



# Returns the actual TCP received from the robot in [X, Y, Z] + Quaternion in [w, x, y, z]
def get_actual_TCP_Pose():
    tcp_pose = rtde_r.getActualTCPPose()  # Returns [X, Y, Z, RX, RY, RZ]

    print("Current TCP Pose (Axis-Angle):", tcp_pose)

    # Extract position (X, Y, Z) and orientation (RX, RY, RZ)
    pos = np.array(tcp_pose[:3])  # [X, Y, Z]
    axis_angle = np.array(tcp_pose[3:])  # [RX, RY, RZ]

    # Convert Axis-Angle to Quaternion
    rot = R.from_rotvec(axis_angle)  # Convert to rotation object
    quat_wxyz = rot.as_quat(scalar_first=True)

    # Normalize the quaternion
    quat_wxyz /= np.linalg.norm(quat_wxyz)

    # Convert back from quaternion to axis-angle
    reconverted_rot = R.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    reconverted_axis_angle = reconverted_rot.as_rotvec()

    # Print the transformed data
    print("Current TCP Pose (Quaternion):", np.concatenate((pos, quat_wxyz), axis=0))
    print("Reconverted Axis-Angle (rx, ry, rz):", reconverted_axis_angle)

    return np.concatenate((pos, quat_wxyz), axis=0)



def main():
    move_robot_to_home()

    print("Home Pose: ", get_actual_TCP_Pose())

    while True:
        target_pose = sample_random_pose()  # Generate a random target pose

        print("Target Pose: ", target_pose)

        speed = 0.7
        acceleration = 0.5
        if rtde_c.moveL(target_pose, speed=speed, acceleration=acceleration):
            print("End Pose: ", get_actual_TCP_Pose())
            
            move_robot_to_home()
            print("Home Pose: ", get_actual_TCP_Pose())


# Run the main function
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Stopping robot execution.")
    finally:
        rtde_c.stopScript()
