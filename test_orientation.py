import rtde_control
import rtde_receive
import numpy as np
from scipy.spatial.transform import Rotation as R

# Replace with the IP address of your robot
ROBOT_IP = "192.168.1.100"

ROT_180_Z = R.from_euler('z', 180, degrees=True)

print("start")
# Initialize RTDE Control and Receive Interfaces
rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
print("Connected to Control Interface")
rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
print("Connected to Receive Interface")


tcp_pose_aa = rtde_r.getActualTCPPose()
print("TCP Axis Angle: ", tcp_pose_aa)


# Extract position (X, Y, Z) and orientation (RX, RY, RZ)
pos = np.array(tcp_pose_aa[:3])  # [X, Y, Z]
axis_angle = np.array(tcp_pose_aa[3:])  # [RX, RY, RZ]

# Convert Axis-Angle to Quaternion
rot = R.from_rotvec(axis_angle)  # Convert to rotation object
quat_wxyz = rot.as_quat(scalar_first=True)

# Reorder quaternion to [w, x, y, z] and normalize
quat_wxyz /= np.linalg.norm(quat_wxyz)

tcp_pose_quat = np.concatenate((pos, quat_wxyz), axis=0)

print("TCP Quaternion: ", tcp_pose_quat)


rotated_quat = ROT_180_Z * R.from_quat(quat_wxyz, scalar_first=True)
rotated_quat_wxyz = rotated_quat.as_quat(scalar_first=True)

print("Rotated TCP Quaternion: ", rotated_quat_wxyz)


joint_position = rtde_r.getActualQ()
print("Joint Position: ", joint_position)
