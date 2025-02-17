import numpy as np
import random
from scipy.spatial.transform import Rotation as R

pos_x = 0.05
pos_y = -0.3
pos_z = 0.3
roll = 0.0
pitch = np.pi
yaw = np.pi #np.pi/4

# r = R.from_euler('xyz', [roll, pitch, yaw], degrees=False)  # RPY to rotation vector
# rx, ry, rz = r.as_rotvec()


# Convert Roll-Pitch-Yaw to Quaternion [w, x, y, z] format
r = R.from_euler('xyz', [roll, pitch, yaw], degrees=False)
print(r.as_matrix())
print(r.as_euler("xyz"))
print("Rotvec: ", r.as_rotvec())
quat = r.as_quat()  # Default format is [x, y, z, w]
print(quat)

# Reorder quaternion to [w, x, y, z]
quat_wxyz = np.array([quat[3], quat[0], quat[1], quat[2]])

# Normalize the quaternion
quat_wxyz /= np.linalg.norm(quat_wxyz)

print(quat_wxyz)