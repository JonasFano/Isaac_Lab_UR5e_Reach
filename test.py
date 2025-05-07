import numpy as np
from scipy.spatial.transform import Rotation as R
import random

# === Original pose ===
# Format: [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z]

ROT_180_Z = R.from_euler('z', 180, degrees=True)

# pos_x = random.uniform(-0.2, 0.2)
# pos_y = random.uniform(-0.45, -0.25)
# pos_z = random.uniform(0.2, 0.5)
pos_x = random.uniform(0.1, 0.1)
pos_y = random.uniform(-0.3, -0.3)
pos_z = random.uniform(0.3, 0.3)
roll = 0.0
pitch = np.pi  # End-effector z-axis pointing down (180 deg rotation)
# yaw = random.uniform(-2.5*np.pi, -1.5*np.pi) # For wrist_3_joint = 0.0
# yaw = random.uniform(-3.0*np.pi, -1.0*np.pi) # For wrist_3_joint = 0.0

# yaw = random.uniform(-np.pi, -np.pi) # For wrist_3_joint = 0.0
yaw = random.uniform(np.pi, np.pi) # For wrist_3_joint = 0.0

euler_xyz_rad = [roll, pitch, yaw]

# Convert Roll-Pitch-Yaw to Quaternion [w, x, y, z] format
r = R.from_euler("xyz", euler_xyz_rad, degrees=False)
# quat_wxyz = r.as_quat(scalar_first=True)


print("\nEuler angles (degrees) [roll, pitch, yaw]:")
print(r.as_euler('xyz', degrees=True))

print("\nEuler angles (radians) [roll, pitch, yaw]:")
print(euler_xyz_rad)

print("\nEuler angles (radians) [roll, pitch, yaw]:")
print(r.as_euler('xyz', degrees=False))

print("\nRotated Quaternion [w, x, y, z]:")
print(r.as_quat(scalar_first=True))


rotated_quat = ROT_180_Z * r #R.from_quat(quat_wxyz, scalar_first=True)
rotated_quat_wxyz = rotated_quat.as_quat(scalar_first=True)


# Apply 180-degree rotation to the position
rotated_pos = ROT_180_Z.apply([pos_x, pos_y, pos_z])

# === Convert to Euler angles ===
euler_xyz = rotated_quat.as_euler('xyz', degrees=True)  # roll, pitch, yaw in degrees

euler_xyz_rad = rotated_quat.as_euler('xyz', degrees=False)  # roll, pitch, yaw in rad

# === Output ===
print("Rotated Position:")
print(rotated_pos)

print("\nRotated Quaternion [w, x, y, z]:")
print(rotated_quat_wxyz)

print("\nEuler angles (degrees) [roll, pitch, yaw]:")
print(euler_xyz)

print("\nEuler angles (radians) [roll, pitch, yaw]:")
print(euler_xyz_rad)



# [roll, pitch, yaw] = [0, π, -π] radians
# Quaternion [w, x, y, z]: [0, 0, 1, 0]

# [roll, pitch, yaw] = [0, π, π]  # in radians
# Quaternion [w, x, y, z] ≈ [0, 0, -1, 0]
