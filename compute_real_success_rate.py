import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv("data/real_robot/domain_rand_model_random_poses_scale_0_05.csv")

# Define target pose columns
target_cols = [f"target_pose_{i}" for i in range(7)]

# Identify target changes (ignoring first row)
target_change = (df[target_cols] != df[target_cols].shift()).any(axis=1)
target_change.iloc[0] = False  # First row does not count as a change

# Assign episode numbers
df["episode"] = target_change.cumsum()

# Define thresholds
pos_threshold = 0.01  # 1 cm
quat_threshold = 0.06981317  # ~4 degrees

# Define relevant pose columns
tcp_pos_cols = [f"tcp_pose_{i}" for i in range(3)]
target_pos_cols = [f"target_pose_{i}" for i in range(3)]
tcp_quat_cols = [f"tcp_pose_{i}" for i in range(3, 7)]
target_quat_cols = [f"target_pose_{i}" for i in range(3, 7)]

# Quaternion angle error function
def quat_angle_error(q1, q2):
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    dot_product = np.clip(np.dot(q1, q2), -1.0, 1.0)
    return 2 * np.arccos(np.abs(dot_product))

# Track errors and success
pos_errors = []
quat_errors = []
success_flags = []

grouped = list(df.groupby("episode"))

# Evaluate each episode at final timestep
for episode_id, episode_data in grouped[:-1]:
    final_row = episode_data.iloc[-1]
    
    tcp_pos = final_row[tcp_pos_cols].values
    target_pos = final_row[target_pos_cols].values
    tcp_quat = final_row[tcp_quat_cols].values
    target_quat = final_row[target_quat_cols].values

    pos_error = np.linalg.norm(tcp_pos - target_pos)
    quat_error = quat_angle_error(tcp_quat, target_quat)

    pos_errors.append(pos_error)
    quat_errors.append(quat_error)

    success = (pos_error < pos_threshold) and (quat_error < quat_threshold)
    success_flags.append(success)

# Convert to NumPy arrays
pos_errors = np.array(pos_errors)
quat_errors = np.array(quat_errors)
success_flags = np.array(success_flags)

# Compute individual and combined success counts
pos_success_count = (pos_errors < pos_threshold).sum()
quat_success_count = (quat_errors < quat_threshold).sum()
combined_success_rate = success_flags.mean() * 100
pos_only_success_rate = (pos_errors < pos_threshold).mean() * 100
quat_only_success_rate = (quat_errors < quat_threshold).mean() * 100

# Print summary
print(f"Number of episodes: {df['episode'].nunique()-1}")
print(f"Min Position Error: {np.min(pos_errors):.6f}")
print(f"Mean Position Error: {np.mean(pos_errors):.6f}")
print(f"Min Quaternion Error: {np.min(quat_errors):.6f}")
print(f"Mean Quaternion Error: {np.mean(quat_errors):.6f}")
print(f"Position Success Count: {pos_success_count}")
print(f"Quaternion Success Count: {quat_success_count}")
print(f"Position-only Success Rate: {pos_only_success_rate:.2f}%")
print(f"Orientation-only Success Rate: {quat_only_success_rate:.2f}%")
print(f"Combined Success Rate: {combined_success_rate:.2f}%")
