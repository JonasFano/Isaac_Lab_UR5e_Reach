# Mainly AI Generated

import pandas as pd
import numpy as np

# === CONFIGURATION ===
# filename = "standard_model_random_poses_scale_0_05_seed_24"
# filename = "standard_model_random_poses_scale_0_05_seed_42"
# filename = "standard_model_random_poses_scale_0_01_seed_24"
# filename = "standard_model_random_poses_scale_0_01_seed_42"
# filename = "optimized_model_random_poses_scale_0_05_seed_24"
# filename = "optimized_model_random_poses_scale_0_05_seed_42"
# filename = "optimized_model_random_poses_scale_0_01_seed_24"
# filename = "optimized_model_random_poses_scale_0_01_seed_42"
# filename = "domain_rand_model_random_poses_scale_0_05_seed_24"
# filename = "domain_rand_model_random_poses_scale_0_05_seed_42"
# filename = "domain_rand_model_random_poses_scale_0_01_seed_24"
# filename = "domain_rand_model_random_poses_scale_0_01_seed_42"

# filename = "domain_rand_model_random_poses_scale_0_05_seed_24_correct_sampling"

filename = "domain_rand_model_random_poses_scale_0_05_seed_24_wrong_quat"

csv_path = f"/home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/data/real_robot/{filename}.csv"
max_num_episodes = 100

# Thresholds
pos_threshold = 0.004          # 4 mm
quat_threshold_rad = 0.05235988  # 3 degrees

# === LOAD DATA ===
df = pd.read_csv(csv_path)

# Define columns
target_cols = [f"target_pose_{i}" for i in range(7)]
target_pos_cols = [f"target_pose_{i}" for i in range(3)]
target_quat_cols = [f"target_pose_{i}" for i in range(3, 7)]
tcp_pos_cols = [f"tcp_pose_{i}" for i in range(3)]
tcp_quat_cols = [f"tcp_pose_{i}" for i in range(3, 7)]

# Detect episode changes
target_change = (df[target_cols] != df[target_cols].shift()).any(axis=1)
target_change.iloc[0] = False
df["episode"] = target_change.cumsum()

# Group by episodes
grouped = list(df.groupby("episode"))[:max_num_episodes]

# === UTILITIES ===
def quat_angle_error(q1, q2):
    """Compute quaternion angular difference in radians."""
    q1, q2 = q1 / np.linalg.norm(q1), q2 / np.linalg.norm(q2)
    return 2 * np.arccos(np.clip(np.abs(np.dot(q1, q2)), -1.0, 1.0))

# === EVALUATE EACH EPISODE ===
pos_errors = []
quat_errors = []
success_flags = []

for _, episode_data in grouped:
    final = episode_data.iloc[-1]

    tcp_pos = final[tcp_pos_cols].values
    target_pos = final[target_pos_cols].values
    tcp_quat = final[tcp_quat_cols].values
    target_quat = final[target_quat_cols].values

    pos_err = np.linalg.norm(tcp_pos - target_pos)
    quat_err = quat_angle_error(tcp_quat, target_quat)

    pos_errors.append(pos_err)
    quat_errors.append(quat_err)
    success_flags.append((pos_err < pos_threshold) and (quat_err < quat_threshold_rad))

# === CONVERT TO NUMPY ===
pos_errors = np.array(pos_errors)
quat_errors = np.array(quat_errors)
success_flags = np.array(success_flags)

# === METRICS ===
pos_success_count = (pos_errors < pos_threshold).sum()
quat_success_count = (quat_errors < quat_threshold_rad).sum()
combined_success_rate = success_flags.mean() * 100
pos_only_success_rate = (pos_errors < pos_threshold).mean() * 100
quat_only_success_rate = (quat_errors < quat_threshold_rad).mean() * 100

# === PRINT SUMMARY ===
print(f"Analysed episodes: {len(grouped)}")
print(f"Min Position Error:        {np.min(pos_errors):.6f}")
print(f"Mean Position Error:       {np.mean(pos_errors):.6f}")
print(f"Max Position Error:        {np.max(pos_errors):.6f}")
print(f"Min Quaternion Error (rad): {np.min(quat_errors):.6f}")
print(f"Mean Quaternion Error (rad): {np.mean(quat_errors):.6f}")
print(f"Max Quaternion Error (rad): {np.max(quat_errors):.6f}")
print(f"Position Success Count:    {pos_success_count}")
print(f"Quaternion Success Count:  {quat_success_count}")
print(f"Position-only Success Rate:     {pos_only_success_rate:.2f}%")
print(f"Orientation-only Success Rate:  {quat_only_success_rate:.2f}%")
print(f"Combined Success Rate:          {combined_success_rate:.2f}%")
