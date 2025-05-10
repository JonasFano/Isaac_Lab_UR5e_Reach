# Mainly AI Generated

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# === CONFIGURATION ===
filename = "domain_rand_model_random_poses_scale_0_05_seed_24_correct_sampling"
csv_path = f"/home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/data/sim/{filename}.csv"
max_num_episodes = 100

# Thresholds
pos_threshold = 0.004  # 4 mm
quat_threshold_rad = 0.05235988  # 3 degrees
pose_change_tolerance = 1e-3  # tolerance for pose change detection

# === LOAD DATA ===
df = pd.read_csv(csv_path)

# Column groups
cmd_cols = [f"pose_command_{i}" for i in range(7)]
tcp_pos_cols = [f"tcp_pose_{i}" for i in range(3)]
cmd_pos_cols = [f"pose_command_{i}" for i in range(3)]
tcp_quat_cols = [f"tcp_pose_{i}" for i in range(3, 7)]
cmd_quat_cols = [f"pose_command_{i}" for i in range(3, 7)]

# Detect episode changes based on pose_command change exceeding tolerance
cmd_diff = (df[cmd_cols] - df[cmd_cols].shift()).abs()
cmd_change = (cmd_diff > pose_change_tolerance).any(axis=1)
cmd_change.iloc[0] = False
df["episode"] = cmd_change.cumsum()

# Group episodes
grouped = list(df.groupby("episode"))[:max_num_episodes]

# === FUNCTION: Quaternion angle error ===
def quat_angle_error(q1, q2):
    q1, q2 = q1 / np.linalg.norm(q1), q2 / np.linalg.norm(q2)
    return 2 * np.arccos(np.clip(np.abs(np.dot(q1, q2)), -1.0, 1.0))  # radians

# === METRICS STORAGE ===
episode_errors_deg = []
pos_errors = []
quat_errors_rad = []
success_flags = []
episode_rolls = []
episode_pitches = []
episode_yaws = []
signflip_counts = []

# === PER-EPISODE ANALYSIS ===
for _, episode_data in grouped:
    tcp_quats = episode_data[tcp_quat_cols].values
    flips = sum(np.dot(tcp_quats[i - 1], tcp_quats[i]) < 0 for i in range(1, len(tcp_quats)))
    signflip_counts.append(flips)

    final = episode_data.iloc[-1]
    tcp_pos = final[tcp_pos_cols].values
    cmd_pos = final[cmd_pos_cols].values
    tcp_quat = final[tcp_quat_cols].values
    cmd_quat = final[cmd_quat_cols].values

    pos_err = np.linalg.norm(tcp_pos - cmd_pos)
    quat_err_rad = quat_angle_error(tcp_quat, cmd_quat)
    quat_err_deg = quat_err_rad * (180 / np.pi)

    pos_errors.append(pos_err)
    quat_errors_rad.append(quat_err_rad)
    episode_errors_deg.append(quat_err_deg)

    success_flags.append((pos_err < pos_threshold) and (quat_err_rad < quat_threshold_rad))

    # Orientation (Euler)
    cmd_rot = R.from_quat(cmd_quat, scalar_first=True)
    cmd_euler = cmd_rot.as_euler('xyz')
    episode_rolls.append(cmd_euler[0])
    episode_pitches.append(np.round(cmd_euler[1], 10))  # reduce noise
    episode_yaws.append(cmd_euler[2])

# === CONVERT TO ARRAYS ===
pos_errors = np.array(pos_errors)
quat_errors_rad = np.array(quat_errors_rad)
success_flags = np.array(success_flags)
episode_errors_deg = np.array(episode_errors_deg)
signflip_counts = np.array(signflip_counts)
episode_rolls = np.array(episode_rolls)
episode_pitches = np.array(episode_pitches)
episode_yaws = np.array(episode_yaws)

# === SUCCESS METRICS ===
print(f"Analysed episodes: {len(grouped)}")
print(f"Min Position Error: {np.min(pos_errors):.6f}")
print(f"Mean Position Error: {np.mean(pos_errors):.6f}")
print(f"Max Position Error: {np.max(pos_errors):.6f}")
print(f"Min Quaternion Error (rad): {np.min(quat_errors_rad):.6f}")
print(f"Mean Quaternion Error (rad): {np.mean(quat_errors_rad):.6f}")
print(f"Max Quaternion Error (rad): {np.max(quat_errors_rad):.6f}")
print(f"Position Success Count: {(pos_errors < pos_threshold).sum()}")
print(f"Quaternion Success Count: {(quat_errors_rad < quat_threshold_rad).sum()}")
print(f"Position-only Success Rate: {(pos_errors < pos_threshold).mean() * 100:.2f}%")
print(f"Orientation-only Success Rate: {(quat_errors_rad < quat_threshold_rad).mean() * 100:.2f}%")
print(f"Combined Success Rate: {success_flags.mean() * 100:.2f}%")

# === PLOTTING FUNCTIONS ===
def plot_angle_vs_error(angle_array, angle_name):
    plt.figure(figsize=(10, 5))
    plt.scatter(angle_array, episode_errors_deg, c=episode_errors_deg, cmap='coolwarm', s=30)
    plt.colorbar(label='Orientation Error (deg)')
    plt.xlabel(f'Commanded {angle_name} (radians)')
    plt.ylabel('Final Orientation Error (degrees)')
    plt.title(f'Final Orientation Error vs Commanded {angle_name}')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_yaw_transition_error(prev_yaws, curr_yaws, errors):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(prev_yaws, curr_yaws, c=errors, cmap='coolwarm', s=40)
    plt.colorbar(scatter, label='Final Orientation Error (deg)')
    plt.xlabel('Previous Target Yaw (rad)')
    plt.ylabel('Current Target Yaw (rad)')
    plt.title('Yaw Transition vs Orientation Error')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# === STANDARD PLOTS ===
plot_angle_vs_error(episode_rolls, "Roll")
plot_angle_vs_error(episode_pitches, "Pitch")
plot_angle_vs_error(episode_yaws, "Yaw")

plt.figure(figsize=(10, 5))
plt.scatter(signflip_counts, episode_errors_deg, c=episode_errors_deg, cmap='viridis', s=30)
plt.colorbar(label='Orientation Error (deg)')
plt.xlabel('Quaternion Sign Flips (per Episode)')
plt.ylabel('Final Orientation Error (degrees)')
plt.title('Orientation Error vs Quaternion Sign Flips')
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.scatter(episode_yaws, signflip_counts, c=signflip_counts, cmap='plasma', s=30)
plt.colorbar(label='Sign Flips')
plt.xlabel('Commanded Yaw (radians)')
plt.ylabel('Sign Flips in TCP Quaternion')
plt.title('Quaternion Sign Flips vs Commanded Yaw')
plt.grid(True)
plt.tight_layout()
plt.show()

# === YAW TRANSITION PLOT ===
prev_target_yaws, curr_target_yaws, transition_errors = [], [], []

for i in range(1, len(grouped)):
    prev_cmd_quat = grouped[i - 1][1].iloc[0][cmd_quat_cols].values
    curr_cmd_quat = grouped[i][1].iloc[0][cmd_quat_cols].values

    prev_yaw = R.from_quat(prev_cmd_quat, scalar_first=True).as_euler('xyz')[2]
    curr_yaw = R.from_quat(curr_cmd_quat, scalar_first=True).as_euler('xyz')[2]

    prev_target_yaws.append(prev_yaw)
    curr_target_yaws.append(curr_yaw)
    transition_errors.append(episode_errors_deg[i])  # orientation error of current episode

plot_yaw_transition_error(np.array(prev_target_yaws), np.array(curr_target_yaws), np.array(transition_errors))
