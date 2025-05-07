import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# === CONFIG ===
filename = "rel_ik_sb3_ppo_ur5e_reach_0_05_pose_without_gripper_1000_trials"
csv_path = f"/home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/data/sim/{filename}.csv"
max_num_episodes = 1000

# === Load Data ===
df = pd.read_csv(csv_path)

# Identify columns
cmd_cols = [f"pose_command_{i}" for i in range(7)]
tcp_quat_cols = [f"tcp_pose_{i}" for i in range(3, 7)]
cmd_quat_cols = [f"pose_command_{i}" for i in range(3, 7)]

# Detect episode changes based on pose_command change
cmd_change = (df[cmd_cols] != df[cmd_cols].shift()).any(axis=1)
cmd_change.iloc[0] = False
df["episode"] = cmd_change.cumsum()

# Group by episodes
grouped = list(df.groupby("episode"))[:max_num_episodes]

# Quaternion angle error function
def quat_angle_error(q1, q2):
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    dot_product = np.clip(np.dot(q1, q2), -1.0, 1.0)
    return 2 * np.arccos(np.abs(dot_product))  # radians

# Storage
episode_rolls = []
episode_pitches = []
episode_yaws = []
episode_errors_deg = []
signflip_counts = []

for _, episode_data in grouped:
    # === Detect sign flips in tcp_pose quaternion within the episode ===
    tcp_quats = episode_data[tcp_quat_cols].values
    flips = 0
    for i in range(1, len(tcp_quats)):
        q_prev = tcp_quats[i - 1]
        q_curr = tcp_quats[i]
        if np.dot(q_prev, q_curr) < 0:  # Sign flip detected
            flips += 1
    signflip_counts.append(flips)

    # === Final orientation error ===
    final_row = episode_data.iloc[-1]
    tcp_quat = final_row[tcp_quat_cols].values
    cmd_quat = final_row[cmd_quat_cols].values

    quat_err_rad = quat_angle_error(tcp_quat, cmd_quat)
    quat_err_deg = quat_err_rad * (180 / np.pi)
    episode_errors_deg.append(quat_err_deg)

    # === Commanded orientation in RPY ===
    cmd_rot = R.from_quat(cmd_quat, scalar_first=True)
    cmd_euler = cmd_rot.as_euler('xyz')

    episode_rolls.append(cmd_euler[0])
    episode_pitches.append(np.round(cmd_euler[1], decimals=10))  # Remove noise
    episode_yaws.append(cmd_euler[2])

# Convert to arrays
episode_rolls = np.array(episode_rolls)
episode_pitches = np.array(episode_pitches)
episode_yaws = np.array(episode_yaws)
episode_errors_deg = np.array(episode_errors_deg)
signflip_counts = np.array(signflip_counts)

# === Plotting Function ===
def plot_angle_vs_error(angle_array, angle_name):
    plt.figure(figsize=(10, 5))
    plt.scatter(angle_array, episode_errors_deg, c=episode_errors_deg, cmap='coolwarm', s=30)
    plt.colorbar(label='Orientation Error (deg)')
    plt.xlabel(f'Commanded {angle_name} (radians)')
    plt.ylabel('Final Orientation Error (degrees)')
    plt.title(f'Final Orientation Error vs Commanded {angle_name} per Episode')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# === Generate Orientation Plots ===
plot_angle_vs_error(episode_rolls, "Roll")
plot_angle_vs_error(episode_pitches, "Pitch")
plot_angle_vs_error(episode_yaws, "Yaw")

# === Plot: Orientation Error vs Sign Flip Count in tcp_pose per Episode ===
plt.figure(figsize=(10, 5))
plt.scatter(signflip_counts, episode_errors_deg, c=episode_errors_deg, cmap='viridis', s=30)
plt.colorbar(label='Orientation Error (deg)')
plt.xlabel('Number of Quaternion Sign Flips in TCP Pose per Episode')
plt.ylabel('Final Orientation Error (degrees)')
plt.title('Orientation Error vs Sign Flips in TCP Pose Quaternion (Simulation)')
plt.grid(True)
plt.tight_layout()
plt.show()


# === Plot: Sign Flip Count vs Commanded Yaw ===
plt.figure(figsize=(10, 5))
plt.scatter(episode_yaws, signflip_counts, c=signflip_counts, cmap='plasma', s=30)
plt.colorbar(label='Number of Sign Flips')
plt.xlabel('Commanded Yaw (radians)')
plt.ylabel('Sign Flips in TCP Quaternion (per Episode)')
plt.title('Quaternion Sign Flips vs Commanded Yaw')
plt.grid(True)
plt.tight_layout()
plt.show()
