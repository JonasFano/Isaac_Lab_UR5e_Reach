import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# === CONFIG ===
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
filename = "domain_rand_model_random_poses_scale_0_01_seed_42"

csv_path = f"/home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/data/real_robot/{filename}.csv"
max_num_episodes = 100

# === Load Data ===
df = pd.read_csv(csv_path)

# Identify columns
target_cols = [f"target_pose_{i}" for i in range(7)]
tcp_quat_cols = [f"tcp_pose_{i}" for i in range(3, 7)]
target_quat_cols = [f"target_pose_{i}" for i in range(3, 7)]

# Detect episode changes
target_change = (df[target_cols] != df[target_cols].shift()).any(axis=1)
target_change.iloc[0] = False
df["episode"] = target_change.cumsum()

# Group by episodes
grouped = list(df.groupby("episode"))[:max_num_episodes]

# Quaternion error function
def quat_angle_error(q1, q2):
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    dot = np.clip(np.dot(q1, q2), -1.0, 1.0)
    return 2 * np.arccos(np.abs(dot)) * (180 / np.pi)  # in degrees

# Storage
episode_rolls = []
episode_pitches = []
episode_yaws = []
orientation_errors = []

prev_target_yaws = []
curr_target_yaws = []
transition_errors = []

# Iterate over episodes
for i, (_, episode_data) in enumerate(grouped):
    first_row = episode_data.iloc[0]
    final_row = episode_data.iloc[-1]

    # Target quaternion and Euler angles
    target_quat = final_row[target_quat_cols].values
    target_rot = R.from_quat(target_quat, scalar_first=True)
    target_euler = target_rot.as_euler('xyz')

    # Final TCP quaternion
    tcp_quat = final_row[tcp_quat_cols].values

    # Orientation error
    error_deg = quat_angle_error(tcp_quat, target_quat)

    # Store orientation data
    episode_rolls.append(target_euler[0])
    episode_pitches.append(target_euler[1])
    episode_yaws.append(target_euler[2])
    orientation_errors.append(error_deg)

    # Store yaw transitions
    if i > 0:
        prev_target_quat = grouped[i - 1][1].iloc[0][target_quat_cols].values
        prev_target_rot = R.from_quat(prev_target_quat, scalar_first=True)
        prev_yaw = prev_target_rot.as_euler('xyz')[2]

        curr_yaw = target_euler[2]

        prev_target_yaws.append(prev_yaw)
        curr_target_yaws.append(curr_yaw)
        transition_errors.append(error_deg)

# Convert to NumPy arrays
episode_rolls = np.array(episode_rolls)
episode_pitches = np.array(episode_pitches)
episode_yaws = np.array(episode_yaws)
orientation_errors = np.array(orientation_errors)
prev_target_yaws = np.array(prev_target_yaws)
curr_target_yaws = np.array(curr_target_yaws)
transition_errors = np.array(transition_errors)

# === Plotting Functions ===
def plot_angle_vs_error(angle_array, angle_name):
    plt.figure(figsize=(10, 5))
    plt.scatter(angle_array, orientation_errors, c=orientation_errors, cmap='coolwarm', s=30)
    plt.colorbar(label='Orientation Error (deg)')
    plt.xlabel(f'Target {angle_name} (radians)')
    plt.ylabel('Final Orientation Error (degrees)')
    plt.title(f'Final Orientation Error vs Target {angle_name}')
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

# === Generate Plots ===
plot_angle_vs_error(episode_rolls, "Roll")
plot_angle_vs_error(episode_pitches, "Pitch")
plot_angle_vs_error(episode_yaws, "Yaw")
plot_yaw_transition_error(prev_target_yaws, curr_target_yaws, transition_errors)
