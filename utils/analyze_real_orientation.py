# Mainly AI Generated

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# === CONFIGURATION ===
# filename = "standard_model_random_poses_scale_0_05_seed_24"
# filename = "standard_model_random_poses_scale_0_05_seed_42"
# filename = "standard_model_random_poses_scale_0_01_seed_24"
# filename = "standard_model_random_poses_scale_0_01_seed_42"
# filename = "optimized_model_random_poses_scale_0_05_seed_24"
# filename = "optimized_model_random_poses_scale_0_05_seed_42"
# filename = "optimized_model_random_poses_scale_0_01_seed_24"
# filename = "optimized_model_random_poses_scale_0_01_seed_42"
filename = "domain_rand_model_random_poses_scale_0_05_seed_24"
# filename = "domain_rand_model_random_poses_scale_0_05_seed_42"
# filename = "domain_rand_model_random_poses_scale_0_01_seed_24"
# filename = "domain_rand_model_random_poses_scale_0_01_seed_42"

# filename = "domain_rand_model_random_poses_scale_0_05_seed_24_correct_sampling"
# filename = "orientation_and_sampling_issues/domain_rand_model_random_poses_scale_0_05_seed_24"

csv_path = f"/home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/data/real_robot/{filename}.csv"
max_num_episodes = 100 #100

# === LOAD DATA ===
df = pd.read_csv(csv_path)

# Define relevant columns
target_cols = [f"target_pose_{i}" for i in range(7)]
tcp_quat_cols = [f"tcp_pose_{i}" for i in range(3, 7)]
target_quat_cols = [f"target_pose_{i}" for i in range(3, 7)]

# Detect episode changes based on target pose change
target_change = (df[target_cols] != df[target_cols].shift()).any(axis=1)
target_change.iloc[0] = False
df["episode"] = target_change.cumsum()

# Group by episode
grouped = list(df.groupby("episode"))[:max_num_episodes]

# === UTILITIES ===
def quat_angle_error(q1, q2):
    """Returns quaternion angle error in degrees."""
    q1, q2 = q1 / np.linalg.norm(q1), q2 / np.linalg.norm(q2)
    dot = np.clip(np.dot(q1, q2), -1.0, 1.0)
    return 2 * np.arccos(np.abs(dot)) * (180 / np.pi)

def plot_angle_vs_error(angle_array, angle_name, orientation_errors):
    plt.figure(figsize=(10, 5))
    plt.scatter(angle_array, orientation_errors, c=orientation_errors, cmap='coolwarm', s=30)
    # plt.colorbar(label='Geodesic Error [deg]')
    plt.xlabel(f'Target {angle_name} [rad]')
    plt.ylabel('Final Geodesic Error [deg]')
    # plt.title(f'Final Geodesic Error vs Target {angle_name}')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_yaw_transition_error(prev_yaws, curr_yaws, errors):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(prev_yaws, curr_yaws, c=errors, cmap='coolwarm', s=40)
    plt.colorbar(scatter, label='Final Geodesic Error [deg]')
    plt.xlabel('Previous Target Yaw [rad]')
    plt.ylabel('Target Yaw [rad]')
    # plt.title('Yaw Transition vs Orientation Error')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_tcp_to_target_yaw_transition(prev_tcp_yaws, curr_target_yaws, errors):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(prev_tcp_yaws, curr_target_yaws, c=errors, cmap='coolwarm', s=40)
    plt.colorbar(scatter, label='Final Geodesic Error [deg]')
    plt.xlabel('Previous Episode - Final TCP Yaw [rad]')
    plt.ylabel('Target Yaw [rad]')
    # plt.title('TCP-to-Target Yaw Transition vs Orientation Error')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# === ANALYSIS STORAGE ===
episode_rolls = []
episode_pitches = []
episode_yaws = []
orientation_errors = []
prev_target_yaws = []
prev_tcp_yaws = []
curr_target_yaws = []
gedoesic_errors = []

# === PER-EPISODE ANALYSIS ===
for i, (_, episode_data) in enumerate(grouped):
    final_row = episode_data.iloc[-1]
    target_quat = final_row[target_quat_cols].values
    tcp_quat = final_row[tcp_quat_cols].values

    prev_tcp_quat = grouped[i - 1][1].iloc[-1][tcp_quat_cols].values
    prev_tcp_yaw = R.from_quat(prev_tcp_quat, scalar_first=True).as_euler('xyz')[2]

    # Orientation error
    error_deg = quat_angle_error(tcp_quat, target_quat)
    orientation_errors.append(error_deg)

    # Convert target quaternion to Euler (XYZ)
    target_euler = R.from_quat(target_quat, scalar_first=True).as_euler('xyz')
    episode_rolls.append(target_euler[0])
    episode_pitches.append(np.round(target_euler[1], 6))  # reduce noise
    episode_yaws.append(target_euler[2])

    # Track yaw transitions
    if i > 0:
        prev_quat = grouped[i - 1][1].iloc[0][target_quat_cols].values
        prev_yaw = R.from_quat(prev_quat, scalar_first=True).as_euler('xyz')[2]
        curr_yaw = target_euler[2]

        prev_target_yaws.append(prev_yaw)
        curr_target_yaws.append(curr_yaw)
        gedoesic_errors.append(error_deg)
        prev_tcp_yaws.append(prev_tcp_yaw)


# === CONVERT TO ARRAYS ===
episode_rolls = np.array(episode_rolls)
episode_pitches = np.array(episode_pitches)
episode_yaws = np.array(episode_yaws)
orientation_errors = np.array(orientation_errors)
prev_target_yaws = np.array(prev_target_yaws)
prev_tcp_yaws = np.array(prev_tcp_yaws)
curr_target_yaws = np.array(curr_target_yaws)
gedoesic_errors = np.array(gedoesic_errors)



# === PLOTTING ===
plot_angle_vs_error(episode_rolls, "Roll", orientation_errors)
plot_angle_vs_error(episode_pitches, "Pitch", orientation_errors)
plot_angle_vs_error(episode_yaws, "Yaw", orientation_errors)
plot_yaw_transition_error(prev_target_yaws, curr_target_yaws, gedoesic_errors)
plot_tcp_to_target_yaw_transition(prev_tcp_yaws, curr_target_yaws[:], gedoesic_errors)
