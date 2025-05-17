# Mainly AI Generated

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# === CONFIGURATION ===
base_path = "/home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/data/real_robot/"
file_groups = {
    ("Standard", "0.05"): [
        "standard_model_random_poses_scale_0_05_seed_24.csv",
        "standard_model_random_poses_scale_0_05_seed_42.csv",
    ],
    ("Standard", "0.01"): [
        "standard_model_random_poses_scale_0_01_seed_24.csv",
        "standard_model_random_poses_scale_0_01_seed_42.csv",
    ],
    ("Curriculum-Shaped", "0.05"): [
        "optimized_model_random_poses_scale_0_05_seed_24.csv",
        "optimized_model_random_poses_scale_0_05_seed_42.csv",
    ],
    ("Curriculum-Shaped", "0.01"): [
        "optimized_model_random_poses_scale_0_01_seed_24.csv",
        "optimized_model_random_poses_scale_0_01_seed_42.csv",
    ],
    ("DomainRand", "0.05"): [
        "domain_rand_model_random_poses_scale_0_05_seed_24.csv",
        "domain_rand_model_random_poses_scale_0_05_seed_42.csv",
    ],
    ("DomainRand", "0.01"): [
        "domain_rand_model_random_poses_scale_0_01_seed_24.csv",
        "domain_rand_model_random_poses_scale_0_01_seed_42.csv",
    ],
    ("Rot6D", "0.05"): [
        "domain_rand_model_random_poses_scale_0_05_seed_24_rotmat.csv",
        "domain_rand_model_random_poses_scale_0_05_seed_42_rotmat.csv",
    ],
    ("Rot6D", "0.01"): [
        "domain_rand_model_random_poses_scale_0_01_seed_24_rotmat.csv",
        "domain_rand_model_random_poses_scale_0_01_seed_42_rotmat.csv",
    ],
}
max_num_episodes = 100

# === Utility Function: Quaternion Angle Error ===
def quat_angle_error(q1, q2):
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    dot_product = np.clip(np.dot(q1, q2), -1.0, 1.0)
    return 2 * np.arccos(np.abs(dot_product))

# === Load and Compute Errors from CSV ===
def compute_errors(csv_path):
    df = pd.read_csv(csv_path)
    target_cols = [f"target_pose_{i}" for i in range(7)]
    target_pos_cols = target_cols[:3]
    target_quat_cols = target_cols[3:7]
    tcp_pos_cols = [f"tcp_pose_{i}" for i in range(3)]
    tcp_quat_cols = [f"tcp_pose_{i}" for i in range(3, 7)]

    target_change = (df[target_cols] != df[target_cols].shift()).any(axis=1)
    target_change.iloc[0] = False
    df["episode"] = target_change.cumsum()
    grouped = list(df.groupby("episode"))[:max_num_episodes]

    pos_errors, quat_errors = [], []
    for _, episode in grouped:
        final = episode.iloc[-1]
        pos_err = np.linalg.norm(final[tcp_pos_cols].values - final[target_pos_cols].values)
        quat_err = quat_angle_error(final[tcp_quat_cols].values, final[target_quat_cols].values)
        pos_errors.append(pos_err)
        quat_errors.append(quat_err)

    return pos_errors, quat_errors

# === Aggregate and Store Errors in Long-Form DataFrame ===
records = []
for (policy, scale), files in file_groups.items():
    for file in files:
        csv_path = os.path.join(base_path, file)
        pos_errors, quat_errors = compute_errors(csv_path)

        for e in pos_errors:
            records.append({
                "Policy": policy,
                "Action Scale": scale,
                "Policy_Scale": f"{policy}_{scale}",
                "Error Type": "Position",
                "Error": e
            })
        for e in quat_errors:
            records.append({
                "Policy": policy,
                "Action Scale": scale,
                "Policy_Scale": f"{policy}_{scale}",
                "Error Type": "Orientation",
                "Error": e
            })

# Create final DataFrame for analysis or plotting
df_errors = pd.DataFrame(records)


# Plot Position Error Boxplot
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_errors[df_errors["Error Type"] == "Position"],
            x="Policy_Scale", y="Error", palette="Set2")
# plt.title("Position (Euclidean) Error")
plt.ylabel("Euclidean Distance [m]")
plt.xlabel("")  # or None
plt.xticks(rotation=45)
plt.grid(True, axis="y")
plt.tight_layout()
plt.show()


# Plot Orientation Error Boxplot
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_errors[df_errors["Error Type"] == "Orientation"],
            x="Policy_Scale", y="Error", palette="Set2")
# plt.title("Orientation (Geodesic) Error")
plt.ylabel("Geodesic Distance [rad]")
plt.xlabel("")  # or None
plt.xticks(rotation=45)
plt.grid(True, axis="y")
plt.tight_layout()
plt.show()
