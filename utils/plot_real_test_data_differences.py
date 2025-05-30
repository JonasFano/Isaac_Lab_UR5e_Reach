# Mainly AI Generated

import pandas as pd
import plotly.graph_objects as go
import os
import numpy as np
from scipy.spatial.transform import Rotation as R

# filename = "standard_model_predefined_poses_scale_0_05_seed_24"
# filename = "standard_model_predefined_poses_scale_0_05_seed_42"
# filename = "standard_model_predefined_poses_scale_0_01_seed_24"
# filename = "standard_model_predefined_poses_scale_0_01_seed_42"
# filename = "optimized_model_predefined_poses_scale_0_05_seed_24"
# filename = "optimized_model_predefined_poses_scale_0_05_seed_42"
filename = "optimized_model_predefined_poses_scale_0_01_seed_24"
# filename = "optimized_model_predefined_poses_scale_0_01_seed_42"
# filename = "domain_rand_model_predefined_poses_scale_0_05_seed_24"
# filename = "domain_rand_model_predefined_poses_scale_0_05_seed_42"
# filename = "domain_rand_model_predefined_poses_scale_0_01_seed_24"
# filename = "domain_rand_model_predefined_poses_scale_0_01_seed_42"

# filename = "domain_rand_model_predefined_poses_scale_0_05_seed_24_rotmat"
# filename = "domain_rand_model_predefined_poses_scale_0_05_seed_42_rotmat"
# filename = "domain_rand_model_predefined_poses_scale_0_01_seed_24_rotmat"
# filename = "domain_rand_model_predefined_poses_scale_0_01_seed_42_rotmat"


csv_path = "/home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/data/real_robot/smoothness_analysis/" + filename + ".csv"

# Define a threshold
# threshold = 0.001 # For action scaling: 0.05
threshold = 0.0002 # For action scaling 0.01

# Thresholds
euclidean_thresh = 0.004  # in meters
geodesic_thresh = 0.05235988  # in radians (~3 degrees)

save = False # False True
plot = True

# Output directory for saving the plots
# output_dir = "/home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/plots"
output_dir = "/home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/plots/real_robot"
os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

# Load the data
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"CSV file not found at {csv_path}")

data = pd.read_csv(csv_path)

# Replace NaN values with zero (or interpolate if necessary)
data.fillna(0, inplace=True)

# Ensure numeric data types
data = data.apply(pd.to_numeric, errors='coerce')

# Round numerical values to 4 decimal places
# data = data.round(4)

# Extract columns for plotting
if "timestep" in data.columns:
    timesteps = data["timestep"]
else:
    raise KeyError("Column 'timestep' not found in the dataset.")

# Dynamically extract valid columns
tcp_pose_cols = [col for col in data.columns if "tcp_pose" in col]
pose_command_cols = [col for col in data.columns if "target_pose" in col]
actions_cols = [col for col in data.columns if "last_action" in col]

tcp_pose = data[tcp_pose_cols]
pose_command = data[pose_command_cols]
actions = data[actions_cols] # Given in robot base frame


# Compute TCP displacement
tcp_displacement = tcp_pose.diff().fillna(0)  # Difference between consecutive timesteps

# Rename columns to tcp_displacement
tcp_displacement.columns = [col.replace("tcp_pose", "tcp_displacement") for col in tcp_displacement.columns]

# print(tcp_pose)
# print(tcp_displacement)
# print(actions)

# Compute TCP Pose - Pose Command difference
tcp_pose_error = pose_command.values - tcp_pose.values

tcp_pose_error = pd.DataFrame(tcp_pose_error, columns=[col.replace("tcp_pose", "tcp_pose_error") for col in tcp_pose.columns], index=data.index)

amount = min(3, len(tcp_pose.columns))  # Ensure we only use available columns


# Function to create and save individual plots
def create_and_save_plot(y_data, amount, title, filename):
    fig = go.Figure()
    for i in range(amount):
        column_name = y_data.columns[i]
        fig.add_trace(go.Scatter(x=timesteps, y=y_data[column_name], mode='lines', name=f"{column_name}"))
    fig.update_layout(title=title, xaxis_title="Timestep", hovermode="x unified")

    if save:
        # Save the figure as PDF and PNG
        png_path = os.path.join(output_dir, filename + ".png")
        fig.write_image(png_path, width=1100, height=600)  # Save as PNG

    fig.show()


# Function to create and save individual plots
def create_and_save_comparison_plot(y_data1, y_data2, amount, title, filename):
    fig = go.Figure()
    for i in range(amount):
        column_name1 = y_data1.columns[i]
        column_name2 = y_data2.columns[i]
        fig.add_trace(go.Scatter(x=timesteps, y=y_data1[column_name1], mode='lines', name=f"{column_name1}"))
        fig.add_trace(go.Scatter(x=timesteps, y=y_data2[column_name2], mode='lines', name=f"{column_name2}"))
    fig.update_layout(title=title, xaxis_title="Timestep", hovermode="x unified")

    if save:
        # Save the figure as PDF and PNG
        png_path = os.path.join(output_dir, filename + ".png")
        fig.write_image(png_path, width=1000, height=600)  # Save as PNG

    fig.show()


def create_and_save_tripple_comparison_plot(y_data1, y_data2, y_data3, amount, title, filename):
    fig = go.Figure()
    for i in range(amount):
        column_name1 = y_data1.columns[i]
        column_name2 = y_data2.columns[i]
        column_name3 = y_data3.columns[i]
        fig.add_trace(go.Scatter(x=timesteps, y=y_data1[column_name1], mode='lines', name=f"{column_name1}"))
        fig.add_trace(go.Scatter(x=timesteps, y=y_data2[column_name2], mode='lines', name=f"{column_name2}"))
        fig.add_trace(go.Scatter(x=timesteps, y=y_data3[column_name3], mode='lines', name=f"{column_name3}"))
    fig.update_layout(title=title, xaxis_title="Timestep", hovermode="x unified")

    if save:
        # Save the figure as PDF and PNG
        png_path = os.path.join(output_dir, filename + ".png")
        fig.write_image(png_path, width=1000, height=600)  # Save as PNG

    fig.show()



# With noise
# create_and_save_plot(tcp_displacement, amount, "TCP Displacement", "tcp_displacement_with_noise")
# create_and_save_plot(actions, amount, "Actions", "actions_with_noise")
# create_and_save_plot(tcp_pose_error, amount, "TCP Position Error (TCP - Pose Command)", "tcp_position_error_with_noise")
# create_and_save_comparison_plot(tcp_pose_error, actions, 1, "Comparison between TCP Position Error and Actions", "comparison_tcp_position_error_and_action_with_noise")
# create_and_save_comparison_plot(tcp_displacement, actions, 1, "Comparison between TCP Displacements and Actions", "comparison_tcp_displacement_and_action_with_noise")


# Without noise
# create_and_save_plot(tcp_displacement, amount, "TCP Displacement", "tcp_displacement")
# create_and_save_plot(actions, amount, "Actions", "actions")
# create_and_save_plot(tcp_pose_error, amount, "TCP Position Error (TCP - Pose Command)", "tcp_position_error")
# create_and_save_comparison_plot(tcp_pose_error, actions, 1, "Comparison between TCP Position Error and Actions", "comparison_tcp_position_error_and_action")
# create_and_save_comparison_plot(tcp_displacement, actions, 1, "Comparison between TCP Displacements and Actions", "comparison_tcp_displacement_and_action")
# create_and_save_tripple_comparison_plot(tcp_displacement, tcp_pose_error, actions, 1, "Comparison between TCP Displacements, TCP Position Error and Actions", "comparison_tcp_displacement_tcp_position_error_and_actions")


# Only visualize TCP Displacement vs Actions
# create_and_save_comparison_plot(tcp_displacement, actions, 3, "Comparison between TCP Displacements and Actions - " + filename, "comparison_tcp_displacement_and_action_" + filename)
# create_and_save_comparison_plot(tcp_pose_error, actions, 3, "Comparison between TCP Position Error and Actions - " + filename, "comparison_tcp_position_error_and_action_" + filename)



# Match corresponding columns manually
tcp_displacement_cols = [f"tcp_displacement_{i}" for i in range(6)]




# Only select position components (first three)
tcp_displacement_pos = tcp_displacement[tcp_displacement_cols[:3]].values  # [:3] = x, y, z
actions_pos = actions[actions_cols[:3]].values

# Get absolute values
tcp_displacement_abs = np.abs(tcp_displacement_pos)
actions_abs = np.abs(actions_pos)

# Create a mask: keep timesteps where any position dimension exceeds the threshold
mask = (tcp_displacement_abs > threshold) | (actions_abs > threshold)
mask = mask.any(axis=1)  # At least one position dimension must exceed threshold

# Apply mask
filtered_tcp_displacement = tcp_displacement_pos[mask]
filtered_actions = actions_pos[mask]

print(f"Number of selected timesteps after filtering: {filtered_tcp_displacement.shape[0]} / {tcp_displacement.shape[0]}")

# Now compute errors on the filtered subset
error = filtered_tcp_displacement - filtered_actions
mse_per_dimension = (error ** 2).mean(axis=0)
mae_per_dimension = np.abs(error).mean(axis=0)
max_error_per_dimension = np.abs(error).max(axis=0)

# Print nicely
for i in range(3):
    print(f"Position Dimension {i}: MSE = {mse_per_dimension[i]:.6f}, MAE = {mae_per_dimension[i]:.6f}, Max Error = {max_error_per_dimension[i]:.6f}")

print("\nMean MSE across position dimensions:", np.mean(mse_per_dimension))
print("Mean MAE across position dimensions:", np.mean(mae_per_dimension))
print("Total Max Error across position dimensions:", np.max(max_error_per_dimension))



delta_t = 0.02  # seconds

# Displacement per step -> velocity in m/s
filtered_tcp_velocity = filtered_tcp_displacement / delta_t

# Then second derivative
acceleration = np.diff(filtered_tcp_velocity, axis=0) / delta_t

# Compute squared norms
squared_acc = np.sum(acceleration**2, axis=1)

# Sum over all timesteps
smoothness = np.sum(squared_acc)

print(f"Smoothness (Sum of Squared Accelerations): {smoothness:.6f} m²/s⁴")










# --- Compute Euclidean and Geodesic Distances ---

# Euclidean distance between positions
euclidean_distance = np.linalg.norm(
    tcp_pose.iloc[:, :3].values - pose_command.iloc[:, :3].values, axis=1
)

# Geodesic distance using scipy Rotation
tcp_rot = R.from_quat(tcp_pose.iloc[:, 3:7].values)
target_rot = R.from_quat(pose_command.iloc[:, 3:7].values)
relative_rot = target_rot.inv() * tcp_rot
geodesic_distance = relative_rot.magnitude()  # Angle in radians


if plot:
    # --- Plotting ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=timesteps, y=euclidean_distance, mode='lines', name="Euclidean Distance (m)"))
    fig.add_trace(go.Scatter(x=timesteps, y=geodesic_distance, mode='lines', name="Geodesic Distance (rad)"))

    fig.update_layout(
        title="Euclidean and Geodesic Distance over Timesteps",
        xaxis_title="Timestep",
        yaxis_title="Distance",
        hovermode="x unified"
    )

    if save:
        fig.write_image(os.path.join(output_dir, f"distance_plot_{filename}.png"), width=1100, height=600)

    fig.show()




# Define ranges
# ranges = [(0, 251), (252, 503), (504, len(timesteps)-1)]
ranges = [(0, 501), (502, 1003), (1004, len(timesteps)-1)]

print("\n--- Threshold Check for Min Distances ---")
for start, end in ranges:
    if end > len(euclidean_distance):  # Avoid index errors if data is shorter
        end = len(euclidean_distance)

    euc_slice = euclidean_distance[start:end]
    geo_slice = geodesic_distance[start:end]

    # Find indices where both conditions are met
    mask = (euc_slice < euclidean_thresh) & (geo_slice < geodesic_thresh)

    if np.any(mask):
        idx = np.where(mask)[0][0] + start  # Get global timestep index
        print(f"✅ Timestep {idx} in range {start}-{end}: Euclidean = {euclidean_distance[idx]:.6f} m, Geodesic = {geodesic_distance[idx]:.6f} rad")
    else:
        # No timestep met both conditions – show best individual values
        min_euc = np.min(euc_slice)
        min_geo = np.min(geo_slice)
        idx_euc = np.argmin(euc_slice) + start
        idx_geo = np.argmin(geo_slice) + start
        print(f"❌ No match in {start}-{end}:")
        print(f"   Min Euclidean = {min_euc:.6f} m at timestep {idx_euc}")
        print(f"   Min Geodesic  = {min_geo:.6f} rad at timestep {idx_geo}")
