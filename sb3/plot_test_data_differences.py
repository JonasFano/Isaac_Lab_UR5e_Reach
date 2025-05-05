import pandas as pd
import plotly.graph_objects as go
import os
import numpy as np
from scipy.spatial.transform import Rotation as R

# Path to the CSV file
# filename = "rel_ik_sb3_ppo_ur5e_reach_0_05_pose_without_gripper_action_rate_pos_0_1_step_16000"
# filename = "rel_ik_sb3_ppo_ur5e_reach_0_05_pose_without_gripper_action_rate_pos_1_0_step_16000"
# filename = "rel_ik_sb3_ppo_ur5e_reach_0_05_pose_without_gripper_action_rate_0_8_step_16000"
# filename = "rel_ik_sb3_ppo_ur5e_reach_0_05_pose_without_gripper_action_rate_1_0_step_16000"
# filename = "rel_ik_sb3_ppo_ur5e_reach_0_05_pose_without_gripper_action_magnitude_pos_0_8_step_16000"
# filename = "rel_ik_sb3_ppo_ur5e_reach_0_05_pose_without_gripper_action_magnitude_0_05_step_16000"
# filename = "rel_ik_sb3_ppo_ur5e_reach_0_05_pose_without_gripper_action_magnitude_0_01_step_16000"
# filename = "rel_ik_sb3_ppo_ur5e_reach_0_05_pose_without_gripper_action_rate_0_1_action_magnitude_0_01_step_16000"
# filename = "rel_ik_sb3_ppo_ur5e_reach_0_05_pose_without_gripper_action_rate_0_5_action_magnitude_0_02_step_16000"
# filename = "rel_ik_sb3_ppo_ur5e_reach_0_05_pose_without_gripper"
# filename = "impedance_ctrl_sb3_ppo_ur5e_reach_0_05_pose_without_gripper"

# filename = "rel_ik_sb3_ppo_ur5e_reach_0_05_pose_without_gripper_domain_rand_robot_initial_joints_0_9_and_gains"
# filename = "rel_ik_sb3_ppo_ur5e_reach_0_05_pose_without_gripper_domain_rand_robot_initial_joints_0_8_and_gains"
filename = "rel_ik_sb3_ppo_ur5e_reach_0_05_pose_without_gripper_domain_rand_robot_initial_joints_0_7_and_gains"

csv_path = "/home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/data/sim/" + filename + ".csv"

save = False # False True

# Output directory for saving the plots
# output_dir = "/home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/plots"
output_dir = "/home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/plots/comparison"
os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

# Load the data
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"CSV file not found at {csv_path}")

data = pd.read_csv(csv_path)

# Replace NaN values with zero (or interpolate if necessary)
data.fillna(0, inplace=True)

# Ensure numeric data types
data = data.apply(pd.to_numeric, errors='coerce')

max_timesteps = 748 #500

if max_timesteps is not None:
    data = data.iloc[:max_timesteps]

# Round numerical values to 4 decimal places
# data = data.round(4)

# Extract columns for plotting
if "timestep" in data.columns:
    timesteps = data["timestep"]
else:
    raise KeyError("Column 'timestep' not found in the dataset.")

# Dynamically extract valid columns
tcp_pose_cols = [col for col in data.columns if "tcp_pose" in col]
pose_command_cols = [col for col in data.columns if "pose_command" in col]
actions_cols = [col for col in data.columns if "actions" in col]

tcp_pose = data[tcp_pose_cols]
pose_command = data[pose_command_cols]
actions = data[actions_cols] * 0.05 # 0.1 # Given in robot base frame



# Convert Quaternion to Axis-Angle before computing displacement
def quaternion_to_axis_angle(quat):
    """ Convert a quaternion (x, y, z, w) to an axis-angle representation """
    r = R.from_quat(quat)
    axis_angle = r.as_rotvec()  # Returns axis-angle representation
    return axis_angle

# Extract position (first 3) and quaternion (last 4) separately
tcp_position = tcp_pose.iloc[:, :3]  # First three columns (XYZ)
tcp_quaternion = tcp_pose.iloc[:, 3:7]  # Last four columns (Quaternion XYZW)

# Apply the conversion function row-wise to the quaternion columns
tcp_axis_angle = np.vstack(tcp_quaternion.apply(lambda q: quaternion_to_axis_angle(q.values), axis=1))

# Convert to DataFrame and rename columns
tcp_axis_angle_df = pd.DataFrame(tcp_axis_angle, columns=["tcp_pose_3", "tcp_pose_4", "tcp_pose_5"], index=tcp_pose.index)

# Reassemble TCP Pose with position + axis-angle
tcp_pose_transformed = pd.concat([tcp_position, tcp_axis_angle_df], axis=1)

# Compute TCP displacement in transformed space
tcp_displacement = tcp_pose_transformed.diff().fillna(0)


# # Compute TCP displacement
# tcp_displacement = tcp_pose.diff().fillna(0)  # Difference between consecutive timesteps

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
create_and_save_comparison_plot(tcp_displacement, actions, 3, "Comparison between TCP Displacements and Actions - " + filename, "comparison_tcp_displacement_and_action_" + filename)
# create_and_save_comparison_plot(tcp_displacement, actions, 6, "Comparison between TCP Displacements and Actions - " + filename, "comparison_tcp_displacement_and_action_" + filename)



# Match corresponding columns manually
tcp_displacement_cols = [f"tcp_displacement_{i}" for i in range(6)]
actions_cols = [f"actions_{i}" for i in range(6)]


# rel_ik_sb3_ppo_ur5e_reach_0_05_pose_without_gripper
# Mean MSE across position dimensions: 0.002708326068215776
# Mean MAE across position dimensions: 0.028883456477092184
# Total Max Error across position dimensions: 0.193931132555008
# Smoothness (Sum of Squared Accelerations): 0.011345

# rel_ik_sb3_ppo_ur5e_reach_0_05_pose_without_gripper_action_rate_pos_0_1_step_16000
# Mean MSE across position dimensions: 0.0006979464525297263
# Mean MAE across position dimensions: 0.01606827213001112
# Total Max Error across position dimensions: 0.08964138925075535
# Smoothness (Sum of Squared Accelerations): 33489.872660 m²/s⁴

# rel_ik_sb3_ppo_ur5e_reach_0_05_pose_without_gripper_action_rate_pos_1_0_step_16000
# Mean MSE across position dimensions: 0.00011988212159580724
# Mean MAE across position dimensions: 0.00612532964959327
# Total Max Error across position dimensions: 0.041737221181392656


# rel_ik_sb3_ppo_ur5e_reach_0_05_pose_without_gripper_action_rate_0_8_step_16000
# Mean MSE across position dimensions: 0.00030663767204684456
# Mean MAE across position dimensions: 0.008501266520599736
# Total Max Error across position dimensions: 0.07829238474369048
# Smoothness (Sum of Squared Accelerations): 0.004127


# rel_ik_sb3_ppo_ur5e_reach_0_05_pose_without_gripper_action_rate_1_0_step_16000
# Mean MSE across position dimensions: 7.467854633694337e-05
# Mean MAE across position dimensions: 0.00436464059708077
# Total Max Error across position dimensions: 0.03550783693790433


# rel_ik_sb3_ppo_ur5e_reach_0_05_pose_without_gripper_action_magnitude_pos_0_8_step_16000
# Mean MSE across position dimensions: 0.00014908182648106034
# Mean MAE across position dimensions: 0.00717901318451671
# Total Max Error across position dimensions: 0.03712572008371352


# rel_ik_sb3_ppo_ur5e_reach_0_05_pose_without_gripper_action_magnitude_0_05_step_16000
# Mean MSE across position dimensions: 0.00048177464049099366
# Mean MAE across position dimensions: 0.007659782973102856
# Total Max Error across position dimensions: 0.12821158468723295


# rel_ik_sb3_ppo_ur5e_reach_0_05_pose_without_gripper_action_magnitude_0_01_step_16000
# Mean MSE across position dimensions: 0.0011074326104506124
# Mean MAE across position dimensions: 0.0140584918636806
# Total Max Error across position dimensions: 0.17446028292179122


# rel_ik_sb3_ppo_ur5e_reach_0_05_pose_without_gripper_action_rate_0_1_action_magnitude_0_01_step_16000
# Mean MSE across position dimensions: 0.0006070796839759171
# Mean MAE across position dimensions: 0.014414018859927796
# Total Max Error across position dimensions: 0.09233542978763573
# Smoothness (Sum of Squared Accelerations): 0.004963


# rel_ik_sb3_ppo_ur5e_reach_0_05_pose_without_gripper_action_rate_0_5_action_magnitude_0_02_step_16000
# Mean MSE across position dimensions: 0.00019258806844600545
# Mean MAE across position dimensions: 0.007439411668446925
# Total Max Error across position dimensions: 0.0579601466655732
# Smoothness (Sum of Squared Accelerations): 0.004246


# Define a threshold
threshold = 0.0005  # You can tune this!

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
