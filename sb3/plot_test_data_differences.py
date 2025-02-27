import pandas as pd
import plotly.graph_objects as go
import os
import numpy as np
from scipy.spatial.transform import Rotation as R

# Path to the CSV file
# csv_path = "/home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/data/observations_ur5e_with_unoise.csv"
# csv_path = "/home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/data/observations_1.csv"

# filename = "observations_rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e_decimation_4"
# filename = "observations_rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e_penalize_joint_vel"
# filename = "observations_rel_ik_sb3_ppo_ur5e_reach_0_1_pose_hand_e_penalize_joint_vel"
# filename = "observations_rel_ik_sb3_ppo_ur5e_reach_0_1_pose_hand_e_penalize_ee_acc"
# filename = "observations_rel_ik_sb3_ppo_ur5e_reach_0_1_pose_hand_e_penalize_ee_acc_v2"
# filename = "observations_rel_ik_sb3_ppo_ur5e_reach_0_1_pose_hand_e_penalize_ee_acc_v3"
# filename = "observations_rel_ik_sb3_ppo_ur5e_reach_0_1_pose_hand_e_penalize_ee_acc_v4"
# filename = "observations_rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e_stiffness_800000"
# filename = "observations_rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e_stiffness_10000000"
# filename = "observations_rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e_stiffness_10000000_v2"
# filename = "observations_rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e_stiffness_10000000_v3"
filename = "observations_rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e_stiffness_10000000_v4"


csv_path = "/home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/data/" + filename + ".csv"


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
actions = data[actions_cols] * 1.0 # 0.1 # Given in robot base frame


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
save = False # False True

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
create_and_save_comparison_plot(tcp_displacement, actions, 1, "Comparison between TCP Displacements and Actions - " + filename, "comparison_tcp_displacement_and_action_" + filename)

