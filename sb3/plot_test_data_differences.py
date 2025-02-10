import pandas as pd
import plotly.graph_objects as go
import os
import numpy as np
from scipy.spatial.transform import Rotation as R

# Path to the CSV file
csv_path = "/home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/data/observations_2.csv"

# Output directory for saving the plots
output_dir = "/home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/plots"
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
actions = data[actions_cols] * 0.05



# Function to transform action from TCP frame to base frame
def transform_action_to_base_frame(action_tcp, tcp_pose):
    """
    Transforms an action from the TCP frame to the base frame using the TCP pose.
    
    Args:
        action_tcp (np.array): Action command in the TCP frame (6D: position + orientation).
        tcp_pose (np.array): TCP pose in the base frame (7D: position [x, y, z] + quaternion [w, x, y, z]).
    
    Returns:
        np.array: Transformed action in the base frame.
    """
    # Extract position (first 3 elements) and quaternion (last 4 elements) from tcp_pose
    tcp_quaternion_wxyz = np.array(tcp_pose[-4:])  # [w, x, y, z]

    # Convert (w, x, y, z) to (x, y, z, w) for SciPy
    tcp_quaternion_xyzw = np.roll(tcp_quaternion_wxyz, -1)

    # Convert quaternion to rotation matrix (R_base_tcp)
    R_base_tcp = R.from_quat(tcp_quaternion_xyzw).as_matrix()

    # Extract action components (assume first 3 elements are position deltas, next 3 are rotation)
    action_position_tcp = np.array(action_tcp[:3])  # Δx, Δy, Δz in TCP frame
    action_rotation_tcp = np.array(action_tcp[3:6])  # Fix: Only take the first 3 rotation components

    # Transform position and rotational action to base frame
    action_position_base = R_base_tcp @ action_position_tcp
    action_rotation_base = R_base_tcp @ action_rotation_tcp

    # Concatenate transformed position and rotation actions
    return np.concatenate([action_position_base, action_rotation_base])

# Transform actions to base frame
transformed_actions = np.array([
    transform_action_to_base_frame(actions.iloc[i].values, tcp_pose.iloc[i].values)
    for i in range(len(actions))
])

# Ensure column names match the transformed shape
transformed_columns = [col.replace("actions", "actions_base_frame") for col in actions.columns[:6]]

# Convert transformed actions back to DataFrame
transformed_actions_df = pd.DataFrame(transformed_actions, columns=transformed_columns)



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

    # # Save the figure as PDF and PNG
    # pdf_path = os.path.join(output_dir, filename + ".pdf")
    # png_path = os.path.join(output_dir, filename + ".png")

    # fig.write_image(pdf_path, width=1100, height=600)  # Save as PDF
    # fig.write_image(png_path, width=1100, height=600)  # Save as PNG

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

    # # Save the figure as PDF and PNG
    # pdf_path = os.path.join(output_dir, filename + ".pdf")
    # png_path = os.path.join(output_dir, filename + ".png")

    # fig.write_image(pdf_path, width=1000, height=600)  # Save as PDF
    # fig.write_image(png_path, width=1000, height=600)  # Save as PNG

    fig.show()



# Generate and save plots
# create_and_save_plot(tcp_displacement, amount, "TCP Displacement", "tcp_displacement")
# create_and_save_plot(actions, amount, "Actions", "actions")
# create_and_save_plot(tcp_pose_error, amount, "TCP Position Error (TCP - Pose Command)", "tcp_position_error")

# create_and_save_comparison_plot(tcp_pose_error, actions, 1, "Comparison between TCP displacements and Actions", "comparison_tcp_displacement_and_action")
# create_and_save_comparison_plot(tcp_displacement, actions, 1, "Comparison between TCP displacements and Actions", "comparison_tcp_displacement_and_action")


create_and_save_plot(tcp_displacement, amount, "TCP Displacement", "tcp_displacement_without_obs_noise")
create_and_save_plot(actions, amount, "Actions", "actions_without_obs_noise")
create_and_save_plot(transformed_actions_df, amount, "Actions in Base Frame", "actions_in_base_frame_without_obs_noise")
create_and_save_plot(tcp_pose_error, amount, "TCP Position Error (TCP - Pose Command)", "tcp_position_error_without_obs_noise")

create_and_save_comparison_plot(tcp_pose_error, actions, 1, "Comparison between TCP displacements and Actions", "comparison_tcp_displacement_and_action_without_obs_noise")
create_and_save_comparison_plot(tcp_displacement, actions, 1, "Comparison between TCP displacements and Actions", "comparison_tcp_displacement_and_action_without_obs_noise")

create_and_save_comparison_plot(tcp_pose_error, transformed_actions_df, 1, "Comparison between TCP displacements and Actions in Base Frame", "comparison_tcp_displacement_and_action_without_obs_noise")
create_and_save_comparison_plot(tcp_displacement, transformed_actions_df, 1, "Comparison between TCP displacements and Actions in Base Frame", "comparison_tcp_displacement_and_action_without_obs_noise")
