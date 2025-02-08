import pandas as pd
import plotly.graph_objects as go
import os

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

    # Save the figure as PDF and PNG
    pdf_path = os.path.join(output_dir, filename + ".pdf")
    png_path = os.path.join(output_dir, filename + ".png")

    fig.write_image(pdf_path, width=1100, height=600)  # Save as PDF
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

    # Save the figure as PDF and PNG
    pdf_path = os.path.join(output_dir, filename + ".pdf")
    png_path = os.path.join(output_dir, filename + ".png")

    fig.write_image(pdf_path, width=1000, height=600)  # Save as PDF
    fig.write_image(png_path, width=1000, height=600)  # Save as PNG

    fig.show()



# Generate and save plots
# create_and_save_plot(tcp_displacement, amount, "TCP Displacement", "tcp_displacement")
# create_and_save_plot(actions, amount, "Actions", "actions")
# create_and_save_plot(tcp_pose_error, amount, "TCP Position Error (TCP - Pose Command)", "tcp_position_error")

# create_and_save_comparison_plot(tcp_pose_error, actions, 1, "Comparison between TCP displacements and Actions", "comparison_tcp_displacement_and_action")
# create_and_save_comparison_plot(tcp_displacement, actions, 1, "Comparison between TCP displacements and Actions", "comparison_tcp_displacement_and_action")


create_and_save_plot(tcp_displacement, amount, "TCP Displacement", "tcp_displacement_without_obs_noise")
create_and_save_plot(actions, amount, "Actions", "actions_without_obs_noise")
create_and_save_plot(tcp_pose_error, amount, "TCP Position Error (TCP - Pose Command)", "tcp_position_error_without_obs_noise")

create_and_save_comparison_plot(tcp_pose_error, actions, 1, "Comparison between TCP displacements and Actions", "comparison_tcp_displacement_and_action_without_obs_noise")
create_and_save_comparison_plot(tcp_displacement, actions, 1, "Comparison between TCP displacements and Actions", "comparison_tcp_displacement_and_action_without_obs_noise")
