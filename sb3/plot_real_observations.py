import plotly.graph_objects as go
import pandas as pd
import os

# filepath = '/home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/data/real_robot/real_observations_pos_only.csv'
# filepath = '/home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/data/real_robot/real_observations.csv'

# filepath = '/home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/data/real_robot/real_observations_1.csv'
# filepath = '/home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/data/real_robot/real_observations_3.csv'

filepath = '/home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/data/real_robot/real_observations_rotate_rx.csv'
# filepath = '/home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/data/real_robot/real_observations_rotate_rz.csv'


# Read data from a CSV file
df = pd.read_csv(filepath)

# Create a Plotly figure for TCP and Target Pose
fig = go.Figure()

# Add traces for TCP poses
fig.add_trace(go.Scatter(x=df['timestep'], y=df['tcp_pose_0'], mode='lines', name='TCP X'))
fig.add_trace(go.Scatter(x=df['timestep'], y=df['tcp_pose_1'], mode='lines', name='TCP Y'))
fig.add_trace(go.Scatter(x=df['timestep'], y=df['tcp_pose_2'], mode='lines', name='TCP Z'))

# Add traces for Target poses
fig.add_trace(go.Scatter(x=df['timestep'], y=df['target_pose_0'], mode='lines', name='Target X', line=dict(dash='dash')))
fig.add_trace(go.Scatter(x=df['timestep'], y=df['target_pose_1'], mode='lines', name='Target Y', line=dict(dash='dash')))
fig.add_trace(go.Scatter(x=df['timestep'], y=df['target_pose_2'], mode='lines', name='Target Z', line=dict(dash='dash')))

# Customize the layout
fig.update_layout(
    xaxis_title='Timestep',
    yaxis_title='Position (m)',
    legend_title='Legend',
    hovermode="x unified",
)

output_dir = "/home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/plots/real_robot"
os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
filename = "tcp_and_target_pose_over_time_real_obs_rotate_rx"
save = True  # False # True

if save:
    png_path = os.path.join(output_dir, filename + ".png")
    fig.write_image(png_path, width=1000, height=600)

# Show the figure
fig.show()

# Create a Plotly figure for TCP and Target Orientation (Quaternion)
fig_quaternion = go.Figure()

# Add traces for TCP quaternion orientation
fig_quaternion.add_trace(go.Scatter(x=df['timestep'], y=df['tcp_pose_3'], mode='lines', name='TCP Quaternion W'))
fig_quaternion.add_trace(go.Scatter(x=df['timestep'], y=df['tcp_pose_4'], mode='lines', name='TCP Quaternion X'))
fig_quaternion.add_trace(go.Scatter(x=df['timestep'], y=df['tcp_pose_5'], mode='lines', name='TCP Quaternion Y'))
fig_quaternion.add_trace(go.Scatter(x=df['timestep'], y=df['tcp_pose_6'], mode='lines', name='TCP Quaternion Z'))

# Add traces for Target quaternion orientation
fig_quaternion.add_trace(go.Scatter(x=df['timestep'], y=df['target_pose_3'], mode='lines', name='Target Quaternion W', line=dict(dash='dash')))
fig_quaternion.add_trace(go.Scatter(x=df['timestep'], y=df['target_pose_4'], mode='lines', name='Target Quaternion X', line=dict(dash='dash')))
fig_quaternion.add_trace(go.Scatter(x=df['timestep'], y=df['target_pose_5'], mode='lines', name='Target Quaternion Y', line=dict(dash='dash')))
fig_quaternion.add_trace(go.Scatter(x=df['timestep'], y=df['target_pose_6'], mode='lines', name='Target Quaternion Z', line=dict(dash='dash')))

# Customize the layout for quaternion plot
fig_quaternion.update_layout(
    xaxis_title='Timestep',
    yaxis_title='Orientation',
    legend_title='Legend',
    hovermode="x unified",
)

if save:
    png_quaternion_path = os.path.join(output_dir, filename + "_quaternion.png")
    fig_quaternion.write_image(png_quaternion_path, width=1000, height=600)

# Show the quaternion figure
fig_quaternion.show()
