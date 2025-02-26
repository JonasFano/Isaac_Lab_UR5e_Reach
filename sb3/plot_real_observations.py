import plotly.graph_objects as go
import pandas as pd
import os

# filepath = '/home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/data/real_robot/real_observations_pos_only.csv'
# filepath = '/home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/data/real_robot/real_observations.csv'

# filepath = '/home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/data/real_robot/real_observations_1.csv'
# filepath = '/home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/data/real_robot/real_observations_3.csv'

# filepath = '/home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/data/real_robot/real_observations_rotate_rx.csv'
# filepath = '/home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/data/real_robot/real_observations_rotate_rz.csv'



# filepath = '/home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/data/real_robot/real_observations_26_02_v1.csv'
# filepath = '/home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/data/real_robot/real_observations_26_02_v2.csv'
# filepath = '/home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/data/real_robot/real_observations_26_02_v3.csv'
# filepath = '/home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/data/real_robot/real_observations_26_02_v4.csv'
filepath = '/home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/data/real_robot/real_observations_predefined_pose.csv'




# Read data from a CSV file
df = pd.read_csv(filepath)

# Define colors for X, Y, Z
colors = {'x': 'red', 'y': 'green', 'z': 'blue'}

# Create a Plotly figure for TCP and Target Pose
fig = go.Figure()

# Add traces for TCP poses
fig.add_trace(go.Scatter(x=df['timestep'], y=df['tcp_pose_0'], mode='lines', name='TCP X', line=dict(color=colors['x'])))
fig.add_trace(go.Scatter(x=df['timestep'], y=df['tcp_pose_1'], mode='lines', name='TCP Y', line=dict(color=colors['y'])))
fig.add_trace(go.Scatter(x=df['timestep'], y=df['tcp_pose_2'], mode='lines', name='TCP Z', line=dict(color=colors['z'])))

# Add traces for Target poses with dashed lines
fig.add_trace(go.Scatter(x=df['timestep'], y=df['target_pose_0'], mode='lines', name='Target X', line=dict(color=colors['x'], dash='dash')))
fig.add_trace(go.Scatter(x=df['timestep'], y=df['target_pose_1'], mode='lines', name='Target Y', line=dict(color=colors['y'], dash='dash')))
fig.add_trace(go.Scatter(x=df['timestep'], y=df['target_pose_2'], mode='lines', name='Target Z', line=dict(color=colors['z'], dash='dash')))

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
save = False  # False # True

if save:
    png_path = os.path.join(output_dir, filename + ".png")
    fig.write_image(png_path, width=1000, height=600)

# Show the figure
fig.show()

# Create a Plotly figure for TCP and Target Orientation (Quaternion)
fig_quaternion = go.Figure()

# Define colors for Quaternion components
quat_colors = {'w': 'black', 'x': 'red', 'y': 'green', 'z': 'blue'}

# Add traces for TCP quaternion orientation
fig_quaternion.add_trace(go.Scatter(x=df['timestep'], y=df['tcp_pose_3'], mode='lines', name='TCP Quaternion W', line=dict(color=quat_colors['w'])))
fig_quaternion.add_trace(go.Scatter(x=df['timestep'], y=df['tcp_pose_4'], mode='lines', name='TCP Quaternion X', line=dict(color=quat_colors['x'])))
fig_quaternion.add_trace(go.Scatter(x=df['timestep'], y=df['tcp_pose_5'], mode='lines', name='TCP Quaternion Y', line=dict(color=quat_colors['y'])))
fig_quaternion.add_trace(go.Scatter(x=df['timestep'], y=df['tcp_pose_6'], mode='lines', name='TCP Quaternion Z', line=dict(color=quat_colors['z'])))

# Add traces for Target quaternion orientation with dashed lines
fig_quaternion.add_trace(go.Scatter(x=df['timestep'], y=df['target_pose_3'], mode='lines', name='Target Quaternion W', line=dict(color=quat_colors['w'], dash='dash')))
fig_quaternion.add_trace(go.Scatter(x=df['timestep'], y=df['target_pose_4'], mode='lines', name='Target Quaternion X', line=dict(color=quat_colors['x'], dash='dash')))
fig_quaternion.add_trace(go.Scatter(x=df['timestep'], y=df['target_pose_5'], mode='lines', name='Target Quaternion Y', line=dict(color=quat_colors['y'], dash='dash')))
fig_quaternion.add_trace(go.Scatter(x=df['timestep'], y=df['target_pose_6'], mode='lines', name='Target Quaternion Z', line=dict(color=quat_colors['z'], dash='dash')))

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
