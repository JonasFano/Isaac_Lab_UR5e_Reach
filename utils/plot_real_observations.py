# Mainly AI Generated

import plotly.graph_objects as go
import pandas as pd
import os
from scipy.spatial.transform import Rotation as R
import numpy as np

# file_dir = "/home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/data/real_robot/"
file_dir = "/home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/data/quaternion_analysis/"

output_dir = "/home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/plots/real_robot"
os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

# filename = 'real_observations_rotate_rx'

# Quaternion analysis
# filename = "domain_rand_model_predefined_poses_scale_0_05_seed_24_with_quat_consistency_clockwise"
filename = "domain_rand_model_predefined_poses_scale_0_05_seed_24_with_quat_consistency_counterclockwise"
# filename = "domain_rand_model_predefined_poses_scale_0_05_seed_24_without_quat_consistency_clockwise"

# Correct hemisphere means quat_wxyz *= -1 for the first observation and ensured quat consistency afterwards
# filename = "domain_rand_model_predefined_poses_scale_0_05_seed_24_with_quat_consistency_correct_hemisphere_counterclockwise"
# filename = "domain_rand_model_predefined_poses_scale_0_05_seed_24_with_quat_consistency_correct_hemisphere_clockwise" 

# filename = "domain_rand_model_predefined_poses_scale_0_05_seed_24_enforce_w_smaller_0_clockwise"
# filename = "domain_rand_model_predefined_poses_scale_0_05_seed_24_enforce_w_smaller_0_counterclockwise"


# Rotation matrix
# filename = "domain_rand_model_predefined_poses_scale_0_05_seed_24_rotmat_clockwise"
# filename = "domain_rand_model_predefined_poses_scale_0_05_seed_24_rotmat_counterclockwise"


save = False  # True False


# Read data from a CSV file
df = pd.read_csv(os.path.join(file_dir, filename + ".csv"))

max_timesteps = None

if max_timesteps is not None:
    df = df.iloc[:max_timesteps]

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


if save:
    png_path = os.path.join(output_dir, "tcp_and_target_position_" + filename + ".png")
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
    yaxis_title='Current and Target Quaternion',
    legend_title='Legend',
    hovermode="x unified",
    showlegend=False,
)


if True: #save:
    png_quaternion_path = os.path.join(output_dir, "tcp_and_target_quaternion_" + filename + ".pdf")
    fig_quaternion.write_image(png_quaternion_path, width=1000, height=600)

# Show the quaternion figure
fig_quaternion.show()



# Convert quaternion to axis-angle
def quaternion_to_axis_angle(quat_df):
    """ Convert quaternion columns to axis-angle representation """
    quaternions = quat_df.to_numpy()  # Convert to numpy array
    rotations = R.from_quat(quaternions, scalar_first=True)  # Convert to rotation object
    return rotations.as_rotvec()  # Convert to axis-angle (rotational vector)

def enforce_axis_angle_continuity(rotvecs):
    """Ensure smooth transitions in axis-angle representation across timesteps."""
    smoothed_rotvecs = np.copy(rotvecs)
    
    for i in range(1, len(rotvecs)):
        if np.dot(smoothed_rotvecs[i], smoothed_rotvecs[i - 1]) < 0:
            smoothed_rotvecs[i] *= -1  # Flip direction to maintain continuity
    
    return smoothed_rotvecs

# Extract quaternion columns for TCP and Target
tcp_quat = df[['tcp_pose_3', 'tcp_pose_4', 'tcp_pose_5', 'tcp_pose_6']]
target_quat = df[['target_pose_3', 'target_pose_4', 'target_pose_5', 'target_pose_6']]

# Convert to axis-angle
tcp_axis_angle = quaternion_to_axis_angle(tcp_quat)
target_axis_angle = quaternion_to_axis_angle(target_quat)

tcp_axis_angle = enforce_axis_angle_continuity(tcp_axis_angle)
target_axis_angle = enforce_axis_angle_continuity(target_axis_angle)

# Convert to DataFrame
tcp_axis_angle_df = pd.DataFrame(tcp_axis_angle, columns=['tcp_axis_x', 'tcp_axis_y', 'tcp_axis_z'])
target_axis_angle_df = pd.DataFrame(target_axis_angle, columns=['target_axis_x', 'target_axis_y', 'target_axis_z'])

# Merge with original dataframe for plotting
df = pd.concat([df, tcp_axis_angle_df, target_axis_angle_df], axis=1)

# Define colors for X, Y, Z
axis_angle_colors = {'x': 'red', 'y': 'green', 'z': 'blue'}

# Create a Plotly figure for Axis-Angle representation
fig_axis_angle = go.Figure()

# Add traces for TCP axis-angle components
fig_axis_angle.add_trace(go.Scatter(x=df['timestep'], y=df['tcp_axis_x'], mode='lines', name='TCP RX', line=dict(color=axis_angle_colors['x'])))
fig_axis_angle.add_trace(go.Scatter(x=df['timestep'], y=df['tcp_axis_y'], mode='lines', name='TCP RY', line=dict(color=axis_angle_colors['y'])))
fig_axis_angle.add_trace(go.Scatter(x=df['timestep'], y=df['tcp_axis_z'], mode='lines', name='TCP RZ', line=dict(color=axis_angle_colors['z'])))

# Add traces for Target axis-angle components with dashed lines
fig_axis_angle.add_trace(go.Scatter(x=df['timestep'], y=df['target_axis_x'], mode='lines', name='Target RX', line=dict(color=axis_angle_colors['x'], dash='dash')))
fig_axis_angle.add_trace(go.Scatter(x=df['timestep'], y=df['target_axis_y'], mode='lines', name='Target RY', line=dict(color=axis_angle_colors['y'], dash='dash')))
fig_axis_angle.add_trace(go.Scatter(x=df['timestep'], y=df['target_axis_z'], mode='lines', name='Target RZ', line=dict(color=axis_angle_colors['z'], dash='dash')))

# Customize layout
fig_axis_angle.update_layout(
    xaxis_title='Timestep',
    yaxis_title='Rotation (rad)',
    legend_title='Legend',
    hovermode="x unified",
)


if save:
    png_axis_angle_path = os.path.join(output_dir, "tcp_and_target_axis_angle_" + filename + ".png")
    fig_axis_angle.write_image(png_axis_angle_path, width=1000, height=600)

# Show the figure
fig_axis_angle.show()




# Create a Plotly figure for Actions
fig_actions = go.Figure()

# Define colors for action components
action_colors = {'x': 'red', 'y': 'green', 'z': 'blue', 'rx': 'purple', 'ry': 'orange', 'rz': 'brown'}

# Add traces for action components
fig_actions.add_trace(go.Scatter(x=df['timestep'], y=df['last_action_0'], mode='lines', name='Action X', line=dict(color=action_colors['x'])))
fig_actions.add_trace(go.Scatter(x=df['timestep'], y=df['last_action_1'], mode='lines', name='Action Y', line=dict(color=action_colors['y'])))
fig_actions.add_trace(go.Scatter(x=df['timestep'], y=df['last_action_2'], mode='lines', name='Action Z', line=dict(color=action_colors['z'])))
fig_actions.add_trace(go.Scatter(x=df['timestep'], y=df['last_action_3'], mode='lines', name='Action RX', line=dict(color=action_colors['rx'])))
fig_actions.add_trace(go.Scatter(x=df['timestep'], y=df['last_action_4'], mode='lines', name='Action RY', line=dict(color=action_colors['ry'])))
fig_actions.add_trace(go.Scatter(x=df['timestep'], y=df['last_action_5'], mode='lines', name='Action RZ', line=dict(color=action_colors['rz'])))

# Customize layout
fig_actions.update_layout(
    xaxis_title='Timestep',
    yaxis_title='Action (m and rad)',
    legend_title='Legend',
    hovermode="x unified",
)

# Save the plot if needed
if save:
    png_actions_path = os.path.join(output_dir, "actions_" + filename + ".png")
    fig_actions.write_image(png_actions_path, width=1000, height=600)

# Show the figure
fig_actions.show()






# tcp_angle = np.linalg.norm(tcp_axis_angle, axis=1)
# target_angle = np.linalg.norm(target_axis_angle, axis=1)

# # Create Plotly figure
# fig_angle = go.Figure()

# # Add traces for TCP and Target orientation angles
# fig_angle.add_trace(go.Scatter(x=df['timestep'], y=tcp_angle, mode='lines', name='TCP Orientation Angle', line=dict(color='blue')))
# fig_angle.add_trace(go.Scatter(x=df['timestep'], y=target_angle, mode='lines', name='Target Orientation Angle', line=dict(color='red', dash='dash')))

# # Customize layout
# fig_angle.update_layout(
#     xaxis_title='Timestep',
#     yaxis_title='Rotation Angle (radians)',
#     title='TCP vs Target Orientation Angle Over Time',
#     legend_title='Legend',
#     hovermode="x unified"
# )

# # Save the plot if needed
# if save:
#     png_angle_path = os.path.join(output_dir, "angle_" + filename + ".png")
#     fig_angle.write_image(png_angle_path, width=1000, height=600)

# # Show the figure
# fig_angle.show()










dot_products = np.einsum('ij,ij->i', tcp_quat, target_quat)  # Compute dot product row-wise
dot_products = np.clip(dot_products, -1.0, 1.0)  # Ensure values are within valid range

geodesic_distance = 2 * np.arccos(np.abs(dot_products))  # Compute geodesic distance

# Create Plotly figure for geodesic distance
fig_geodesic = go.Figure()

# Add trace for geodesic distance
fig_geodesic.add_trace(go.Scatter(x=df['timestep'], y=geodesic_distance, mode='lines',
                                  name='Geodesic Distance', line=dict(color='purple')))

# Customize layout
fig_geodesic.update_layout(
    xaxis_title='Timestep',
    yaxis_title='Geodesic Distance (radians)',
    legend_title='Legend',
    hovermode="x unified"
)

# Save the plot if needed
if save:
    png_geodesic_distance_path = os.path.join(output_dir, "geodesic_distance_" + filename + ".png")
    fig_geodesic.write_image(png_geodesic_distance_path, width=1000, height=600)

# Show the figure
fig_geodesic.show()



# Compute min, max, and differences for x, y, z
min_x, max_x = df['tcp_pose_0'].min(), df['tcp_pose_0'].max()
min_y, max_y = df['tcp_pose_1'].min(), df['tcp_pose_1'].max()
min_z, max_z = df['tcp_pose_2'].min(), df['tcp_pose_2'].max()

diff_x = max_x - min_x
diff_y = max_y - min_y
diff_z = max_z - min_z

print(f"x: min = {min_x:.6f}, max = {max_x:.6f}, diff = {diff_x:.6f}")
print(f"y: min = {min_y:.6f}, max = {max_y:.6f}, diff = {diff_y:.6f}")
print(f"z: min = {min_z:.6f}, max = {max_z:.6f}, diff = {diff_z:.6f}")

# Print highest difference
max_diff = max(diff_x, diff_y, diff_z)
print(f"\nHighest difference across diff = {max_diff:.6f}")

displacement = np.sqrt((df['tcp_pose_0'] - np.mean(df['tcp_pose_0']))**2 + 
                       (df['tcp_pose_1'] - np.mean(df['tcp_pose_1']))**2 + 
                       (df['tcp_pose_2'] - np.mean(df['tcp_pose_2']))**2)
print(f"Max TCP deviation: {displacement.max()}")
print(f"RMS TCP deviation: {np.sqrt((displacement**2).mean())}")









# Plot rotvec_action (axis-angle command sent to robot)
fig_rotvec = go.Figure()

# Define colours
rotvec_colors = {'rx': 'purple', 'ry': 'orange', 'rz': 'brown'}

# Add traces for rotation vector action
fig_rotvec.add_trace(go.Scatter(x=df['timestep'], y=df['rotvec_action_0'], mode='lines', name='RotVec RX', line=dict(color=rotvec_colors['rx'])))
fig_rotvec.add_trace(go.Scatter(x=df['timestep'], y=df['rotvec_action_1'], mode='lines', name='RotVec RY', line=dict(color=rotvec_colors['ry'])))
fig_rotvec.add_trace(go.Scatter(x=df['timestep'], y=df['rotvec_action_2'], mode='lines', name='RotVec RZ', line=dict(color=rotvec_colors['rz'])))

# Layout
fig_rotvec.update_layout(
    xaxis_title='Timestep',
    yaxis_title='Rotation Vector (rad)',
    legend_title='Legend',
    title='Final Rotation Vector Sent to Robot',
    hovermode="x unified"
)

# Save if needed
if save:
    rotvec_plot_path = os.path.join(output_dir, "rotvec_action_" + filename + ".png")
    fig_rotvec.write_image(rotvec_plot_path, width=1000, height=600)

# Show the plot
fig_rotvec.show()
