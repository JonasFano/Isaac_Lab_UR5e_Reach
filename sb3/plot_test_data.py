import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp
import os

# Path to the CSV file
csv_path = "/home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/data/observations_1.csv"

# Load the data
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"CSV file not found at {csv_path}")

data = pd.read_csv(csv_path)

# Round numerical values to 4 decimal places
data = data.round(4)

# Extract columns for plotting
timesteps = data["timestep"]
tcp_pose = data[[f"tcp_pose_{i}" for i in range(7)]]
pose_command = data[[f"pose_command_{i}" for i in range(7)]]
actions = data[[f"actions_{i}" for i in range(7)]]

# Create subplots
fig = sp.make_subplots(rows=3, cols=1, subplot_titles=("TCP Pose", "Actions", "Pose Command"))

amount = 3 # Only positions
# amount = 7 # All data (with quaternions)

# Add TCP Pose traces
for i in range(amount):
    fig.add_trace(go.Scatter(x=timesteps, y=tcp_pose[f"tcp_pose_{i}"], mode='lines', name=f"TCP Pose {i}"), row=1, col=1)

# Add Actions traces
for i in range(amount):
    fig.add_trace(go.Scatter(x=timesteps, y=actions[f"actions_{i}"], mode='lines', name=f"Actions {i}"), row=2, col=1)

# Add Pose Command traces
for i in range(amount):
    fig.add_trace(go.Scatter(x=timesteps, y=pose_command[f"pose_command_{i}"], mode='lines', name=f"Pose Command {i}"), row=3, col=1)

# Update layout
fig.update_layout(
    title="Observations Over Time",
    xaxis_title="Timestep",
    hovermode="x unified",
    height=900,
)

fig.show()
