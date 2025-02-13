import os
import csv
import rtde_control
import rtde_receive
import numpy as np
import torch
from stable_baselines3 import PPO
import random
from scipy.spatial.transform import Rotation as R

# Replace with the IP address of your robot
ROBOT_IP = "192.168.1.100"

print("start")
# Initialize RTDE Control and Receive Interfaces
rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
print("Connected to Control Interface")
rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
print("Connected to Receive Interface")

# Load the pre-trained model
checkpoint_path = "/home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/rel_ik_sb3_ppo_ur5e_reach_0_1_pose_hand_e_penalize_ee_acc_v2/j90cbcg2/model.zip"
print(f"Loading checkpoint from: {checkpoint_path}")
agent = PPO.load(checkpoint_path)

# Set CSV save directory
save_dir = "/home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/data/real_robot/"
csv_path = os.path.join(save_dir, "real_observations_3.csv")

save = True # False

def save_observations_to_csv(file_path, timestep, tcp_pose, target_pose, last_action):
    """Save TCP pose, target pose, and last action to a CSV file."""
    header = (
        ["timestep"]
        + [f"tcp_pose_{i}" for i in range(7)]
        + [f"target_pose_{i}" for i in range(7)]
        + [f"last_action_{i}" for i in range(6)]
    )

    # Flatten all observations into a single list
    data = [timestep] + tcp_pose.tolist() + target_pose.tolist() + last_action.tolist()

    file_exists = os.path.isfile(file_path)

    with open(file_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(header)  # Write header if file doesn't exist
        writer.writerow(data)


def move_robot_to_home():
    """Move robot to home pose specified with joint positions."""
    home_pose = np.array([1.3, -2.0, 2.0, -1.5, -1.5, 3.14])
    speed = 0.5
    acceleration = 0.5
    rtde_c.moveJ(home_pose, speed=speed, acceleration=acceleration)


# Random Pose Sampling Function
def sample_random_pose():
    """Randomly sample a target pose within specified bounds."""
    pos_x = 0.05
    pos_y = 0.3
    pos_z = 0.3
    roll = 0.0
    pitch = np.pi
    yaw = np.pi

    # Convert Roll-Pitch-Yaw to Quaternion [w, x, y, z] format
    r = R.from_euler("xyz", [roll, pitch, yaw], degrees=False)
    quat = r.as_quat()  # Default format is [x, y, z, w]

    # Reorder quaternion to [w, x, y, z] and normalize
    quat_wxyz = np.array([quat[3], quat[0], quat[1], quat[2]])
    quat_wxyz /= np.linalg.norm(quat_wxyz)

    return np.array([pos_x, pos_y, pos_z, *quat_wxyz])


# Returns the actual TCP received from the robot in [X, Y, Z] + Quaternion in [w, x, y, z]
def get_actual_TCP_Pose():
    tcp_pose = rtde_r.getActualTCPPose()  # Returns [X, Y, Z, RX, RY, RZ]

    # Extract position (X, Y, Z) and orientation (RX, RY, RZ)
    pos = np.array(tcp_pose[:3])  # [X, Y, Z]
    axis_angle = np.array(tcp_pose[3:])  # [RX, RY, RZ]

    # Convert Axis-Angle to Quaternion
    rot = R.from_rotvec(axis_angle)  # Convert to rotation object
    quat = rot.as_quat()  # Default output: [x, y, z, w]

    # Reorder quaternion to [w, x, y, z] and normalize
    quat_wxyz = np.array([quat[3], quat[0], quat[1], quat[2]])
    quat_wxyz /= np.linalg.norm(quat_wxyz)

    return np.concatenate((pos, quat_wxyz), axis=0)


# Function to get the robot's current state with quaternion in [w, x, y, z] format
def get_robot_state(target_pose, previous_action):
    tcp_pose = get_actual_TCP_Pose()
    return np.concatenate((tcp_pose, target_pose, previous_action), axis=0)


# Function to send actions to the robot
def execute_action_on_real_robot(action):
    """Send a TCP displacement action to the robot."""
    current_tcp = np.array(rtde_r.getActualTCPPose())  # [x, y, z, rx, ry, rz]
    
    # Apply positional displacements (dx, dy, dz)
    new_tcp = current_tcp.copy()
    new_tcp[:3] += action[:3]
    
    # Convert current orientation from axis-angle to a rotation matrix
    current_rotation = R.from_rotvec(current_tcp[3:])  # Axis-angle to rotation matrix
    
    # Compute the orientation displacement as a rotation matrix
    displacement_rotation = R.from_rotvec(np.array(action[3:]))  
    
    # Combine rotations: new_rotation = current_rotation * displacement_rotation
    new_rotation = current_rotation * displacement_rotation
    
    # Convert the new rotation matrix back to axis-angle representation
    new_tcp[3:] = new_rotation.as_rotvec()
    
    # Command the robot to move to the new TCP pose
    rtde_c.moveL(new_tcp.tolist(), speed=0.2, acceleration=0.5)


# Main control loop
def run_on_real_robot():
    move_robot_to_home()
    previous_action = np.zeros(6)
    target_pose = sample_random_pose()
    print(f"Target Pose: {target_pose}")

    timestep = 0  # Initialize timestep counter

    while True:
        tcp_pose = get_actual_TCP_Pose()
        obs = get_robot_state(target_pose, previous_action)

        with torch.inference_mode():
            action, _ = agent.predict(obs, deterministic=True)
            action *= 0.01

        if save:
            # Save the TCP pose, target pose, and last action to CSV
            save_observations_to_csv(csv_path, timestep, tcp_pose, target_pose, previous_action)

        previous_action = action  # Save the action
        execute_action_on_real_robot(action)  # Send the action to the robot

        # Check if target pose is reached
        current_tcp = get_actual_TCP_Pose()
        distance = np.linalg.norm(target_pose[:3] - current_tcp[:3])

        if distance < 0.005:  # Target reached
            print("Done")
            return

        timestep += 1  # Increment timestep counter


# Run the main function
if __name__ == "__main__":
    try:
        run_on_real_robot()
    except KeyboardInterrupt:
        print("Stopping robot execution.")
    finally:
        rtde_c.stopScript()
