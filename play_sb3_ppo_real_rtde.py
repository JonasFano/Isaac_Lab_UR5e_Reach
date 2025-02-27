import os
import csv
import rtde_control
import rtde_receive
import numpy as np
import torch
from stable_baselines3 import PPO
import random
import time
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
# checkpoint_path = "/home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/rel_ik_sb3_ppo_ur5e_reach_0_1_pose_hand_e_penalize_ee_acc_v2/j90cbcg2/model.zip"
# checkpoint_path = "/home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e_stiffness_800000/apyblhie/model.zip"
checkpoint_path = "/home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e_stiffness_10000000_v2/su6rb80q/model.zip"
# checkpoint_path = "/home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e_stiffness_10000000_v3/w10lhrjg/model.zip"

print(f"Loading checkpoint from: {checkpoint_path}")
agent = PPO.load(checkpoint_path)

# Set CSV save directory
save_dir = "/home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/data/real_robot/"
csv_path = os.path.join(save_dir, "real_observations_predefined_pose_rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e_stiffness_10000000_v2_decaying_scale_random_poses_v2.csv")

save = False # True # False

# 180-degree rotation around Z-axis
ROT_180_Z = R.from_euler('z', 180, degrees=True)

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
    home_pose = np.array([1.3, -2.0, 2.0, -1.5, -1.5, 3.14]) # 0.0]) # 3.14])
    speed = 0.5
    acceleration = 0.5
    rtde_c.moveJ(home_pose, speed=speed, acceleration=acceleration)


# Random Pose Sampling Function
def sample_random_pose():
    """Randomly sample a target pose within specified bounds."""
    # pos_x = 0.05
    # pos_y = -0.3
    # pos_z = 0.3
    # roll = 0.0
    # pitch = np.pi
    # yaw = np.pi

    # pos_x = -0.25
    # pos_y = -0.3
    # pos_z = 0.2
    pos_x = random.uniform(-0.2, 0.2)
    pos_y = random.uniform(-0.4, -0.3)
    # pos_y = random.uniform(0.3, 0.4)
    pos_z = random.uniform(0.2, 0.5)
    roll = 0.0
    pitch = np.pi  # End-effector pointing down
    yaw = random.uniform(-np.pi/2, np.pi/2) # 90 degrees
    
    # pos_x = -0.2
    # pos_y = 0.4
    # pos_z = 0.3
    # roll = 0.0
    # pitch = np.pi
    # yaw = np.pi

    # Convert Roll-Pitch-Yaw to Quaternion [w, x, y, z] format
    r = R.from_euler("xyz", [roll, pitch, yaw], degrees=False)
    quat_wxyz = r.as_quat(scalar_first=True)  # Default format is [x, y, z, w]

    pos = [pos_x, pos_y, pos_z]  # [X, Y, Z]

    rotated_quat = R.from_quat(quat_wxyz, scalar_first=True) * ROT_180_Z
    rotated_quat_wxyz = rotated_quat.as_quat(scalar_first=True)

    # Apply 180-degree rotation to the position
    rotated_pos = ROT_180_Z.apply(pos)

    rotated_quat_wxyz /= np.linalg.norm(rotated_quat_wxyz)

    return np.concatenate((rotated_pos, rotated_quat_wxyz), axis=0)


def sample_predefined_pose(current_index):
    predefined_poses = np.array([
        [0.2, -0.4, 0.3, 0.0, 1.0, 0.0, 0.0],
        [0.15, -0.25, 0.1, 0.0, 1.0, 0.0, 0.0],  # Pose 1
        [-0.1,  -0.5,  0.4, 0.0, 1.0, 0.0, 0.0],  # Pose 2
        [-0.1, 0.35, 0.2, 0.0, 1.0, 0.0, 0.0],  # Pose 3
    ])
    target_pose = predefined_poses[current_index]

    pos = target_pose[:3]  # [X, Y, Z]
    quat_wxyz = target_pose[3:]  # Quat in [W X Y Z]

    rotated_quat = R.from_quat(quat_wxyz, scalar_first=True) * ROT_180_Z
    rotated_quat_wxyz = rotated_quat.as_quat(scalar_first=True)

    # Apply 180-degree rotation to the position
    rotated_pos = ROT_180_Z.apply(pos)

    return np.concatenate((rotated_pos, rotated_quat_wxyz), axis=0)



def normalize_axis_angle(axis_angle, prev_axis_angle):
    """Ensure a stable axis-angle representation by enforcing a consistent sign convention."""
    norm = np.linalg.norm(axis_angle)

    # If rotation is near 180 degrees, ensure sign consistency
    if norm > np.pi:
        axis_angle *= -1  # Flip to maintain consistency

    # Ensure continuity with previous frames to prevent flipping
    if prev_axis_angle is not None:
        if np.dot(prev_axis_angle, axis_angle) < 0:
            axis_angle *= -1  # Flip to maintain smooth transitions

    prev_axis_angle = axis_angle.copy()  # Store for next iteration
    return axis_angle, prev_axis_angle


def axis_angle_to_quaternion(axis_angle, prev_quaternion):
    """
    Convert a single axis-angle representation to a quaternion (w, x, y, z),
    ensuring a consistent sign convention to prevent flipping.
    """
    # Convert axis-angle to quaternion
    quat_wxyz = R.from_rotvec(axis_angle).as_quat(scalar_first=True)[0]  # Output format: [w, x, y, z]

    # Ensure quaternion continuity (prevent sign flips based on vector part)
    if prev_quaternion is not None:
        if np.dot(quat_wxyz[1:], prev_quaternion[1:]) < 0:  # Use only vector part [x, y, z]
            quat_wxyz *= -1  # Flip quaternion to maintain consistency
    # else:
    #     quat_wxyz *= -1

    prev_quaternion = quat_wxyz.copy()  # Store for next iteration

    return quat_wxyz, prev_quaternion


# Returns the actual TCP received from the robot in [X, Y, Z] + Quaternion in [w, x, y, z]
# and applies a 180-degree Z-axis rotation
def get_actual_TCP_Pose(prev_axis_angle, prev_quaternion):
    tcp_pose = np.array(rtde_r.getActualTCPPose())  # [x, y, z, rx, ry, rz]
    
    # Ensure axis-angle formatting before converting to quaternion
    # tcp_pose[3:], prev_axis_angle = normalize_axis_angle(tcp_pose[3:], prev_axis_angle)  

    pos = tcp_pose[:3]  # [X, Y, Z]
    axis_angle = [tcp_pose[3:]]  # Convert single axis-angle to list for processing

    # Convert Axis-Angle to Quaternion with proper formatting
    quat_wxyz, prev_quaternion = axis_angle_to_quaternion(axis_angle, prev_quaternion)  # Extract single quaternion

    # Apply 180-degree rotation around Z-axis
    rotated_quat = R.from_quat(quat_wxyz, scalar_first=True) * ROT_180_Z
    rotated_quat_wxyz = rotated_quat.as_quat(scalar_first=True)

    # Apply 180-degree rotation to the position
    rotated_pos = ROT_180_Z.apply(pos)

    return np.concatenate((rotated_pos, rotated_quat_wxyz), axis=0), prev_axis_angle, prev_quaternion




# Function to get the robot's current state with quaternion in [w, x, y, z] format

def get_robot_state(tcp_pose, target_pose, previous_action):
    return np.concatenate((tcp_pose, target_pose, previous_action), axis=0)


# Function to send actions to the robot
def execute_action_on_real_robot(action, tcp_pose):
    """Send a TCP displacement action to the robot, ensuring it is rotated back 180 degrees before execution."""
    
    # print(tcp_pose)
    # Convert input tcp_pose (3D pos + 4D quaternion) to axis-angle format
    current_tcp = np.array(tcp_pose[:3])  # Extract position
    current_rotation = R.from_quat(tcp_pose[3:], scalar_first=True)  # Convert quaternion to rotation object
    
    corrected_tcp = np.zeros(6)
    corrected_tcp[:3] = current_tcp  # Keep position the same
    corrected_tcp[3:] = current_rotation.as_rotvec()  # Convert to axis-angle
    
    # Compute new pose based on action
    new_tcp = corrected_tcp.copy()
    new_tcp[:3] += action[:3]  # Apply displacement in the rotated frame
    
    displacement_rotation = R.from_rotvec(action[3:])
    new_rotation = current_rotation * displacement_rotation
    
    new_tcp[3:] = new_rotation.as_rotvec()
    
    # Rotate back 180 degrees around Z-axis for both position and orientation before sending to the robot
    final_rotation = ROT_180_Z.inv() * R.from_rotvec(new_tcp[3:])
    new_tcp[3:] = final_rotation.as_rotvec()
    new_tcp[:3] = ROT_180_Z.inv().apply(new_tcp[:3])  # Rotate position back
    
    rtde_c.moveL(new_tcp.tolist(), speed=0.2, acceleration=0.5)


# Function to compute quaternion distance
def quaternion_distance(q1, q2):
    """ Compute the quaternion similarity (distance metric) """
    return 1 - abs(np.dot(q1, q2))  # Ensure shortest path



# Main control loop
def run_on_real_robot():
    move_robot_to_home()
    previous_action = np.zeros(6)
    prev_axis_angle = None  # Store the previous axis-angle value
    prev_quaternion = None
    current_index = 0

    target_pose = sample_random_pose()
    # target_pose = sample_predefined_pose(current_index)
    # current_index += 1
    print(f"Target Pose: {target_pose}")

    timestep = 0  # Initialize timestep counter
    decay_timestep = 0

    while True:
        tcp_pose, prev_axis_angle, prev_quaternion = get_actual_TCP_Pose(prev_axis_angle, prev_quaternion)
        obs = get_robot_state(tcp_pose, target_pose, previous_action)

        # print("Current TCP Pose: ", tcp_pose)
        # print("Target Pose: ", target_pose)

        with torch.inference_mode():
            action, _ = agent.predict(obs, deterministic=True)
            # action *= 0.002

            tunable_decay_steps = 10
            action_scaling = max(0.00001, 0.025 * max(0, (tunable_decay_steps - decay_timestep)) / tunable_decay_steps)
            action *= action_scaling
            print(action)


        if save:
            # Save the TCP pose, target pose, and last action to CSV
            save_observations_to_csv(csv_path, timestep, tcp_pose, target_pose, previous_action)

        previous_action = action  # Save the action
        execute_action_on_real_robot(action, tcp_pose)  # Send the action to the robot

        # Check if target position is reached
        position_distance = np.linalg.norm(target_pose[:3] - tcp_pose[:3])

        # Check if target orientation is reached
        current_quat = tcp_pose[3:]  # (w, x, y, z)
        target_quat = target_pose[3:]  # (w, x, y, z)
        orientation_distance = quaternion_distance(current_quat, target_quat)

        print(f"Position Error: {position_distance}, Orientation Error: {orientation_distance}")

        if position_distance < 0.04 and orientation_distance < 1.0:  # Adjust threshold if needed
            print("Target reached!")
            print("Final TCP Pose: ", tcp_pose)

            target_pose = sample_random_pose()
            decay_timestep = -1
            # target_pose = sample_predefined_pose(current_index)
            # current_index += 1

        decay_timestep += 1
        timestep += 1  # Increment timestep counter


# Run the main function
if __name__ == "__main__":
    try:
        run_on_real_robot()
    except KeyboardInterrupt:
        print("Stopping robot execution.")
    finally:
        rtde_c.stopScript()
