import rtde_control
import rtde_receive
import numpy as np
import torch
from stable_baselines3 import PPO
import random
from scipy.spatial.transform import Rotation as R

# Replace with the IP address of your robot
ROBOT_IP = "192.168.0.100"

# Initialize RTDE Control and Receive Interfaces
rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)

# Load the pre-trained model
checkpoint_path = "/home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/logs/sb3/ppo/UR5e-Reach-Pose-IK/Quat_implementation/model.zip"
print(f"Loading checkpoint from: {checkpoint_path}")
agent = PPO.load(checkpoint_path)

def move_robot_to_home():
    """Move robot to home pose specified with joint positions"""
    home_pose = np.array((1.3, -2.0, 2.0, -1.5, -1.5, 3.14, 0.0, 0.0))
    speed = 0.5
    acceleration = 0.5
    rtde_c.moveJ(home_pose, speed=speed, acceleration=acceleration)


# Random Pose Sampling Function
def sample_random_pose():
    """Randomly sample a target pose within specified bounds."""
    pos_x = random.uniform(-0.2, 0.2)
    pos_y = random.uniform(0.35, 0.55)
    pos_z = random.uniform(0.15, 0.4)
    roll = 0.0
    pitch = np.pi  # End-effector pointing down
    yaw = random.uniform(-np.pi, np.pi)
    return [pos_x, pos_y, pos_z, roll, pitch, yaw]


# Function to get the robot's current state
def get_robot_state(target_pose, previous_actions):
    tcp_pose = rtde_r.getActualTCPPose()
    return np.concatenate((tcp_pose, target_pose, previous_actions[-1]), axis=0)


# Function to send actions to the robot
def send_robot_action(action):
    """
    Send a TCP displacement action to the robot.
    :param action: 6D displacement (dx, dy, dz, d_axis_angle_x, d_axis_angle_y, d_axis_angle_z).
    """
    # Get the current TCP pose (position + orientation in axis-angle)
    current_tcp = np.array(rtde_r.getActualTCPPose())  # [x, y, z, rx, ry, rz]
    
    # Apply positional displacements (dx, dy, dz)
    new_tcp = current_tcp.copy()
    new_tcp[:3] += action[:3]
    
    # Convert current orientation from axis-angle to a rotation matrix
    current_rotation = R.from_rotvec(current_tcp[3:])  # Axis-angle to rotation matrix
    
    # Compute the orientation displacement as a rotation matrix
    displacement_rotation = R.from_rotvec(action[3:])  # Axis-angle to rotation matrix
    
    # Combine rotations: new_rotation = current_rotation * displacement_rotation
    new_rotation = current_rotation * displacement_rotation
    
    # Convert the new rotation matrix back to axis-angle representation
    new_tcp[3:] = new_rotation.as_rotvec()
    
    # Command the robot to move to the new TCP pose
    rtde_c.moveL(new_tcp.tolist(), speed=0.1, acceleration=0.5)


# Main control loop
def run_on_real_robot():
    move_robot_to_home()
    previous_actions = np.zeros(6)
    target_pose = sample_random_pose()  # Generate a random target pose
    print(f"Target Pose: {target_pose}")

    while True:
        obs = get_robot_state(target_pose, previous_actions)  # Initial observation

        with torch.inference_mode():
            # Get the action from the agent
            action, _ = agent.predict(obs, deterministic=True)

        previous_actions = action  # Save the action
        send_robot_action(action)  # Send the action to the robot

        # Check if target pose is reached (simple distance threshold for now)
        current_tcp = rtde_r.getActualTCPPose()
        distance = np.linalg.norm(np.array(target_pose[:3]) - np.array(current_tcp[:3]))
        if distance < 0.02:  # Target reached within a 2 cm threshold
            move_robot_to_home()
            print("Target pose reached. Sampling new pose.")
            target_pose = sample_random_pose()  # Sample a new pose
            print(f"New Target Pose: {target_pose}")


# Run the main function
if __name__ == "__main__":
    try:
        run_on_real_robot()
    except KeyboardInterrupt:
        print("Stopping robot execution.")
    finally:
        rtde_c.stopScript()
