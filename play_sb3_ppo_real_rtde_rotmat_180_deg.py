import os
import csv
import random
import numpy as np
import torch
from stable_baselines3 import PPO
import rtde_control
import rtde_receive

from isaaclab.utils.math import quat_apply, quat_mul, quat_from_angle_axis, quat_from_euler_xyz, quat_inv, axis_angle_from_quat, apply_delta_pose

class UR5eRobotController:
    """
    Class to control a UR5e robot using RTDE interface and a pre-trained PPO model.
    """

    def __init__(self, robot_ip, model_path, action_scaling, save_dir, filename, mode, save=False):
        # Initialize RTDE Control and Receive Interfaces
        self.robot_ip = robot_ip
        self.rtde_c = rtde_control.RTDEControlInterface(robot_ip)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)
        print("Connected to Interfaces")

        # Load the pre-trained model
        print(f"Loading checkpoint from: {model_path}")
        self.agent = PPO.load(model_path)

        # Set CSV save directory
        self.save_dir = save_dir
        self.csv_path = os.path.join(save_dir, filename)

        self.save = save
        self.action_scaling = action_scaling
        self.mode = mode
        self.current_index = 0
        self.rotvec_action = np.zeros(3)
        self.previous_action = np.zeros(6)

        # 180-degree rotation around Z-axis
        self.rot_180_z = quat_from_euler_xyz(
            torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([np.pi])
        )[0]
        # Define inv 180° Z rotation 
        self.rot_180_z_inv = quat_inv(self.rot_180_z)


    def save_observations_to_csv(self, total_timestep):
        """Save TCP pose, target pose, and last action to a CSV file."""
        tcp_pose = np.concatenate((self.tcp_pose[:3], self.quat), axis=0)
        target_pose = np.concatenate((self.target_pose[:3], self.target_quat), axis=0)
        header = (
            ["timestep"]
            + [f"tcp_pose_{i}" for i in range(7)]
            + [f"target_pose_{i}" for i in range(7)]
            + [f"last_action_{i}" for i in range(6)]
            + [f"rotvec_action_{i}" for i in range(3)]
        )
        data = (
            [total_timestep]
            + tcp_pose.tolist()
            + target_pose.tolist()
            + self.previous_action.tolist()
            + self.rotvec_action.tolist()
        )

        file_exists = os.path.isfile(self.csv_path)

        with open(self.csv_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(header)  # Write header if file doesn't exist
            writer.writerow(data)


    def quat_to_rotmat(self, q):
        # q: [batch, 4] in (w, x, y, z) format
        # if q.ndim == 1:
        #     q = q.unsqueeze(0)

        q = q / q.norm(dim=1, keepdim=True)

        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        B = q.shape[0]
        R = torch.empty(B, 3, 3, device=q.device)
        R[:, 0, 0] = 1 - 2*y**2 - 2*z**2
        R[:, 0, 1] = 2*x*y - 2*z*w
        R[:, 0, 2] = 2*x*z + 2*y*w

        R[:, 1, 0] = 2*x*y + 2*z*w
        R[:, 1, 1] = 1 - 2*x**2 - 2*z**2
        R[:, 1, 2] = 2*y*z - 2*x*w

        R[:, 2, 0] = 2*x*z - 2*y*w
        R[:, 2, 1] = 2*y*z + 2*x*w
        R[:, 2, 2] = 1 - 2*x**2 - 2*y**2
        return R



    def move_robot_to_home(self):
        """Move robot to home pose specified with joint positions."""
        # Home Pose: [0.05291, -0.33409, 0.442, 0.417, 3.032, 0.093]
        home_pose = np.array([1.3, -2.0, 2.0, -1.5, -1.5, 0.0]) # 3.14])
        speed = 1.5
        acceleration = 1.5
        self.rtde_c.moveJ(home_pose, speed=speed, acceleration=acceleration)


    def sample_random_pose(self):
        """Randomly sample a target pose within specified bounds."""
        pos_x = random.uniform(-0.2, 0.2)
        pos_y = random.uniform(-0.25, -0.5)
        pos_z = random.uniform(0.2, 0.5)
        roll = 0.0
        pitch = np.pi  # End-effector z-axis pointing down (180 deg rotation)
        yaw = random.uniform(-np.pi, np.pi) # For wrist_3_joint = 0.0

        roll = torch.tensor(roll)
        pitch = torch.tensor(pitch)
        yaw = torch.tensor(yaw)

        # Convert Roll-Pitch-Yaw to Quaternion [w, x, y, z] format
        quat_wxyz = quat_from_euler_xyz(roll, pitch, yaw)
        pos = torch.tensor([pos_x, pos_y, pos_z])

        # Rotate position and orientation
        pos = quat_apply(self.rot_180_z_inv, pos)
        self.quat = quat_mul(self.rot_180_z_inv, quat_wxyz)

        rotmat = self.quat_to_rotmat(quat_wxyz.unsqueeze(0))
        rot = rotmat[:, :, :2].reshape(-1, 6)  # take first two columns and flatten

        self.target_quat = quat_wxyz.squeeze(0).numpy()
        self.target_pose = np.concatenate((pos.squeeze(0).numpy(), rot.squeeze(0).numpy()), axis=0)


    def get_predefined_pose(self):
        """Get a target pose from a predefined array of poses"""
        # predefined_poses = torch.tensor([
        #     [0.2, 0.4, 0.3, 0.0, 0.0, 1.0, 0.0],
        #     [0.15, 0.3, 0.15, 0.0, 0.0, 1.0, 0.0],
        #     [-0.1,  0.5,  0.4, 0.0, 0.0, 1.0, 0.0],
        # ])

        # Sim poses:
        predefined_poses = torch.tensor([
            [-0.2, 0.4, 0.3, 0.0000146, 0.3153224, 0.9489846, -0.000044], # -2.5 yaw angle
            [-0.2, 0.4, 0.3, 0.0000033, 0.070737, 0.997495, -0.0000462], # -3 yaw angle
            [-0.2, 0.4, 0.3, -0.0000033, -0.070737, 0.997495, -0.0000462], # 3 yaw angle
            [-0.2, 0.4, 0.3, -0.0000146, -0.3153224, 0.9489846, -0.000044], # 2.5 yaw angle
        ])

        # predefined_poses = torch.tensor([
        #     [-0.2, 0.4, 0.3, -0.0000146, -0.3153224, 0.9489846, -0.000044], # 2.5 yaw angle
        #     [-0.2, 0.4, 0.3, -0.0000033, -0.070737, 0.997495, -0.0000462], # 3 yaw angle
        #     [-0.2, 0.4, 0.3, 0.0000033, 0.070737, 0.997495, -0.0000462], # -3 yaw angle
        #     [-0.2, 0.4, 0.3, 0.0000146, 0.3153224, 0.9489846, -0.000044], # -2.5 yaw angle
        # ])
        
        # Split position and quaternion
        positions = predefined_poses[:, :3]
        quats = predefined_poses[:, 3:]

        # Normalize quaternions to unit length
        quats = quats / quats.norm(dim=1, keepdim=True)

        rotmats = self.quat_to_rotmat(quats)
        rots = rotmats[:, :, :2].reshape(-1, 6)  # take first two columns and flatten

        # Recombine into the full tensor
        predefined_poses = torch.cat([positions, rots], dim=1)

        self.target_quat = quats[self.current_index]
        self.target_pose = predefined_poses[self.current_index]


    def axis_angle_to_quaternion(self, axis_angle):
        """
        Convert a single axis-angle representation to a quaternion (w, x, y, z),
        ensuring a consistent sign convention to prevent flipping.
        """
        if not isinstance(axis_angle, torch.Tensor):
            axis_angle = torch.tensor(axis_angle, dtype=torch.float32)

        angle = torch.norm(axis_angle)
        if angle < 1e-6:
            # Identity rotation
            quat_wxyz = torch.tensor([1.0, 0.0, 0.0, 0.0], device=axis_angle.device)
        else:
            axis = axis_angle / angle
            quat_wxyz = quat_from_angle_axis(torch.tensor([angle], device=axis_angle.device), axis.unsqueeze(0))[0]  # (4,)

        return quat_wxyz
    

    def axis_angle_to_rotmat(self, axis_angle):
        """
        Convert axis-angle vectors to rotation matrices.
        axis_angle: [B, 3] tensor, each row is axis * angle (in radians)
        Returns: [B, 3, 3] rotation matrices
        """
        if isinstance(axis_angle, np.ndarray):
            axis_angle = torch.tensor(axis_angle, dtype=torch.float32)

        if axis_angle.ndim == 1:
            axis_angle = axis_angle.unsqueeze(0)  # ensure batch shape [1, 3]

        B = axis_angle.shape[0]
        theta = torch.norm(axis_angle, dim=1, keepdim=True)  # [B, 1]
        axis = axis_angle / (theta + 1e-8)  # normalised axis, [B, 3]

        # Components for Rodrigues' formula
        x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
        zero = torch.zeros_like(x)
        K = torch.stack([
            torch.stack([zero, -z, y], dim=1),
            torch.stack([z, zero, -x], dim=1),
            torch.stack([-y, x, zero], dim=1)
        ], dim=1)  # shape [B, 3, 3]

        I = torch.eye(3, device=axis_angle.device).expand(B, -1, -1)
        theta = theta.view(B, 1, 1)

        R = I + torch.sin(theta) * K + (1 - torch.cos(theta)) * K @ K
        return R

    
    def get_actual_tcp_pose(self):
        """
        Returns the actual TCP received from the robot in [X, Y, Z] + Quaternion in [w, x, y, z]
        and applies a 180-degree Z-axis rotation.
        """
        tcp_pose = np.array(self.rtde_r.getActualTCPPose())  # [x, y, z, rx, ry, rz]
        pos = torch.tensor(tcp_pose[:3], dtype=torch.float32)  
        axis_angle = tcp_pose[3:]

        # Convert Axis-Angle to Quaternion with proper formatting
        self.quat = self.axis_angle_to_quaternion(axis_angle)

        # Rotate position and orientation
        pos = quat_apply(self.rot_180_z_inv, pos)
        self.quat = quat_mul(self.rot_180_z_inv, self.quat)

        rotmat = self.quat_to_rotmat(self.quat.unsqueeze(0))
        rot = rotmat[:, :, :2].reshape(-1, 6)  # take first two columns and flatten
        self.tcp_pose = np.concatenate((pos.numpy(), rot.squeeze(0).numpy()), axis=0)


    def execute_action_on_real_robot(self):
        """Send a TCP displacement action to the robot, ensuring it is rotated back 180 degrees before execution."""
        # Convert inputs to torch
        action = torch.tensor(self.action, dtype=torch.float32).unsqueeze(0)  # shape [1, 6]
        tcp_pos = torch.tensor(self.tcp_pose[:3], dtype=torch.float32).unsqueeze(0)
        tcp_quat = torch.tensor(self.quat, dtype=torch.float32).unsqueeze(0)

        new_tcp_pos, new_tcp_quat = apply_delta_pose(tcp_pos, tcp_quat, action)
        new_tcp_pos = new_tcp_pos.squeeze(0)
        new_tcp_quat = new_tcp_quat.squeeze(0)

        new_tcp_pos = quat_apply(self.rot_180_z, new_tcp_pos)  # position rotated back
        new_tcp_quat = quat_mul(self.rot_180_z, new_tcp_quat)  # orientation rotated back

        final_rotvec = axis_angle_from_quat(new_tcp_quat)
        final_rotvec = final_rotvec.numpy()
        self.rotvec_action = final_rotvec.copy()

        # Combine final pose and send to robot
        new_tcp = np.concatenate((new_tcp_pos.numpy(), final_rotvec), axis=0)
        
        self.rtde_c.moveL(new_tcp.tolist(), speed=1.5, acceleration=1.5)
        # self.rtde_c.moveL(new_tcp.tolist(), speed=0.2, acceleration=0.2)


    # Function to get the robot's current state with quaternion in [w, x, y, z] format
    def get_robot_state(self):
        return np.concatenate((self.tcp_pose, self.target_pose, self.previous_action), axis=0)


    def quaternion_geodesic_distance(self, q1, q2):
        """Compute geodesic distance between two quaternions."""
        dot_product = np.clip(np.dot(q1, q2), -1.0, 1.0)  # Ensure valid range
        return 2 * np.arccos(abs(dot_product))  # Compute angle difference


    def run_on_real_robot(self):
        """Main control loop to move the robot towards sampled poses."""
        self.move_robot_to_home()
        
        if self.mode == "Predefined":
            self.get_predefined_pose()
        elif self.mode == "Sample":
            self.sample_random_pose()
        
        total_timestep = 0
        target_timestep = 0

        while True:
            self.get_actual_tcp_pose()
            obs = self.get_robot_state()

            with torch.inference_mode():
                self.action, _ = self.agent.predict(obs, deterministic=True)
                self.action *= self.action_scaling 

            if self.save:
                self.save_observations_to_csv(total_timestep)

            self.previous_action = self.action
            self.execute_action_on_real_robot()  

            position_distance = np.linalg.norm(self.target_pose[:3] - self.tcp_pose[:3])

            orientation_distance = self.quaternion_geodesic_distance(self.quat, self.target_quat)

            print(f"Position Error: {position_distance}, Orientation Error: {orientation_distance}")

            if target_timestep > 250: #500:
                print("\n\nAmount of Targets: ", self.current_index + 1)

                if self.mode == "Predefined":
                    self.current_index += 1
                    if self.current_index == 4: #3:
                        return
                    self.get_predefined_pose()
                elif self.mode == "Sample":
                    self.move_robot_to_home()
                    self.sample_random_pose()
                    self.current_index += 1
                target_timestep = -1

            total_timestep += 1  
            target_timestep += 1


if __name__ == "__main__":
    # robot_ip = "10.52.4.219"
    robot_ip = "10.126.49.30"
    # robot_ip = "192.168.1.100"
    
    model_path = "/home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/rot6d/hjh00riy/model.zip"
    # model_path = "/home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/rot6d/9vkwdb4c/model.zip"

    save_dir = "/home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/data/real_robot/"

    # filename = "domain_rand_model_predefined_poses_scale_0_05_seed_24_rotmat.csv"
    # filename = "domain_rand_model_predefined_poses_scale_0_05_seed_42_rotmat.csv"
    # filename = "domain_rand_model_predefined_poses_scale_0_01_seed_24_rotmat.csv"
    # filename = "domain_rand_model_predefined_poses_scale_0_01_seed_42_rotmat.csv"

    # filename = "domain_rand_model_predefined_poses_scale_0_05_seed_24_rotmat_clockwise.csv"
    # filename = "domain_rand_model_predefined_poses_scale_0_05_seed_24_rotmat_counterclockwise.csv"

    filename = "domain_rand_model_random_poses_scale_0_05_seed_24_rotmat.csv"
    # filename = "domain_rand_model_random_poses_scale_0_05_seed_42_rotmat.csv"
    # filename = "domain_rand_model_random_poses_scale_0_01_seed_24_rotmat.csv"
    # filename = "domain_rand_model_random_poses_scale_0_01_seed_42_rotmat.csv"
    
    action_scaling = 0.05
    save = False # False # True
    mode = "Sample" # Options available: "Sample" (sample uniformly from specified range), "Predefined" (predefined poses)

    controller = UR5eRobotController(robot_ip, model_path, action_scaling, save_dir, filename, mode, save)

    try:
        controller.run_on_real_robot()
    except KeyboardInterrupt:
        print("Stopping robot execution.")
    finally:
        controller.rtde_c.stopScript()
