# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, List
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms, quat_apply, axis_angle_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def quat_rotate_vector(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """
    Rotate a vector by a quaternion using provided utility functions.
    
    Args:
        quat: [..., 4] tensor representing quaternions (w, x, y, z).
        vec: [..., 3] tensor representing the vectors to rotate.
    
    Returns:
        Rotated vector of shape [..., 3].
    """
    # Ensure the quaternion is normalized to avoid unintended scaling effects
    quat = torch.nn.functional.normalize(quat, p=2, dim=-1)
    
    # Rotate the input vector using the quaternion
    rotated_vec = quat_apply(quat, vec)
    
    return rotated_vec



def get_current_tcp_pose(env: ManagerBasedRLEnv, gripper_offset: List[float], robot_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Compute the current TCP pose in both the base frame and world frame.
    
    Args:
        env: ManagerBasedRLEnv object containing the virtual environment.
        robot_cfg: Configuration for the robot entity, defaults to "robot".
    
    Returns:
        tcp_pose_b: TCP pose in the robot's base frame (position + quaternion).
    """
    # Access the robot object from the scene using the provided configuration
    robot: RigidObject = env.scene[robot_cfg.name]

    # Clone the body states in the world frame to avoid modifying the original tensor
    body_state_w_list = robot.data.body_state_w.clone()

    # Extract the pose of the end-effector (position + orientation) in the world frame
    ee_pose_w = body_state_w_list[:, robot_cfg.body_ids[0], :7]

    # Define the offset from the end-effector frame to the TCP in the end-effector frame
    offset_ee = torch.tensor(gripper_offset, dtype=torch.float32, device="cuda").unsqueeze(0).repeat(env.scene.num_envs, 1)

    # Rotate the offset from the end-effector frame to the world frame
    offset_w = quat_rotate_vector(ee_pose_w[:, 3:7], offset_ee)

    # Compute the TCP pose in the world frame by adding the offset to the end-effector's position
    tcp_pose_w = torch.cat((ee_pose_w[:, :3] + offset_w, ee_pose_w[:, 3:7]), dim=-1)

    # Transform the TCP pose from the world frame to the robot's base frame
    tcp_pos_b, tcp_quat_b = subtract_frame_transforms(
        robot.data.root_state_w[:, :3],  # Robot base position in world frame
        robot.data.root_state_w[:, 3:7],  # Robot base orientation in world frame
        tcp_pose_w[:, :3],  # TCP position in world frame
        tcp_pose_w[:, 3:7]  # TCP orientation in world frame
    )

    # Either axis angle orientation representation:
    # # Convert orientation from quat to axis-angle
    # tcp_axis_angle_b = axis_angle_from_quat(tcp_quat_b)
    # # Concatenate the position and axis-angle into a single tensor
    # tcp_pose_b = torch.cat((tcp_pos_b, tcp_axis_angle_b), dim=-1)

    # tcp_pose_b = torch.cat((tcp_pos_b, rot), dim=-1)
    tcp_pose_b = torch.cat((tcp_pos_b, tcp_quat_b), dim=-1)
    return tcp_pose_b


def get_current_tcp_pose_rot6d(env: ManagerBasedRLEnv, gripper_offset: List[float], robot_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Compute the current TCP pose in both the base frame and world frame with Rot6D representation of orientations.
    
    Args:
        env: ManagerBasedRLEnv object containing the virtual environment.
        robot_cfg: Configuration for the robot entity, defaults to "robot".
    
    Returns:
        tcp_pose_b: TCP pose in the robot's base frame (position + quaternion).
    """
    tcp_pose_b = get_current_tcp_pose(env, gripper_offset, robot_cfg)

    tcp_pos_b = tcp_pose_b[:, :3]
    tcp_quat_b = tcp_pose_b[:, 3:]

    rotmat = quat_to_rotmat(tcp_quat_b)  # shape: [N, 3, 3]
    rot = rotmat[:, :, :2].reshape(-1, 6)  # take first two columns and flatten
    # rot = rotmat.reshape(-1, 9)  # take 9D representation

    tcp_pose_b = torch.cat((tcp_pos_b, rot), dim=-1)
    return tcp_pose_b



def generated_commands_axis_angle(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name in axis-angle representation."""
    pose_quat_b = env.command_manager.get_command(command_name)

    # Convert orientation from quat to euler angles xyz
    axis_angle_b = axis_angle_from_quat(pose_quat_b[:, 3:7])
    # Concatenate the position and axis angle orientation into a single tensor
    pose_b = torch.cat((pose_quat_b[:, :3], axis_angle_b), dim=-1)

    return pose_b



def generated_commands_rot6d(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Return command as position + 6D rotation representation (first two columns of rotation matrix)."""
    pose_b = env.command_manager.get_command(command_name)  # shape: [N, 7], [x, y, z, qw, qx, qy, qz]

    pos_b = pose_b[:, :3]
    quat_b = pose_b[:, 3:]

    rotmat = quat_to_rotmat(quat_b)  # shape: [N, 3, 3]
    rot = rotmat[:, :, :2].reshape(-1, 6)  # take first two columns and flatten
    # rot = rotmat.reshape(-1, 9)  # take 9D representation

    return torch.cat((pos_b, rot), dim=-1)  # shape: [N, 9]




def quat_to_rotmat(q):
    # q: [batch, 4] in (w, x, y, z) format
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
