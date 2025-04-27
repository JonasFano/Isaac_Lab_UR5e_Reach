# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, List

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul, quat_apply

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


def get_current_tcp_pose_w(env: ManagerBasedRLEnv, gripper_offset: List[float], robot_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Compute the current TCP pose in both the base frame and world frame.
    
    Args:
        env: ManagerBasedRLEnv object containing the virtual environment.
        robot_cfg: Configuration for the robot entity, defaults to "robot".
    
    Returns:
        tcp_pose_w: TCP pose in the world frame (position + quaternion).
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

    return tcp_pose_w


def position_command_error(env: ManagerBasedRLEnv, gripper_offset: List[float], command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame). The position error is computed as the L2-norm
    of the difference between the desired and current positions.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)

    # Current TCP position in world frame -> Function accounts for end-effector to TCP offset
    # curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
    curr_pos_w = get_current_tcp_pose_w(env, gripper_offset, asset_cfg)[:, :3]

    # print("Position l2 norm: ", torch.norm(curr_pos_w - des_pos_w, dim=1))

    return torch.norm(curr_pos_w - des_pos_w, dim=1)


def position_command_error_tanh(
    env: ManagerBasedRLEnv, gripper_offset: List[float], std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of the position using the tanh kernel.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame) and maps it with a tanh kernel.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)


    # Current TCP position in world frame -> Function accounts for end-effector to TCP offset
    # curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
    curr_pos_w = get_current_tcp_pose_w(env, gripper_offset, asset_cfg)[:, :3]

    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    # print("Position Error: ", distance)
    return 1 - torch.tanh(distance / std)


def orientation_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking orientation error using shortest path.

    The function computes the orientation error between the desired orientation (from the command) and the
    current orientation of the asset's body (in world frame). The orientation error is computed as the shortest
    path between the desired and current orientations.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current orientations
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)

    # Not necessary to account for end-effector to TCP offset, as there is no change in orientation 
    curr_quat_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]  # type: ignore

    quat_error = quat_error_magnitude(curr_quat_w, des_quat_w)

    # print("Quaternion Error: ", quat_error)
    # print("Geodesic magnitude: ", quaternion_geodesic_distance(curr_quat_w, des_quat_w))
    # print("Log Error: ", quat_log_error(curr_quat_w, des_quat_w))

    # return quat_error_magnitude(curr_quat_w, des_quat_w)
    return quat_error


def action_rate_l2_position(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action[:, :3] - env.action_manager.prev_action[:, :3]), dim=1)


def action_l2_position(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action[:, :3]), dim=1)


def action_clip(env: ManagerBasedRLEnv, pos_threshold: float, axis_angle_threshold: float) -> torch.Tensor:
    """Penalize the actions if their absolute values exceed specified limits"""
    action_pos = env.action_manager.action[:, :3]  # Smaller than 0.05
    action_quat = env.action_manager.action[:, 3:7]  # Smaller than 0.08

    # Check if any element of the absolute action_pos exceeds pos_threshold
    pos_exceeds = (torch.abs(action_pos) > pos_threshold).any(dim=1)

    # Check if any element of the absolute action_quat exceeds quat_threshold
    quat_exceeds = (torch.abs(action_quat) > axis_angle_threshold).any(dim=1)

    # Return 1 if any condition is met, otherwise return 0
    reward = torch.where(pos_exceeds | quat_exceeds, 
                         torch.tensor(1.0, device=env.action_manager.action.device), 
                         torch.tensor(0.0, device=env.action_manager.action.device))

    return reward
