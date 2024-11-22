# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def get_current_tcp_pose(
        env: ManagerBasedRLEnv,
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: RigidObject = env.scene[robot_cfg.name]

    wrist_3_index = 8
    body_state_w_list = robot.data.body_state_w.clone()

    wrist_3_state = body_state_w_list[:, wrist_3_index]
    wrist_3_state[:, :3] += torch.tensor([0.0, 0.0, 0.135], device="cuda")
    return wrist_3_state[:, :7]