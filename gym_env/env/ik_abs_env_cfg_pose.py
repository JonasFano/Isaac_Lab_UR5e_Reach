from isaaclab.utils import configclass
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg

from . import reach_env_cfg_pose
from taskparameters_ur5e import TaskParams

@configclass
class AbsIK_UR5e_ReachEnvCfg(reach_env_cfg_pose.UR5e_ReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/robot/base_link",
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/robot/wrist_3_link",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=TaskParams.gripper_offset,
                    ),
                ),
            ],
        )

        # print(self.commands.ee_pose) # Do not show current end-effector frame
        self.commands.ee_pose.current_pose_visualizer_cfg.markers['frame'].visible = False

        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=TaskParams.joint_names,
            body_name=TaskParams.ee_body_name,
            controller=DifferentialIKControllerCfg(command_type=TaskParams.command_type, use_relative_mode=False, ik_method=TaskParams.ik_method),
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=TaskParams.gripper_offset),
            scale=TaskParams.action_scale,
            debug_vis=False  # Enable debug visualization
        )



@configclass
class AbsIK_UR5e_ReachEnvCfg_PLAY(AbsIK_UR5e_ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 4
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False

