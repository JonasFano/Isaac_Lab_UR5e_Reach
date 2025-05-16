from isaaclab.utils import configclass
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from gym_env.env.controller.impedance_control_cfg import ImpedanceControllerCfg
from gym_env.env.mdp.actions.actions_cfg import ImpedanceControllerActionCfg

from . import reach_env_cfg_pose


# from taskparameters_peginsert import TaskParams

@configclass
class ImpCtrl_UR5e_EnvCfg(reach_env_cfg_pose.UR5e_ReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Visualize Base frame
        base_marker_cfg = FRAME_MARKER_CFG.copy()
        base_marker_cfg.markers["frame"].scale = (0.15, 0.15, 0.25)
        base_marker_cfg.prim_path = "/Visuals/BaseFrame"
        self.scene.world_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/robot/base_link",
            debug_vis=True,
            visualizer_cfg=base_marker_cfg,
            target_frames=[  # Add at least one target frame
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/robot/base_link",
                    name="base_frame",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.0],
                    ),
                )
            ],
        )

        # Visualize TCP frame
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

        # print(self.commands.object_pose) # Do not show current end-effector frame
        # self.commands.object_pose.current_pose_visualizer_cfg.markers['frame'].visible = False

        self.actions.arm_action = ImpedanceControllerActionCfg(
            asset_name="robot",
            joint_names=[
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
            body_name="wrist_3_link",
            controller_cfg=ImpedanceControllerCfg(
                gravity_compensation=True,
                coriolis_centrifugal_compensation = True,
                inertial_dynamics_decoupling=True,
                max_torque_clamping=None, # Array of max torques to clamp computed torques - no clamping if None - [150.0, 150.0, 150.0, 28.0, 28.0, 28.0]
                stiffness=[1000, 1000, 1000, 300, 300, 300],
                damping=None, # None = Critically damped
            ),
            action_scale=0.05,
            # tcp_offset=[0.0, 0.0, 0.0],
        )


@configclass
class ImpCtrl_UR5e_EnvCfg_PLAY(ImpCtrl_UR5e_EnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 4
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False

