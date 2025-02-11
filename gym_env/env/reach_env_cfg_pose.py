from dataclasses import MISSING

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from . import mdp
import os
import math

# This script includes several parts that can be commented in/out depending on the specific preference. 
# This can be used to run pretrained models with the specific setting used to train these models in sb3/models/
# This folder includes several subfolders that are named according to the wandb run name. Possible names are:
# rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e
# rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e_final
# rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e_final_v2
# rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e_final_v3
# rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e_final_v4

# With ctrl + F and these names, it is possible to comment in the specific settings used during training of these models.

# Or search for 
# New training setting
# to start a new training with new settings.


##
# Scene definition
##

MODEL_PATH = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")), "scene_models")

@configclass
class UR5e_ReachSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object."""
    # articulation
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/robot", 
        spawn=sim_utils.UsdFileCfg(
            # usd_path=os.path.join(MODEL_PATH, "ur5e_robotiq_new.usd"), # SDU gripper
            usd_path=os.path.join(MODEL_PATH, "ur5e_robotiq_hand_e.usd"), # Robotiq Hand E
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            # articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            #     enabled_self_collisions=True, 
            #     solver_position_iteration_count=8, 
            #     solver_velocity_iteration_count=0
            # ),
            activate_contact_sensors=True,), 
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.175, -0.175, 0.0), 
            joint_pos={
                "shoulder_pan_joint": 1.3, 
                "shoulder_lift_joint": -2.0, 
                "elbow_joint": 2.0, 
                "wrist_1_joint": -1.5, 
                "wrist_2_joint": -1.5, 
                "wrist_3_joint": 3.14, 
                "joint_left": 0.0, 
                "joint_right": 0.0,
            }
        ),
        actuators={
            "all_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*"],  # Match all joints
                velocity_limit={
                    "shoulder_pan_joint": 180.0,
                    "shoulder_lift_joint": 180.0,
                    "elbow_joint": 180.0,
                    "wrist_1_joint": 180.0,
                    "wrist_2_joint": 180.0,
                    "wrist_3_joint": 180.0,
                    "joint_left": 1000000.0,
                    "joint_right": 1000000.0,
                },
                effort_limit={
                    "shoulder_pan_joint": 87.0,
                    "shoulder_lift_joint": 87.0,
                    "elbow_joint": 87.0,
                    "wrist_1_joint": 87.0,
                    "wrist_2_joint": 87.0,
                    "wrist_3_joint": 87.0,
                    "joint_left": 200.0,
                    "joint_right": 200.0,
                },
                # stiffness={
                #     "shoulder_pan_joint": 261.79941,
                #     "shoulder_lift_joint": 261.79941,
                #     "elbow_joint": 261.79941,
                #     "wrist_1_joint": 261.79941,
                #     "wrist_2_joint": 261.79941,
                #     "wrist_3_joint": 261.79941,
                #     "joint_left": 3000.0,
                #     "joint_right": 3000.0,
                # },
                # damping={
                #     "shoulder_pan_joint": 26.17994,
                #     "shoulder_lift_joint": 26.17994,
                #     "elbow_joint": 26.17994,
                #     "wrist_1_joint": 26.17994,
                #     "wrist_2_joint": 26.17994,
                #     "wrist_3_joint": 26.17994,
                #     "joint_left": 800.0,
                #     "joint_right": 800.0,
                # }
                # stiffness={
                #     "shoulder_pan_joint": 1000.0,
                #     "shoulder_lift_joint": 1000.0,
                #     "elbow_joint": 1000.0,
                #     "wrist_1_joint": 1000.0,
                #     "wrist_2_joint": 1000.0,
                #     "wrist_3_joint": 1000.0,
                #     "joint_left": 3000.0,
                #     "joint_right": 3000.0,
                # },
                # damping={
                #     "shoulder_pan_joint": 121.66,
                #     "shoulder_lift_joint": 183.23,
                #     "elbow_joint": 96.54,
                #     "wrist_1_joint": 69.83,
                #     "wrist_2_joint": 69.83,
                #     "wrist_3_joint": 27.42,
                #     "joint_left": 500.0,
                #     "joint_right": 500.0,
                # }
                ############### Stiffness 800 ###############
                stiffness={
                    "shoulder_pan_joint": 800.0,
                    "shoulder_lift_joint": 800.0,
                    "elbow_joint": 800.0,
                    "wrist_1_joint": 800.0,
                    "wrist_2_joint": 800.0,
                    "wrist_3_joint": 800.0,
                    "joint_left": 3000.0,
                    "joint_right": 3000.0,
                },
                damping={
                    "shoulder_pan_joint": 108.82,
                    "shoulder_lift_joint": 163.89,
                    "elbow_joint": 86.35,
                    "wrist_1_joint": 62.46,
                    "wrist_2_joint": 62.46,
                    "wrist_3_joint": 24.53,
                    "joint_left": 500.0,
                    "joint_right": 500.0,
                }
            )
        }
    )

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -0.74]), 
        spawn=sim_utils.GroundPlaneCfg()
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Lights/Dome", 
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )    
    
    # Add the siegmund table to the scene
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/table", 
        spawn=sim_utils.UsdFileCfg(usd_path=os.path.join(MODEL_PATH, "Single_Siegmund_table.usd")), 
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(0.7071, 0.0, 0.0, 0.7071)),
    )



##
# MDP settings
##

@configclass
class CommandsCfg:
    """Command terms for the MDP."""
    # New training setting
    ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="wrist_3_link",
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(-0.15, 0.15),
            pos_y=(0.25, 0.5),
            pos_z=(0.1, 0.4),
            roll=(0.0, 0.0),
            pitch=(math.pi, math.pi),  # depends on end-effector axis
            yaw=(-3.14, 3.14), # (0.0, 0.0), # y
        ),
    )
    # # rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e
    # ee_pose = mdp.UniformPoseCommandCfg(
    #     asset_name="robot",
    #     body_name="wrist_3_link",
    #     resampling_time_range=(5.0, 5.0),
    #     debug_vis=True,
    #     ranges=mdp.UniformPoseCommandCfg.Ranges(
    #         pos_x=(-0.05, 0.05),
    #         pos_y=(0.35, 0.45),
    #         pos_z=(0.25, 0.35),
    #         roll=(0.0, 0.0),
    #         pitch=(math.pi, math.pi),  # depends on end-effector axis
    #         yaw=(-3.14, 3.14), # (0.0, 0.0), # y
    #     ),
    # )
    # # rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e_final
    # ee_pose = mdp.UniformPoseCommandCfg(
    #     asset_name="robot",
    #     body_name="wrist_3_link",
    #     resampling_time_range=(5.0, 5.0),
    #     debug_vis=True,
    #     ranges=mdp.UniformPoseCommandCfg.Ranges(
    #         pos_x=(-0.2, 0.2),
    #         pos_y=(0.25, 0.5),
    #         pos_z=(0.1, 0.4),
    #         roll=(0.0, 0.0),
    #         pitch=(math.pi, math.pi),  # depends on end-effector axis
    #         yaw=(-3.14, 3.14), # (0.0, 0.0), # y
    #     ),
    # )
    # # rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e_final_v2
    # # rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e_final_v3
    # # rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e_final_v4
    # ee_pose = mdp.UniformPoseCommandCfg(
    #     asset_name="robot",
    #     body_name="wrist_3_link",
    #     resampling_time_range=(5.0, 5.0),
    #     debug_vis=True,
    #     ranges=mdp.UniformPoseCommandCfg.Ranges(
    #         pos_x=(-0.15, 0.15),
    #         pos_y=(0.25, 0.5),
    #         pos_z=(0.1, 0.4),
    #         roll=(0.0, 0.0),
    #         pitch=(math.pi, math.pi),  # depends on end-effector axis
    #         yaw=(-3.14, 3.14), # (0.0, 0.0), # y
    #     ),
    # )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    # Set actions
    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""              
        # TCP pose in base frame
        tcp_pose = ObsTerm(
            func=mdp.get_current_tcp_pose,
            params={"gripper_offset": [0.0, 0.0, 0.15], "robot_cfg": SceneEntityCfg("robot", body_names=["wrist_3_link"])},
            noise=Unoise(n_min=-0.0001, n_max=0.0001), # New training setting
            # # No Unoise for rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e
            # noise=Unoise(n_min=-0.001, n_max=0.001), # rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e_final
            # noise=Unoise(n_min=-0.0001, n_max=0.0001), # rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e_final_v2, rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e_final_v3, rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e_final_v4

        )

        # Desired ee (or tcp) pose in base frame
        # pose_command = ObsTerm(
        #     func=mdp.generated_commands_axis_angle,
        #     params={"command_name": "ee_pose"},
        # )

        pose_command = ObsTerm(
            func=mdp.generated_commands, 
            params={"command_name": "ee_pose"},
            noise=Unoise(n_min=-0.0001, n_max=0.0001), # New training setting
            # No Unoise for rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e
            # noise=Unoise(n_min=-0.001, n_max=0.001), # rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e_final
            # noise=Unoise(n_min=-0.0001, n_max=0.0001), # rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e_final_v2, rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e_final_v3, rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e_final_v4
        )

        # Previous action
        actions = ObsTerm(
            func=mdp.last_action
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # New training setting
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.8, 1.2),
            "velocity_range": (0.0, 0.0),
        },
    )
    # # rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e
    # reset_robot_joints = EventTerm(
    #     func=mdp.reset_joints_by_scale,
    #     mode="reset",
    #     params={
    #         "position_range": (1.0, 1.0),
    #         "velocity_range": (0.0, 0.0),
    #     },
    # )
    # # rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e_final
    # # rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e_final_v2
    # reset_robot_joints = EventTerm(
    #     func=mdp.reset_joints_by_scale,
    #     mode="reset",
    #     params={
    #         "position_range": (0.5, 1.5),
    #         "velocity_range": (0.0, 0.0),
    #     },
    # )
    # # rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e_final_v3
    # reset_robot_joints = EventTerm(
    #     func=mdp.reset_joints_by_scale,
    #     mode="reset",
    #     params={
    #         "position_range": (0.8, 1.2),
    #         "velocity_range": (0.0, 0.0),
    #     },
    # )
    # # rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e_final_v4
    # reset_robot_joints = EventTerm(
    #     func=mdp.reset_joints_by_scale,
    #     mode="reset",
    #     params={
    #         "position_range": (0.7, 1.3),
    #         "velocity_range": (0.0, 0.0),
    #     },
    # )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    # task terms
    end_effector_position_tracking = RewTerm(
        func=mdp.position_command_error,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["wrist_3_link"]), "command_name": "ee_pose"},
    )
    end_effector_position_tracking_fine_grained = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["wrist_3_link"]), "std": 0.1, "command_name": "ee_pose"},
    )
    end_effector_orientation_tracking = RewTerm(
        func=mdp.orientation_command_error,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["wrist_3_link"]), "command_name": "ee_pose"},
    )


    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.0001)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)



@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -0.005, "num_steps": 4500}
    )


##
# Environment configuration
##

@configclass
class UR5e_ReachEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""
    # Scene settings
    scene: UR5e_ReachSceneCfg = UR5e_ReachSceneCfg(num_envs=4, env_spacing=2.5)

    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()

    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 15.0
        # simulation settings
        self.sim.dt = 0.01 #1/60
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.physx.gpu_collision_stack_size = 4096 * 4096 * 120 # Was added due to an PhysX error: collisionStackSize buffer overflow detected