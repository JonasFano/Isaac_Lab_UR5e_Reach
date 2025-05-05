from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg
from . import mdp
import os
from gym_env.env.controller.impedance_control_cfg import ImpedanceControllerCfg

from taskparameters import TaskParams

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
            usd_path=os.path.join(MODEL_PATH, "ur5e_old.usd"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, 
                solver_position_iteration_count=8, 
                solver_velocity_iteration_count=0
            ),
            activate_contact_sensors=True,), 
        init_state=ArticulationCfg.InitialStateCfg(
            pos=TaskParams.robot_base_init_position, 
            joint_pos={
                "shoulder_pan_joint": TaskParams.robot_initial_joint_pos[0], 
                "shoulder_lift_joint": TaskParams.robot_initial_joint_pos[1], 
                "elbow_joint": TaskParams.robot_initial_joint_pos[2], 
                "wrist_1_joint": TaskParams.robot_initial_joint_pos[3], 
                "wrist_2_joint": TaskParams.robot_initial_joint_pos[4], 
                "wrist_3_joint": TaskParams.robot_initial_joint_pos[5], 
            }
        ),
        actuators={
            "all_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*"],  # Match all joints
                velocity_limit={
                    "shoulder_pan_joint": TaskParams.robot_vel_limit,
                    "shoulder_lift_joint": TaskParams.robot_vel_limit,
                    "elbow_joint": TaskParams.robot_vel_limit,
                    "wrist_1_joint": TaskParams.robot_vel_limit,
                    "wrist_2_joint": TaskParams.robot_vel_limit,
                    "wrist_3_joint": TaskParams.robot_vel_limit,
                },
                effort_limit={
                    "shoulder_pan_joint": TaskParams.robot_effort_limit,
                    "shoulder_lift_joint": TaskParams.robot_effort_limit,
                    "elbow_joint": TaskParams.robot_effort_limit,
                    "wrist_1_joint": TaskParams.robot_effort_limit,
                    "wrist_2_joint": TaskParams.robot_effort_limit,
                    "wrist_3_joint": TaskParams.robot_effort_limit,
                },
                stiffness = {
                    "shoulder_pan_joint": TaskParams.robot_stiffness,
                    "shoulder_lift_joint": TaskParams.robot_stiffness,
                    "elbow_joint": TaskParams.robot_stiffness,
                    "wrist_1_joint": TaskParams.robot_stiffness,
                    "wrist_2_joint": TaskParams.robot_stiffness,
                    "wrist_3_joint": TaskParams.robot_stiffness,
                },
                damping = {
                    "shoulder_pan_joint": TaskParams.shoulder_pan_damping,
                    "shoulder_lift_joint": TaskParams.shoulder_lift_damping,
                    "elbow_joint": TaskParams.elbow_damping,
                    "wrist_1_joint": TaskParams.wrist_1_damping,
                    "wrist_2_joint": TaskParams.wrist_2_damping,
                    "wrist_3_joint": TaskParams.wrist_3_damping,
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
    ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="wrist_3_link",
        resampling_time_range=TaskParams.resampling_time_range,
        debug_vis=TaskParams.visualize_frame,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=TaskParams.sample_range_pos_x,
            pos_y=TaskParams.sample_range_pos_y,
            pos_z=TaskParams.sample_range_pos_z,
            roll=TaskParams.sample_range_roll,
            pitch=TaskParams.sample_range_pitch,  
            yaw=TaskParams.sample_range_yaw,
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    # Set actions
    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg | ImpedanceControllerCfg = MISSING 


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""              
        # TCP pose in base frame
        tcp_pose = ObsTerm(
            func=mdp.get_current_tcp_pose,
            params={"gripper_offset": TaskParams.gripper_offset, "robot_cfg": SceneEntityCfg("robot", body_names=["wrist_3_link"])},
        )

        # Desired ee (or tcp) pose in base frame
        # pose_command = ObsTerm(
        #     func=mdp.generated_commands_axis_angle,
        #     params={"command_name": "ee_pose"},
        # )

        pose_command = ObsTerm(
            func=mdp.generated_commands, 
            params={"command_name": "ee_pose"},
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

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    # task terms
    end_effector_position_tracking = RewTerm(
        func=mdp.position_command_error,
        weight=TaskParams.end_effector_position_tracking_weight,
        params={"gripper_offset": TaskParams.gripper_offset, "asset_cfg": SceneEntityCfg("robot", body_names=["wrist_3_link"]), "command_name": "ee_pose"},
    )
    end_effector_position_tracking_fine_grained = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=TaskParams.end_effector_position_tracking_fine_grained_weight,
        params={"gripper_offset": TaskParams.gripper_offset, "asset_cfg": SceneEntityCfg("robot", body_names=["wrist_3_link"]), "std": TaskParams.end_effector_position_tracking_fine_grained_std, "command_name": "ee_pose"},
    )
    # end_effector_orientation_tracking = RewTerm(
    #     func=mdp.orientation_command_error,
    #     weight=TaskParams.end_effector_orientation_tracking_weight,
    #     params={"asset_cfg": SceneEntityCfg("robot", body_names=["wrist_3_link"]), "command_name": "ee_pose"},
    # )

    # action_rate = RewTerm(func=mdp.action_rate_l2, weight=TaskParams.action_rate_weight)
    action_rate = RewTerm(func=mdp.action_rate_l2_position, weight=TaskParams.action_rate_weight)

    # action_magnitude = RewTerm(func=mdp.action_l2, weight=TaskParams.action_magnitude_weight)
    # action_magnitude = RewTerm(func=mdp.action_l2_position, weight=TaskParams.action_magnitude_weight)

    # ee_acc = RewTerm(
    #     func=mdp.body_lin_acc_l2,
    #     weight=TaskParams.ee_acc_weight,
    #     params={"asset_cfg": SceneEntityCfg("robot", body_names=["wrist_3_link"]),}
    # )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)



@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # action_rate = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": TaskParams.action_rate_curriculum_weight, "num_steps": TaskParams.curriculum_num_steps} 
    # )

    # action_magnitude = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "action_magnitude", "weight": TaskParams.action_magnitude_curriculum_weight, "num_steps": TaskParams.curriculum_num_steps}
    # )

    # ee_acc = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "ee_acc", "weight": TaskParams.ee_acc_curriculum_weight, "num_steps": TaskParams.curriculum_num_steps}
    # )


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
        self.decimation = TaskParams.decimation
        self.episode_length_s = TaskParams.episode_length_s
        # simulation settings
        self.sim.dt = TaskParams.dt
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.physx.gpu_collision_stack_size = 4096 * 4096 * 100 # Was added due to an PhysX error: collisionStackSize buffer overflow detected