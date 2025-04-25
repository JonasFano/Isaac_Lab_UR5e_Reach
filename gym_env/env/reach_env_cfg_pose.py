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
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
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
    # robot = ArticulationCfg(
    #     prim_path="{ENV_REGEX_NS}/robot", 
    #     spawn=sim_utils.UsdFileCfg(
    #         # usd_path=os.path.join(MODEL_PATH, "ur5e_robotiq_new.usd"), # SDU gripper
    #         usd_path=os.path.join(MODEL_PATH, "ur5e_robotiq_hand_e.usd"), # Robotiq Hand E
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #             disable_gravity=False,
    #             max_depenetration_velocity=5.0,
    #         ),
    #         # articulation_props=sim_utils.ArticulationRootPropertiesCfg(
    #         #     enabled_self_collisions=True, 
    #         #     solver_position_iteration_count=8, 
    #         #     solver_velocity_iteration_count=0
    #         # ),
    #         activate_contact_sensors=True,), 
    #     init_state=ArticulationCfg.InitialStateCfg(
    #         pos=(0.175, -0.175, 0.0), 
    #         joint_pos={
    #             "shoulder_pan_joint": 1.3, 
    #             "shoulder_lift_joint": -2.0, 
    #             "elbow_joint": 2.0, 
    #             "wrist_1_joint": -1.5, 
    #             "wrist_2_joint": -1.5, 
    #             "wrist_3_joint": 3.14, 
    #             "joint_left": 0.0, 
    #             "joint_right": 0.0,
    #         }
    #     ),
    #     actuators={
    #         "all_joints": ImplicitActuatorCfg(
    #             joint_names_expr=[".*"],  # Match all joints
    #             velocity_limit={
    #                 "shoulder_pan_joint": 180.0,
    #                 "shoulder_lift_joint": 180.0,
    #                 "elbow_joint": 180.0,
    #                 "wrist_1_joint": 180.0,
    #                 "wrist_2_joint": 180.0,
    #                 "wrist_3_joint": 180.0,
    #                 "joint_left": 1000000.0,
    #                 "joint_right": 1000000.0,
    #             },
    #             effort_limit={
    #                 "shoulder_pan_joint": 87.0,
    #                 "shoulder_lift_joint": 87.0,
    #                 "elbow_joint": 87.0,
    #                 "wrist_1_joint": 87.0,
    #                 "wrist_2_joint": 87.0,
    #                 "wrist_3_joint": 87.0,
    #                 "joint_left": 200.0,
    #                 "joint_right": 200.0,
    #             },
    #             # stiffness={
    #             #     "shoulder_pan_joint": 261.79941,
    #             #     "shoulder_lift_joint": 261.79941,
    #             #     "elbow_joint": 261.79941,
    #             #     "wrist_1_joint": 261.79941,
    #             #     "wrist_2_joint": 261.79941,
    #             #     "wrist_3_joint": 261.79941,
    #             #     "joint_left": 3000.0,
    #             #     "joint_right": 3000.0,
    #             # },
    #             # damping={
    #             #     "shoulder_pan_joint": 26.17994,
    #             #     "shoulder_lift_joint": 26.17994,
    #             #     "elbow_joint": 26.17994,
    #             #     "wrist_1_joint": 26.17994,
    #             #     "wrist_2_joint": 26.17994,
    #             #     "wrist_3_joint": 26.17994,
    #             #     "joint_left": 800.0,
    #             #     "joint_right": 800.0,
    #             # }
    #             # stiffness={
    #             #     "shoulder_pan_joint": 1000.0,
    #             #     "shoulder_lift_joint": 1000.0,
    #             #     "elbow_joint": 1000.0,
    #             #     "wrist_1_joint": 1000.0,
    #             #     "wrist_2_joint": 1000.0,
    #             #     "wrist_3_joint": 1000.0,
    #             #     "joint_left": 3000.0,
    #             #     "joint_right": 3000.0,
    #             # },
    #             # damping={
    #             #     "shoulder_pan_joint": 121.66,
    #             #     "shoulder_lift_joint": 183.23,
    #             #     "elbow_joint": 96.54,
    #             #     "wrist_1_joint": 69.83,
    #             #     "wrist_2_joint": 69.83,
    #             #     "wrist_3_joint": 27.42,
    #             #     "joint_left": 500.0,
    #             #     "joint_right": 500.0,
    #             # }
    #             # ############### Stiffness 800 ###############
    #             # stiffness={
    #             #     "shoulder_pan_joint": 800.0,
    #             #     "shoulder_lift_joint": 800.0,
    #             #     "elbow_joint": 800.0,
    #             #     "wrist_1_joint": 800.0,
    #             #     "wrist_2_joint": 800.0,
    #             #     "wrist_3_joint": 800.0,
    #             #     "joint_left": 3000.0,
    #             #     "joint_right": 3000.0,
    #             # },
    #             # damping={
    #             #     "shoulder_pan_joint": 108.82,
    #             #     "shoulder_lift_joint": 163.89,
    #             #     "elbow_joint": 86.35,
    #             #     "wrist_1_joint": 62.46,
    #             #     "wrist_2_joint": 62.46,
    #             #     "wrist_3_joint": 24.53,
    #             #     "joint_left": 500.0,
    #             #     "joint_right": 500.0,
    #             # }
    #             # ############### Stiffness 5000000 ###############
    #             # stiffness = {
    #             #     "shoulder_pan_joint": 5000000.0,
    #             #     "shoulder_lift_joint": 5000000.0,
    #             #     "elbow_joint": 5000000.0,
    #             #     "wrist_1_joint": 5000000.0,
    #             #     "wrist_2_joint": 5000000.0,
    #             #     "wrist_3_joint": 5000000.0,
    #             #     "joint_left": 3000.0,
    #             #     "joint_right": 3000.0,
    #             # },
    #             # damping = {
    #             #     "shoulder_pan_joint": 8591.93,
    #             #     "shoulder_lift_joint": 12954.54,
    #             #     "elbow_joint": 6815.50,
    #             #     "wrist_1_joint": 4937.82,
    #             #     "wrist_2_joint": 4937.82,
    #             #     "wrist_3_joint": 1943.25,
    #             #     "joint_left": 500.0,
    #             #     "joint_right": 500.0,
    #             # }
    #             ############### Stiffness 100000000 ###############
    #             stiffness = {
    #                 "shoulder_pan_joint": 10000000.0,
    #                 "shoulder_lift_joint": 10000000.0,
    #                 "elbow_joint": 10000000.0,
    #                 "wrist_1_joint": 10000000.0,
    #                 "wrist_2_joint": 10000000.0,
    #                 "wrist_3_joint": 10000000.0,
    #                 "joint_left": 10000000.0,
    #                 "joint_right": 10000000.0,
    #             },
    #             damping = {
    #                 "shoulder_pan_joint": 12166.86,
    #                 "shoulder_lift_joint": 18333.30,
    #                 "elbow_joint": 9651.90,
    #                 "wrist_1_joint": 6991.16,
    #                 "wrist_2_joint": 6991.16,
    #                 "wrist_3_joint": 2752.97,
    #                 "joint_left": 50000.0,
    #                 "joint_right": 50000.0,
    #             }
    #         )
    #     }
    # )
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/robot", 
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(MODEL_PATH, "ur5e_old.usd"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
                # disable_gravity=False,
                # max_depenetration_velocity=5.0,
                # linear_damping=0.0,
                # angular_damping=0.0,
                # max_linear_velocity=1000.0,
                # max_angular_velocity=3666.0,
                # enable_gyroscopic_forces=True,
                # solver_position_iteration_count=192,
                # solver_velocity_iteration_count=1,
                # max_contact_impulse=1e32,
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
                "wrist_3_joint": 0.0, #3.14, 
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
                },
                effort_limit={
                    "shoulder_pan_joint": 87.0,
                    "shoulder_lift_joint": 87.0,
                    "elbow_joint": 87.0,
                    "wrist_1_joint": 87.0,
                    "wrist_2_joint": 87.0,
                    "wrist_3_joint": 87.0,
                },
                # ############### Stiffness 100 ###############
                # stiffness={
                #     "shoulder_pan_joint": 100.0,
                #     "shoulder_lift_joint": 100.0,
                #     "elbow_joint": 100.0,
                #     "wrist_1_joint": 100.0,
                #     "wrist_2_joint": 100.0,
                #     "wrist_3_joint": 100.0,
                # },
                # damping={
                #     "shoulder_pan_joint": 38.47,
                #     "shoulder_lift_joint": 57.94,
                #     "elbow_joint": 30.53,
                #     "wrist_1_joint": 22.08,
                #     "wrist_2_joint": 22.08,
                #     "wrist_3_joint": 8.67,
                # }
                # ############### Stiffness 500 ###############
                # stiffness={
                #     "shoulder_pan_joint": 500.0,
                #     "shoulder_lift_joint": 500.0,
                #     "elbow_joint": 500.0,
                #     "wrist_1_joint": 500.0,
                #     "wrist_2_joint": 500.0,
                #     "wrist_3_joint": 500.0,
                # },
                # damping={
                #     "shoulder_pan_joint": 86.03,
                #     "shoulder_lift_joint": 129.56,
                #     "elbow_joint": 68.26,
                #     "wrist_1_joint": 49.38,
                #     "wrist_2_joint": 49.38,
                #     "wrist_3_joint": 19.39,
                # }
                # ############### Stiffness 800 ###############
                # stiffness={
                #     "shoulder_pan_joint": 800.0,
                #     "shoulder_lift_joint": 800.0,
                #     "elbow_joint": 800.0,
                #     "wrist_1_joint": 800.0,
                #     "wrist_2_joint": 800.0,
                #     "wrist_3_joint": 800.0,
                # },
                # damping={
                #     "shoulder_pan_joint": 108.82,
                #     "shoulder_lift_joint": 163.89,
                #     "elbow_joint": 86.35,
                #     "wrist_1_joint": 62.46,
                #     "wrist_2_joint": 62.46,
                #     "wrist_3_joint": 24.53,
                # }
                # ############### Stiffness 1000 ###############
                # stiffness={
                #     "shoulder_pan_joint": 1000.0,
                #     "shoulder_lift_joint": 1000.0,
                #     "elbow_joint": 1000.0,
                #     "wrist_1_joint": 1000.0,
                #     "wrist_2_joint": 1000.0,
                #     "wrist_3_joint": 1000.0,
                # },
                # damping={
                #     "shoulder_pan_joint": 121.66,
                #     "shoulder_lift_joint": 183.23,
                #     "elbow_joint": 96.54,
                #     "wrist_1_joint": 69.83,
                #     "wrist_2_joint": 69.83,
                #     "wrist_3_joint": 27.42,
                # }
                # ############### Stiffness 1200 ###############
                # stiffness={
                #     "shoulder_pan_joint": 1200.0,
                #     "shoulder_lift_joint": 1200.0,
                #     "elbow_joint": 1200.0,
                #     "wrist_1_joint": 1200.0,
                #     "wrist_2_joint": 1200.0,
                #     "wrist_3_joint": 1200.0,
                # },RelIK_UR5e_ReachEnvCfg
                # damping={
                #     "shoulder_pan_joint": 133.27,
                #     "shoulder_lift_joint": 200.72,
                #     "elbow_joint": 105.75,
                #     "wrist_1_joint": 76.49,
                #     "wrist_2_joint": 76.49,
                #     "wrist_3_joint": 30.04,
                # }
                # ############### Stiffness 1500 ###############
                # stiffness={
                #     "shoulder_pan_joint": 1500.0,
                #     "shoulder_lift_joint": 1500.0,
                #     "elbow_joint": 1500.0,
                #     "wrist_1_joint": 1500.0,
                #     "wrist_2_joint": 1500.0,
                #     "wrist_3_joint": 1500.0,
                # },
                # damping={
                #     "shoulder_pan_joint": 149.00,
                #     "shoulder_lift_joint": 224.41,
                #     "elbow_joint": 118.24,
                #     "wrist_1_joint": 85.52,
                #     "wrist_2_joint": 85.52,
                #     "wrist_3_joint": 33.58,
                # }
                # ############### Stiffness 2000 ###############
                # stiffness={
                #     "shoulder_pan_joint": 2000.0,
                #     "shoulder_lift_joint": 2000.0,
                #     "elbow_joint": 2000.0,
                #     "wrist_1_joint": 2000.0,
                #     "wrist_2_joint": 2000.0,
                #     "wrist_3_joint": 2000.0,
                # },
                # damping={
                #     "shoulder_pan_joint": 172.03,
                #     "shoulder_lift_joint": 259.27,
                #     "elbow_joint": 136.59,
                #     "wrist_1_joint": 98.84,
                #     "wrist_2_joint": 98.84,
                #     "wrist_3_joint": 38.83,
                # }

                # ############### Stiffness 50000 ###############
                # stiffness = {
                #     "shoulder_pan_joint": 50000.0,
                #     "shoulder_lift_joint": 50000.0,
                #     "elbow_joint": 50000.0,
                #     "wrist_1_joint": 50000.0,
                #     "wrist_2_joint": 50000.0,
                #     "wrist_3_joint": 50000.0,
                # },
                # damping = {
                #     "shoulder_pan_joint": 860.33,
                #     "shoulder_lift_joint": 1296.36,
                #     "elbow_joint": 682.49,
                #     "wrist_1_joint": 494.35,
                #     "wrist_2_joint": 494.35,
                #     "wrist_3_joint": 194.00,
                # }
                # ############### Stiffness 200000 ###############
                # stiffness = {
                #     "shoulder_pan_joint": 200000.0,
                #     "shoulder_lift_joint": 200000.0,
                #     "elbow_joint": 200000.0,
                #     "wrist_1_joint": 200000.0,
                #     "wrist_2_joint": 200000.0,
                #     "wrist_3_joint": 200000.0,
                # },
                # damping = {
                #     "shoulder_pan_joint": 1720.65,
                #     "shoulder_lift_joint": 2592.72,
                #     "elbow_joint": 1364.99,
                #     "wrist_1_joint": 988.70,
                #     "wrist_2_joint": 988.70,
                #     "wrist_3_joint": 388.00,
                # }
                # ############### Stiffness 400000 ###############
                # stiffness = {
                #     "shoulder_pan_joint": 400000.0,
                #     "shoulder_lift_joint": 400000.0,
                #     "elbow_joint": 400000.0,
                #     "wrist_1_joint": 400000.0,
                #     "wrist_2_joint": 400000.0,
                #     "wrist_3_joint": 400000.0,
                # },
                # damping = {
                #     "shoulder_pan_joint": 2433.37,
                #     "shoulder_lift_joint": 3666.66,
                #     "elbow_joint": 1930.38,
                #     "wrist_1_joint": 1398.23,
                #     "wrist_2_joint": 1398.23,
                #     "wrist_3_joint": 549.01,
                # }
                # ############### Stiffness 800000 ###############
                # stiffness={
                #     "shoulder_pan_joint": 800000.0,
                #     "shoulder_lift_joint": 800000.0,
                #     "elbow_joint": 800000.0,
                #     "wrist_1_joint": 800000.0,
                #     "wrist_2_joint": 800000.0,
                #     "wrist_3_joint": 800000.0,
                # },
                # damping={
                #     "shoulder_pan_joint": 3441,
                #     "shoulder_lift_joint": 5185,
                #     "elbow_joint": 2732,
                #     "wrist_1_joint": 1977,
                #     "wrist_2_joint": 1977,
                #     "wrist_3_joint": 776,
                # }
                # ############### Stiffness 1000000 ###############
                # stiffness={
                #     "shoulder_pan_joint": 1000000.0,
                #     "shoulder_lift_joint": 1000000.0,
                #     "elbow_joint": 1000000.0,
                #     "wrist_1_joint": 1000000.0,
                #     "wrist_2_joint": 1000000.0,
                #     "wrist_3_joint": 1000000.0,
                # },
                # damping={
                #     "shoulder_pan_joint": 3847.5,
                #     "shoulder_lift_joint": 5797.5,
                #     "elbow_joint": 3052.2,
                #     "wrist_1_joint": 2210.8,
                #     "wrist_2_joint": 2210.8,
                #     "wrist_3_joint": 868.1,
                # }
                # ############### Stiffness 1200000 ###############
                # stiffness = {
                #     "shoulder_pan_joint": 1200000.0,
                #     "shoulder_lift_joint": 1200000.0,
                #     "elbow_joint": 1200000.0,
                #     "wrist_1_joint": 1200000.0,
                #     "wrist_2_joint": 1200000.0,
                #     "wrist_3_joint": 1200000.0,
                # },
                # damping = {
                #     "shoulder_pan_joint": 4214.73,
                #     "shoulder_lift_joint": 6350.84,
                #     "elbow_joint": 3343.52,
                #     "wrist_1_joint": 2421.81,
                #     "wrist_2_joint": 2421.81,
                #     "wrist_3_joint": 950.99,
                # }
                # ############### Stiffness 1500000 ###############
                # stiffness = {
                #     "shoulder_pan_joint": 1500000.0,
                #     "shoulder_lift_joint": 1500000.0,
                #     "elbow_joint": 1500000.0,
                #     "wrist_1_joint": 1500000.0,
                #     "wrist_2_joint": 1500000.0,
                #     "wrist_3_joint": 1500000.0,
                # },
                # damping = {
                #     "shoulder_pan_joint": 4722.86,
                #     "shoulder_lift_joint": 7116.67,
                #     "elbow_joint": 3743.94,
                #     "wrist_1_joint": 2713.60,
                #     "wrist_2_joint": 2713.60,
                #     "wrist_3_joint": 1065.79,
                # }
                # ############### Stiffness 2000000 ###############
                # stiffness = {
                #     "shoulder_pan_joint": 2000000.0,
                #     "shoulder_lift_joint": 2000000.0,
                #     "elbow_joint": 2000000.0,
                #     "wrist_1_joint": 2000000.0,
                #     "wrist_2_joint": 2000000.0,
                #     "wrist_3_joint": 2000000.0,
                # },
                # damping = {
                #     "shoulder_pan_joint": 5443.58,
                #     "shoulder_lift_joint": 8205.78,
                #     "elbow_joint": 4314.83,
                #     "wrist_1_joint": 3127.89,
                #     "wrist_2_joint": 3127.89,
                #     "wrist_3_joint": 1229.97,
                # }
                # ############### Stiffness 5000000 ###############
                # stiffness = {
                #     "shoulder_pan_joint": 5000000.0,
                #     "shoulder_lift_joint": 5000000.0,
                #     "elbow_joint": 5000000.0,
                #     "wrist_1_joint": 5000000.0,
                #     "wrist_2_joint": 5000000.0,
                #     "wrist_3_joint": 5000000.0,
                # },
                # damping = {
                #     "shoulder_pan_joint": 8591.93,
                #     "shoulder_lift_joint": 12954.54,
                #     "elbow_joint": 6815.50,
                #     "wrist_1_joint": 4937.82,
                #     "wrist_2_joint": 4937.82,
                #     "wrist_3_joint": 1943.25,
                # }
                ############### Stiffness 10000000 ###############
                stiffness = {
                    "shoulder_pan_joint": 10000000.0,
                    "shoulder_lift_joint": 10000000.0,
                    "elbow_joint": 10000000.0,
                    "wrist_1_joint": 10000000.0,
                    "wrist_2_joint": 10000000.0,
                    "wrist_3_joint": 10000000.0,
                },
                damping = {
                    "shoulder_pan_joint": 12166.86,
                    "shoulder_lift_joint": 18333.30,
                    "elbow_joint": 9651.90,
                    "wrist_1_joint": 6991.16,
                    "wrist_2_joint": 6991.16,
                    "wrist_3_joint": 2752.97,
                }
                # ############### Stiffness 50000000 ###############
                # stiffness = {
                #     "shoulder_pan_joint": 50000000.0,
                #     "shoulder_lift_joint": 50000000.0,
                #     "elbow_joint": 50000000.0,
                #     "wrist_1_joint": 50000000.0,
                #     "wrist_2_joint": 50000000.0,
                #     "wrist_3_joint": 50000000.0,
                # },
                # damping = {
                #     "shoulder_pan_joint": 27210.91,
                #     "shoulder_lift_joint": 41031.83,
                #     "elbow_joint": 21614.89,
                #     "wrist_1_joint": 15651.96,
                #     "wrist_2_joint": 15651.96,
                #     "wrist_3_joint": 6146.74,
                # }
                # ############### Stiffness 100000000 ###############
                # stiffness = {
                #     "shoulder_pan_joint": 100000000.0,
                #     "shoulder_lift_joint": 100000000.0,
                #     "elbow_joint": 100000000.0,
                #     "wrist_1_joint": 100000000.0,
                #     "wrist_2_joint": 100000000.0,
                #     "wrist_3_joint": 100000000.0,
                # },
                # damping = {
                #     "shoulder_pan_joint": 38475.00,
                #     "shoulder_lift_joint": 57975.00,
                #     "elbow_joint": 30522.00,
                #     "wrist_1_joint": 22108.00,
                #     "wrist_2_joint": 22108.00,
                #     "wrist_3_joint": 8681.00,
                # }
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
        resampling_time_range=(5.0, 5.0), #(5.0, 5.0), # (15.0, 15.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(-0.15, 0.15),
            pos_y=(0.25, 0.5),
            pos_z=(0.2, 0.5),
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
            params={"gripper_offset": [0.0, 0.0, 0.0], "robot_cfg": SceneEntityCfg("robot", body_names=["wrist_3_link"])},
            # noise=Unoise(n_min=-0.0001, n_max=0.0001), # New training setting
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
            # noise=Unoise(n_min=-0.0001, n_max=0.0001), # New training setting
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
    # reset_robot_joints = EventTerm(
    #     func=mdp.reset_joints_by_scale,
    #     mode="reset",
    #     params={
    #         "position_range": (0.8, 1.2),
    #         "velocity_range": (0.0, 0.0),
    #     },
    # )
    # rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )
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
        params={"gripper_offset": [0.0, 0.0, 0.0], "asset_cfg": SceneEntityCfg("robot", body_names=["wrist_3_link"]), "command_name": "ee_pose"},
    )
    end_effector_position_tracking_fine_grained = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.1,
        params={"gripper_offset": [0.0, 0.0, 0.0], "asset_cfg": SceneEntityCfg("robot", body_names=["wrist_3_link"]), "std": 0.1, "command_name": "ee_pose"},
    )
    end_effector_orientation_tracking = RewTerm(
        func=mdp.orientation_command_error,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["wrist_3_link"]), "command_name": "ee_pose"},
    )


    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2_position, weight=-1e-4)

    # action_magnitude = RewTerm(func=mdp.action_l2_position, weight=-1e-4)

    # joint_vel = RewTerm(
    #     func=mdp.joint_vel_l2,
    #     weight=-1e-4,
    #     params={"asset_cfg": SceneEntityCfg("robot")},
    # )

    # action_clip = RewTerm(
    #     func=mdp.action_clip, 
    #     weight=-0.05,
    #     params={"pos_threshold": 1.0, "axis_angle_threshold": 2.0}) # "pos_threshold": 0.05, "axis_angle_threshold": 0.08})

    # ee_acc = RewTerm(
    #     func=mdp.body_lin_acc_l2,
    #     weight=-1e-4,
    #     params={"asset_cfg": SceneEntityCfg("robot", body_names=["wrist_3_link"]),}
    # )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)



@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -0.02, "num_steps": 16000} #15000 #4500
    )

    # action_magnitude = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "action_magnitude", "weight": -0.02, "num_steps": 20000} #15000 #4500
    # )

    # action_rate_v2 = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -0.05, "num_steps": 40000} #15000 #4500
    # )

    # action_magnitude_v2 = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "action_magnitude", "weight": -0.05, "num_steps": 40000} #15000 #4500
    # )

    # joint_vel = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -0.01, "num_steps": 4500}
    # )

    # ee_acc = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "ee_acc", "weight": -0.001, "num_steps": 20000} #4500
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
        self.decimation = 2 # 4 # 50 # 2
        self.episode_length_s = 15.0
        # simulation settings
        self.sim.dt = 0.01 #1/60
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.physx.gpu_collision_stack_size = 4096 * 4096 * 100 # Was added due to an PhysX error: collisionStackSize buffer overflow detected




        # self.sim.physx.solver_type=1,
        # self.sim.physx.max_position_iteration_count=192,  # Important to avoid interpenetration.
        # self.sim.physx.max_velocity_iteration_count=1,
        # self.sim.physx.bounce_threshold_velocity=0.2,
        # self.sim.physx.friction_offset_threshold=0.01,
        # self.sim.physx.friction_correlation_distance=0.00625,
        # self.sim.physx.gpu_max_rigid_contact_count=2**23,
        # self.sim.physx.gpu_max_rigid_patch_count=2**23,
        # self.sim.physx.gpu_max_num_partitions=1,  # Important for stable simulation.