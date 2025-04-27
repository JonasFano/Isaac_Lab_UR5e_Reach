# Isaac_Lab_UR5e_Reach
Utilize Reinforcement Learning in Isaac Lab using the UR5e to reach desired target poses with differential IK control. 



#######################
# Relative IK Control #
#######################

# Stable-baselines3 - UR5e
    source isaaclab/bin/activate
    cd isaaclab/IsaacLab
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/train_sb3_ppo.py --num_envs 1 --task UR5e-Reach-IK --headless --no_logging
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/train_sb3_ppo.py --num_envs 4096 --task UR5e-Reach-Pose-IK --headless
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/train_sb3_ppo.py --num_envs 1 --task UR5e-Reach-Pose-IK --no_logging


    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo.py --task UR5e-Reach-Pose-IK --num_envs 4 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/logs/sb3/ppo/UR5e-Reach-Pose-IK/2024-12-13_16-04-26/model.zip


# Hand E training runs
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo.py --task UR5e-Reach-Pose-IK --num_envs 4 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e/tnecdfk7/model.zip

    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo.py --task UR5e-Reach-Pose-IK --num_envs 4 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e/3crumnxy/model.zip

    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo.py --task UR5e-Reach-Pose-IK --num_envs 4 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e/mirz9884/model.zip



### rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e: Actuator stiffness: 800 - Unoise: 0.0 - Robot Reset: "position_range" (1.0, 1.0) - Pose Generation: pos_x=(-0.05, 0.05), pos_y=(0.35, 0.45), pos_z=(0.25, 0.35) - Decimation 2 - Dt 0.01
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo.py --task UR5e-Reach-Pose-IK --num_envs 4 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e/cccnto37/model.zip


### rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e_final: Actuator stiffness: 800 - Unoise: 0.001 - Robot Reset: "position_range" (0.5, 1.5) - Pose Generation: pos_x=(-0.2, 0.2), pos_y=(0.25, 0.5), pos_z=(0.1, 0.4) - Decimation 2 - Dt 0.01
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo.py --task UR5e-Reach-Pose-IK --num_envs 4 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e_final/yrvkdbup/model.zip


### rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e_final_v2: Actuator stiffness: 800 - Unoise: 0.0001 - Robot Reset: "position_range" (0.5, 1.5) - Pose Generation: pos_x=(-0.15, 0.15), pos_y=(0.25, 0.5), pos_z=(0.1, 0.4) - Decimation 2 - Dt 0.01
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo.py --task UR5e-Reach-Pose-IK --num_envs 4 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e_final_v2/tzb7dro2/model.zip


### rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e_final_v3: Actuator stiffness: 800 - Unoise: 0.0001 - Robot Reset: "position_range" (0.8, 1.2) - Pose Generation: pos_x=(-0.15, 0.15), pos_y=(0.25, 0.5), pos_z=(0.1, 0.4) - Decimation 2 - Dt 0.01
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo.py --task UR5e-Reach-Pose-IK --num_envs 4 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e_final_v3/m3itgft1/model.zip


### rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e_final_v4: Actuator stiffness: 800 - Unoise: 0.0001 - Robot Reset: "position_range" (0.7, 1.3) - Pose Generation: pos_x=(-0.15, 0.15), pos_y=(0.25, 0.5), pos_z=(0.1, 0.4) - Decimation 2 - Dt 0.01
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo.py --task UR5e-Reach-Pose-IK --num_envs 4 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e_final_v4/lyrdk1rh/model.zip


### rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e_domain_rand: Actuator stiffness: 800 - Unoise: 0.0001 - Robot Reset: "position_range" (0.8, 1.2) - Pose Generation: pos_x=(-0.15, 0.15), pos_y=(0.25, 0.5), pos_z=(0.1, 0.4) - Actuator randomization: "stiffness_distribution_params": (0.8, 1.2), "damping_distribution_params": (0.8, 1.2) - Decimation 2 - Dt 0.01
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo.py --task UR5e-Domain-Rand-Reach-Pose-IK --num_envs 4 --checkpoint 


### rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e_domain_rand_v2: Actuator stiffness: 800 - Unoise: 0.0001 - Robot Reset: "position_range" (0.8, 1.2) - Pose Generation: pos_x=(-0.15, 0.15), pos_y=(0.25, 0.5), pos_z=(0.1, 0.4) - Actuator randomization: "stiffness_distribution_params": (0.7, 1.3), "damping_distribution_params": (0.7, 1.3) - Decimation 2 - Dt 0.01
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo.py --task UR5e-Domain-Rand-Reach-Pose-IK --num_envs 4 --checkpoint 
    

### rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e_decimation_4: Actuator stiffness: 800 - Unoise: 0.0001 - Robot Reset: "position_range" (0.8, 1.2) - Pose Generation: pos_x=(-0.15, 0.15), pos_y=(0.25, 0.5), pos_z=(0.1, 0.4) - Decimation 4 - Dt 0.01
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo.py --task UR5e-Reach-Pose-IK --num_envs 4 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e_decimation_4/vp0dv3v3/model.zip
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo_save_observations.py --task UR5e-Reach-Pose-IK --num_envs 1 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e_decimation_4/vp0dv3v3/model.zip


### rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e_penalize_joint_vel: Actuator stiffness: 800 - Unoise: 0.0001 - Robot Reset: "position_range" (0.8, 1.2) - Pose Generation: pos_x=(-0.15, 0.15), pos_y=(0.25, 0.5), pos_z=(0.1, 0.4) - Decimation 2 - Dt 0.01 - Added penalty for joint velocity
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo.py --task UR5e-Reach-Pose-IK --num_envs 4 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e_penalize_joint_vel/bvwa3dka/model.zip
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo_save_observations.py --task UR5e-Reach-Pose-IK --num_envs 1 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e_penalize_joint_vel/bvwa3dka/model.zip


### rel_ik_sb3_ppo_ur5e_reach_0_1_pose_hand_e_penalize_joint_vel: Actuator stiffness: 800 - Unoise: 0.0001 - Robot Reset: "position_range" (0.8, 1.2) - Pose Generation: pos_x=(-0.15, 0.15), pos_y=(0.25, 0.5), pos_z=(0.1, 0.4) - Decimation 2 - Dt 0.01 - Added penalty for joint velocity
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo.py --task UR5e-Reach-Pose-IK --num_envs 4 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/rel_ik_sb3_ppo_ur5e_reach_0_1_pose_hand_e_penalize_joint_vel/86bnjuiw/model.zip
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo_save_observations.py --task UR5e-Reach-Pose-IK --num_envs 1 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/rel_ik_sb3_ppo_ur5e_reach_0_1_pose_hand_e_penalize_joint_vel/86bnjuiw/model.zip


### rel_ik_sb3_ppo_ur5e_reach_0_1_pose_hand_e_penalize_ee_acc: Actuator stiffness: 800 - Unoise: 0.0001 - Robot Reset: "position_range" (0.8, 1.2) - Pose Generation: pos_x=(-0.15, 0.15), pos_y=(0.25, 0.5), pos_z=(0.1, 0.4) - Decimation 2 - Dt 0.01 - Added penalty for end-effector acceleration - Penalty weight: 0.001
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo.py --task UR5e-Reach-Pose-IK --num_envs 4 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/rel_ik_sb3_ppo_ur5e_reach_0_1_pose_hand_e_penalize_ee_acc/nrxve995/model.zip
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo_save_observations.py --task UR5e-Reach-Pose-IK --num_envs 1 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/rel_ik_sb3_ppo_ur5e_reach_0_1_pose_hand_e_penalize_ee_acc/nrxve995/model.zip


### rel_ik_sb3_ppo_ur5e_reach_0_1_pose_hand_e_penalize_ee_acc_v2: Actuator stiffness: 800 - Unoise: 0.0001 - Robot Reset: "position_range" (0.8, 1.2) - Pose Generation: pos_x=(-0.15, 0.15), pos_y=(0.25, 0.5), pos_z=(0.1, 0.4) - Decimation 2 - Dt 0.01 - Added penalty for end-effector acceleration - Penalty weight: 0.01
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo.py --task UR5e-Reach-Pose-IK --num_envs 4 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/rel_ik_sb3_ppo_ur5e_reach_0_1_pose_hand_e_penalize_ee_acc_v2/j90cbcg2/model.zip
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo_save_observations.py --task UR5e-Reach-Pose-IK --num_envs 1 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/rel_ik_sb3_ppo_ur5e_reach_0_1_pose_hand_e_penalize_ee_acc_v2/j90cbcg2/model.zip


### rel_ik_sb3_ppo_ur5e_reach_0_1_pose_hand_e_penalize_ee_acc_v3: Actuator stiffness: 800 - Unoise: 0.0001 - Robot Reset: "position_range" (0.8, 1.2) - Pose Generation: pos_x=(-0.15, 0.15), pos_y=(0.25, 0.5), pos_z=(0.1, 0.4) - Decimation 2 - Dt 0.01 - Added penalty for end-effector acceleration - Penalty weight: 0.005
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo.py --task UR5e-Reach-Pose-IK --num_envs 4 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/rel_ik_sb3_ppo_ur5e_reach_0_1_pose_hand_e_penalize_ee_acc_v3/0ct6f5jv/model.zip
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo_save_observations.py --task UR5e-Reach-Pose-IK --num_envs 1 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/rel_ik_sb3_ppo_ur5e_reach_0_1_pose_hand_e_penalize_ee_acc_v3/0ct6f5jv/model.zip


### rel_ik_sb3_ppo_ur5e_reach_0_1_pose_hand_e_penalize_ee_acc_v4: Actuator stiffness: 800 - Unoise: 0.0001 - Robot Reset: "position_range" (0.8, 1.2) - Pose Generation: pos_x=(-0.15, 0.15), pos_y=(0.25, 0.5), pos_z=(0.1, 0.4) - Decimation 2 - Dt 0.01 - Added penalty for end-effector acceleration and joint velocity - Both penalty weights: 0.001
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo.py --task UR5e-Reach-Pose-IK --num_envs 4 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/rel_ik_sb3_ppo_ur5e_reach_0_1_pose_hand_e_penalize_ee_acc_v4/iepxkc9t/model.zip
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo_save_observations.py --task UR5e-Reach-Pose-IK --num_envs 1 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/rel_ik_sb3_ppo_ur5e_reach_0_1_pose_hand_e_penalize_ee_acc_v4/iepxkc9t/model.zip


### rel_ik_sb3_ppo_ur5e_reach_0_01_pose_hand_e_penalize_ee_acc_v5: Actuator stiffness: 800 - Unoise: 0.0001 - Robot Reset: "position_range" (0.8, 1.2) - Pose Generation: pos_x=(-0.15, 0.15), pos_y=(0.25, 0.5), pos_z=(0.1, 0.4) - Decimation 2 - Dt 0.01 - Added penalty for end-effector acceleration - Penalty weight: 0.001
    Negative reward - poor performance - Too small action scaling


### rel_ik_sb3_ppo_ur5e_reach_0_01_pose_hand_e_penalize_ee_acc_v6: Actuator stiffness: 800 - Unoise: 0.0001 - Robot Reset: "position_range" (0.8, 1.2) - Pose Generation: pos_x=(-0.15, 0.15), pos_y=(0.25, 0.5), pos_z=(0.1, 0.4) - Decimation 2 - Dt 0.01 - Added penalty for end-effector acceleration and joint velocity - Both penalty weights: 0.01
    Negative reward - poor performance - Too small action scaling


# New gripper Robotiq Hand E UR5e
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo.py --task UR5e-Hand-E-Reach-Pose-Abs-IK --num_envs 4 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/abs_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e_v3/9gkgww2n/model.zip


# UR5e Record data Rel IK
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo_save_observations.py --task UR5e-Reach-Pose-IK --num_envs 1 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/rel_ik_sb3_ppo_ur5e_reach_0_1_pose_hand_e_penalize_ee_acc_v4/iepxkc9t/model.zip


## UR3e
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo.py --task UR3e-Reach-Pose-IK --num_envs 1 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/rel_ik_sb3_ppo_ur5e_reach_0_1_pose_hand_e_penalize_ee_acc_v2/j90cbcg2/model.zip

## UR3e Record Data
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo_save_observations.py --task UR3e-Reach-Pose-IK --num_envs 1 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/rel_ik_sb3_ppo_ur5e_reach_0_1_pose_hand_e_penalize_ee_acc_v2/j90cbcg2/model.zip



# Tensorboard
    tensorboard --logdir='directory'



### rel_ik_sb3_ppo_ur5e_reach_0_05_pose_without_gripper - without action penalties
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo_save_observations.py --task UR5e-Reach-Pose-IK --num_envs 1 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/ppo_parameter_optimization/relative_vs_absolute/01gt11w7/model.zip


### rel_ik_sb3_ppo_ur5e_reach_0_05_pose_without_gripper_action_rate_pos_0_1_step_16000
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo_save_observations.py --task UR5e-Reach-Pose-IK --num_envs 1 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/ppo_parameter_optimization/action_rate_pos_penalty_0_1_step_16000/myqhu6i5/model.zip

### rel_ik_sb3_ppo_ur5e_reach_0_05_pose_without_gripper_action_rate_pos_1_0_step_16000
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo_save_observations.py --task UR5e-Reach-Pose-IK --num_envs 1 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/ppo_parameter_optimization/action_rate_pos_penalty_1_0_step_16000/4onkm2st/model.zip

### rel_ik_sb3_ppo_ur5e_reach_0_05_pose_without_gripper_action_rate_1_0_step_16000
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo_save_observations.py --task UR5e-Reach-Pose-IK --num_envs 1 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/ppo_parameter_optimization/action_rate_penalty_1_0_step_16000/m649vcbd/model.zip

### rel_ik_sb3_ppo_ur5e_reach_0_05_pose_without_gripper_action_magnitude_pos_0_8_step_16000
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo_save_observations.py --task UR5e-Reach-Pose-IK --num_envs 1 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/ppo_parameter_optimization/action_magnitude_pos_penalty_0_8_step_16000/eqx4kgv4/model.zip

### rel_ik_sb3_ppo_ur5e_reach_0_05_pose_without_gripper_action_magnitude_0_05_step_16000
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo_save_observations.py --task UR5e-Reach-Pose-IK --num_envs 1 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/ppo_parameter_optimization/action_magnitude_penalty_0_05_step_16000/gsffkw68/model.zip

### rel_ik_sb3_ppo_ur5e_reach_0_05_pose_without_gripper_action_magnitude_0_01_step_16000
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo_save_observations.py --task UR5e-Reach-Pose-IK --num_envs 1 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/ppo_parameter_optimization/action_magnitude_penalty_0_01_step_16000/but8nbmx/model.zip

### rel_ik_sb3_ppo_ur5e_reach_0_05_pose_without_gripper_action_rate_0_1_action_magnitude_0_01_step_16000
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo_save_observations.py --task UR5e-Reach-Pose-IK --num_envs 1 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/ppo_parameter_optimization/combined_action_rate_0_1_action_magnitude_0_01_penalty/kupdnkqo/model.zip

### rel_ik_sb3_ppo_ur5e_reach_0_05_pose_without_gripper_action_rate_0_5_action_magnitude_0_02_step_16000
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo_save_observations.py --task UR5e-Reach-Pose-IK --num_envs 1 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/ppo_parameter_optimization/combined_action_rate_0_5_action_magnitude_0_02_penalty/i6vjdnxj/model.zip








### rel_ik_sb3_ppo_ur5e_reach_0_05_pose_without_gripper - without action penalties
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo.py --task UR5e-Reach-Pose-IK --num_envs 1000 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/ppo_parameter_optimization/relative_vs_absolute/01gt11w7/model.zip --headless


### rel_ik_sb3_ppo_ur5e_reach_0_05_pose_without_gripper_action_rate_pos_0_1_step_16000
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo.py --task UR5e-Reach-Pose-IK --num_envs 1000 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/ppo_parameter_optimization/action_rate_pos_penalty_0_1_step_16000/myqhu6i5/model.zip --headless

### rel_ik_sb3_ppo_ur5e_reach_0_05_pose_without_gripper_action_rate_pos_1_0_step_16000
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo.py --task UR5e-Reach-Pose-IK --num_envs 1000 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/ppo_parameter_optimization/action_rate_pos_penalty_1_0_step_16000/4onkm2st/model.zip --headless

### rel_ik_sb3_ppo_ur5e_reach_0_05_pose_without_gripper_action_rate_1_0_step_16000
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo.py --task UR5e-Reach-Pose-IK --num_envs 1000 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/ppo_parameter_optimization/action_rate_penalty_1_0_step_16000/m649vcbd/model.zip --headless

### rel_ik_sb3_ppo_ur5e_reach_0_05_pose_without_gripper_action_magnitude_pos_0_8_step_16000
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo.py --task UR5e-Reach-Pose-IK --num_envs 1000 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/ppo_parameter_optimization/action_magnitude_pos_penalty_0_8_step_16000/eqx4kgv4/model.zip --headless

### rel_ik_sb3_ppo_ur5e_reach_0_05_pose_without_gripper_action_magnitude_0_05_step_16000
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo.py --task UR5e-Reach-Pose-IK --num_envs 1000 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/ppo_parameter_optimization/action_magnitude_penalty_0_05_step_16000/gsffkw68/model.zip --headless

### rel_ik_sb3_ppo_ur5e_reach_0_05_pose_without_gripper_action_magnitude_0_01_step_16000
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo.py --task UR5e-Reach-Pose-IK --num_envs 1000 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/ppo_parameter_optimization/action_magnitude_penalty_0_01_step_16000/but8nbmx/model.zip --headless

### rel_ik_sb3_ppo_ur5e_reach_0_05_pose_without_gripper_action_rate_0_1_action_magnitude_0_01_step_16000
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo.py --task UR5e-Reach-Pose-IK --num_envs 1000 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/ppo_parameter_optimization/combined_action_rate_0_1_action_magnitude_0_01_penalty/kupdnkqo/model.zip --headless

### rel_ik_sb3_ppo_ur5e_reach_0_05_pose_without_gripper_action_rate_0_5_action_magnitude_0_02_step_16000
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo.py --task UR5e-Reach-Pose-IK --num_envs 1000 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/ppo_parameter_optimization/combined_action_rate_0_5_action_magnitude_0_02_penalty/i6vjdnxj/model.zip --headless




#######################
# IK Absolute Control #
#######################

# Stable-baselines3 - UR5e
    source isaaclab/bin/activate
    cd isaaclab/IsaacLab
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/train_sb3_ppo.py --num_envs 2048 --task UR5e-Reach-Pose-Abs-IK --headless


# Play
    source isaaclab/bin/activate
    cd isaaclab/IsaacLab
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo.py --task UR5e-Reach-Pose-Abs-IK --num_envs 1 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/logs/sb3/ppo/UR5e-Reach-Pose-Abs-IK/2024-12-07_21-18-23/model.zip



##########################
# Joint Position Control #
##########################

# Stable-baselines3 - UR5e
    source isaaclab/bin/activate
    cd isaaclab/IsaacLab
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/train_sb3_ppo.py --num_envs 8192 --task UR5e-Reach --headless

    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo.py --task UR5e-Reach --num_envs 4 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/logs/sb3/ppo/UR5e-Reach/2024-11-26_07-31-34/model_204800000_steps.zip




######################
# Weights and Biases #
######################

# REL IK UR5e Wandb PPO
    source isaaclab/bin/activate
    cd /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3
    wandb sweep --project rel_ik_sb3_ppo_ur5e_reach_0_05_pose config_sb3_ppo.yaml

    wandb sweep --project rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e_decimation_4 config_sb3_ppo.yaml

    wandb sweep --project rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e_penalize_joint_vel config_sb3_ppo.yaml
    wandb sweep --project rel_ik_sb3_ppo_ur5e_reach_0_1_pose_hand_e_penalize_joint_vel config_sb3_ppo.yaml
    wandb sweep --project rel_ik_sb3_ppo_ur5e_reach_0_1_pose_hand_e_penalize_ee_acc config_sb3_ppo.yaml
    wandb sweep --project rel_ik_sb3_ppo_ur5e_reach_0_1_pose_hand_e_penalize_ee_acc_v2 config_sb3_ppo.yaml
    wandb sweep --project rel_ik_sb3_ppo_ur5e_reach_0_1_pose_hand_e_penalize_ee_acc_v3 config_sb3_ppo.yaml
    wandb sweep --project rel_ik_sb3_ppo_ur5e_reach_0_1_pose_hand_e_penalize_ee_acc_v4 config_sb3_ppo.yaml
    wandb sweep --project rel_ik_sb3_ppo_ur5e_reach_0_01_pose_hand_e_penalize_ee_acc_v5 config_sb3_ppo.yaml
    wandb sweep --project rel_ik_sb3_ppo_ur5e_reach_0_01_pose_hand_e_penalize_ee_acc_v6 config_sb3_ppo.yaml
    wandb sweep --project rel_ik_sb3_ppo_ur5e_reach_0_01_pose_hand_e_penalize_ee_acc_v7 config_sb3_ppo.yaml


    wandb sweep --project rel_ik_sb3_ppo_ur5e_reach_0_05_pose_without_gripper_parameter_optimization config_sb3_ppo.yaml
    wandb sweep --project rel_ik_sb3_ppo_ur5e_reach_0_05_pose_without_gripper_parameter_optimization_800 config_sb3_ppo.yaml


    

# REL IK UR5e Wandb PPO with domain randomization
    source isaaclab/bin/activate
    cd /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3

    wandb sweep --project rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e_domain_rand config_sb3_ppo_domain_rand.yaml

    wandb sweep --project rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e_domain_rand_v2 config_sb3_ppo_domain_rand.yaml


# Abs IK UR5e Wandb PPO
    source isaaclab/bin/activate
    cd /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3
    wandb sweep --project abs_ik_sb3_ppo_ur5e_reach_0_05_pose_without_gripper config_sb3_ppo.yaml

### Absolute Mode
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo.py --task UR5e-Reach-Pose-Abs-IK --num_envs 4 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/ppo_parameter_optimization/relative_vs_absolute/f414sb65/model.zip



### Normalize obs and reward 
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo.py --task UR5e-Reach-Pose-IK --num_envs 4 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/ppo_parameter_optimization/group_g/2zlcpe8a/model.zip

### Normalize obs and do not normalize reward
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo.py --task UR5e-Reach-Pose-IK --num_envs 4 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/ppo_parameter_optimization/group_f/9nutoono/model.zip

### Do not normalize both obs and reward
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo.py --task UR5e-Reach-Pose-IK --num_envs 4 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/ppo_parameter_optimization/group_b/q229e8ps/model.zip





# UR5e Wandb TD3
    source isaaclab/bin/activate
    cd /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3
    wandb sweep --project rel_ik_sb3_td3_ur5e_reach_0_05_pose config_sb3_td3.yaml

    wandb sweep --project rel_ik_sb3_td3_ur5e_reach_0_05_pose_grid_search config_sb3_td3.yaml



# UR5e Wandb DDPG
    source isaaclab/bin/activate
    cd /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3
    wandb sweep --project rel_ik_sb3_ddpg_ur5e_reach_0_05_pose_grid_search config_sb3_ddpg.yaml


# Remote connection
ssh -I 10.178.107.200 jofa@ia-li-2wqd414.unirobotts.local


# Change ip address
ip link 
sudo ifconfig enp0s31f6 192.168.1.101 up

# Run test_rtde.py
python3 test_rtde.py