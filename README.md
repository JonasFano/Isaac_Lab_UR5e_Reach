# Isaac_Lab_UR5e_Reach
Utilize Reinforcement Learning in Isaac Lab using the UR5e to reach desired target poses with differential IK control. 



###########################
# Rel IK Relative Control #
###########################

# Stable-baselines3 - UR5e
    source isaaclab/bin/activate
    cd isaaclab/IsaacLab
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/train_sb3_ppo.py --num_envs 1 --task UR5e-Reach-IK --headless --no_logging
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/train_sb3_ppo.py --num_envs 4096 --task UR5e-Reach-Pose-IK --headless


    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo.py --task UR5e-Reach-Pose-IK --num_envs 4 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/logs/sb3/ppo/UR5e-Reach-Pose-IK/2024-12-13_16-04-26/model.zip


# Hand E training runs
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo.py --task UR5e-Reach-Pose-IK --num_envs 4 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e/tnecdfk7/model.zip

    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo.py --task UR5e-Reach-Pose-IK --num_envs 4 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e/3crumnxy/model.zip

    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo.py --task UR5e-Reach-Pose-IK --num_envs 4 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e/mirz9884/model.zip



### rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e: Unoise: 0.0 - Robot Reset: "position_range" (1.0, 1.0) - Pose Generation: pos_x=(-0.05, 0.05), pos_y=(0.35, 0.45), pos_z=(0.25, 0.35)
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo.py --task UR5e-Reach-Pose-IK --num_envs 4 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e/cccnto37/model.zip


### rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e_final: Unoise: 0.001 - Robot Reset: "position_range" (0.5, 1.5) - Pose Generation: pos_x=(-0.2, 0.2), pos_y=(0.25, 0.5), pos_z=(0.1, 0.4)
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo.py --task UR5e-Reach-Pose-IK --num_envs 4 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e_final/yrvkdbup/model.zip


### rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e_final_v2: Unoise: 0.0001 - Robot Reset: "position_range" (0.5, 1.5) - Pose Generation: pos_x=(-0.15, 0.15), pos_y=(0.25, 0.5), pos_z=(0.1, 0.4)
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo.py --task UR5e-Reach-Pose-IK --num_envs 4 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e_final_v2/tzb7dro2/model.zip


### rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e_final_v3: Unoise: 0.0001 - Robot Reset: "position_range" (0.8, 1.2) - Pose Generation: pos_x=(-0.15, 0.15), pos_y=(0.25, 0.5), pos_z=(0.1, 0.4)
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo.py --task UR5e-Reach-Pose-IK --num_envs 4 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e_final_v3/m3itgft1/model.zip


### rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e_final_v4: Unoise: 0.0001 - Robot Reset: "position_range" (0.7, 1.3) - Pose Generation: pos_x=(-0.15, 0.15), pos_y=(0.25, 0.5), pos_z=(0.1, 0.4)
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo.py --task UR5e-Reach-Pose-IK --num_envs 4 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e_final_v4/lyrdk1rh/model.zip


### rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e_domain_rand: Unoise: 0.0001 - Robot Reset: "position_range" (0.8, 1.2) - Pose Generation: pos_x=(-0.15, 0.15), pos_y=(0.25, 0.5), pos_z=(0.1, 0.4) - Actuator randomization: "stiffness_distribution_params": (0.8, 1.2), "damping_distribution_params": (0.8, 1.2)
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo.py --task UR5e-Domain-Rand-Reach-Pose-IK --num_envs 4 --checkpoint 


### rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e_domain_rand_v2: Unoise: 0.0001 - Robot Reset: "position_range" (0.8, 1.2) - Pose Generation: pos_x=(-0.15, 0.15), pos_y=(0.25, 0.5), pos_z=(0.1, 0.4) - Actuator randomization: "stiffness_distribution_params": (0.7, 1.3), "damping_distribution_params": (0.7, 1.3)


### rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e_decimation_4: Unoise: 0.0001 - Robot Reset: "position_range" (0.8, 1.2) - Pose Generation: pos_x=(-0.15, 0.15), pos_y=(0.25, 0.5), pos_z=(0.1, 0.4) - Decimation 4 - Dt 0.01
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo.py --task UR5e-Reach-Pose-IK --num_envs 4 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e_decimation_4/vp0dv3v3/model.zip


### rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e_penalize_joint_vel: Unoise: 0.0001 - Robot Reset: "position_range" (0.8, 1.2) - Pose Generation: pos_x=(-0.15, 0.15), pos_y=(0.25, 0.5), pos_z=(0.1, 0.4) - Decimation 2 - Dt 0.01 - Added penalty for joint velocity
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo.py --task UR5e-Reach-Pose-IK --num_envs 4 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e_penalize_joint_vel/bvwa3dka/model.zip


### rel_ik_sb3_ppo_ur5e_reach_0_1_pose_hand_e_penalize_joint_vel: Unoise: 0.0001 - Robot Reset: "position_range" (0.8, 1.2) - Pose Generation: pos_x=(-0.15, 0.15), pos_y=(0.25, 0.5), pos_z=(0.1, 0.4) - Decimation 2 - Dt 0.01 - Added penalty for joint velocity
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo.py --task UR5e-Reach-Pose-IK --num_envs 4 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/rel_ik_sb3_ppo_ur5e_reach_0_1_pose_hand_e_penalize_joint_vel/86bnjuiw/model.zip

### rel_ik_sb3_ppo_ur5e_reach_0_1_pose_hand_e_penalize_ee_acc: Unoise: 0.0001 - Robot Reset: "position_range" (0.8, 1.2) - Pose Generation: pos_x=(-0.15, 0.15), pos_y=(0.25, 0.5), pos_z=(0.1, 0.4) - Decimation 2 - Dt 0.01 - Added penalty for end-effector acceleration - Penalty weight: 0.001
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo.py --task UR5e-Reach-Pose-IK --num_envs 4 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/rel_ik_sb3_ppo_ur5e_reach_0_1_pose_hand_e_penalize_ee_acc/nrxve995/model.zip


### rel_ik_sb3_ppo_ur5e_reach_0_1_pose_hand_e_penalize_ee_acc_v2: Unoise: 0.0001 - Robot Reset: "position_range" (0.8, 1.2) - Pose Generation: pos_x=(-0.15, 0.15), pos_y=(0.25, 0.5), pos_z=(0.1, 0.4) - Decimation 2 - Dt 0.01 - Added penalty for end-effector acceleration - Penalty weight: 0.01
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo.py --task UR5e-Reach-Pose-IK --num_envs 4 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/rel_ik_sb3_ppo_ur5e_reach_0_1_pose_hand_e_penalize_ee_acc_v2/j90cbcg2/model.zip


### rel_ik_sb3_ppo_ur5e_reach_0_1_pose_hand_e_penalize_ee_acc_v3: Unoise: 0.0001 - Robot Reset: "position_range" (0.8, 1.2) - Pose Generation: pos_x=(-0.15, 0.15), pos_y=(0.25, 0.5), pos_z=(0.1, 0.4) - Decimation 2 - Dt 0.01 - Added penalty for end-effector acceleration - Penalty weight: 0.005
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo.py --task UR5e-Reach-Pose-IK --num_envs 4 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/rel_ik_sb3_ppo_ur5e_reach_0_1_pose_hand_e_penalize_ee_acc_v3/0ct6f5jv/model.zip


### rel_ik_sb3_ppo_ur5e_reach_0_1_pose_hand_e_penalize_ee_acc_v4: Unoise: 0.0001 - Robot Reset: "position_range" (0.8, 1.2) - Pose Generation: pos_x=(-0.15, 0.15), pos_y=(0.25, 0.5), pos_z=(0.1, 0.4) - Decimation 2 - Dt 0.01 - Added penalty for end-effector acceleration and joint velocity - Both penalty weights: 0.001
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo.py --task UR5e-Reach-Pose-IK --num_envs 4 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/rel_ik_sb3_ppo_ur5e_reach_0_1_pose_hand_e_penalize_ee_acc_v4/iepxkc9t/model.zip


### rel_ik_sb3_ppo_ur5e_reach_0_01_pose_hand_e_penalize_ee_acc_v5: Unoise: 0.0001 - Robot Reset: "position_range" (0.8, 1.2) - Pose Generation: pos_x=(-0.15, 0.15), pos_y=(0.25, 0.5), pos_z=(0.1, 0.4) - Decimation 2 - Dt 0.01 - Added penalty for end-effector acceleration - Penalty weight: 0.001
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo.py --task UR5e-Reach-Pose-IK --num_envs 4 --checkpoint 


### rel_ik_sb3_ppo_ur5e_reach_0_01_pose_hand_e_penalize_ee_acc_v6: Unoise: 0.0001 - Robot Reset: "position_range" (0.8, 1.2) - Pose Generation: pos_x=(-0.15, 0.15), pos_y=(0.25, 0.5), pos_z=(0.1, 0.4) - Decimation 2 - Dt 0.01 - Added penalty for end-effector acceleration and joint velocity - Both penalty weights: 0.01
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo.py --task UR5e-Reach-Pose-IK --num_envs 4 --checkpoint 



# UR5e Record data
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo_save_observations.py --task UR5e-Reach-Pose-IK --num_envs 1 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/rel_ik_sb3_ppo_ur5e_reach_0_1_pose_hand_e_penalize_ee_acc_v4/iepxkc9t/model.zip


## UR3e
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo.py --task UR3e-Reach-Pose-IK --num_envs 4 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/rel_ik_sb3_ppo_ur5e_reach_0_1_pose_hand_e_penalize_ee_acc_v2/j90cbcg2/model.zip

## UR3e Record Data
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo_save_observations.py --task UR3e-Reach-Pose-IK --num_envs 1 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/rel_ik_sb3_ppo_ur5e_reach_0_1_pose_hand_e_penalize_ee_acc_v2/j90cbcg2/model.zip






# SAC
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/train_sb3_sac.py --num_envs 4096 --task UR5e-Reach-Pose-IK --headless

    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_sac.py --task UR5e-Reach-Pose-IK --num_envs 4 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/logs/sb3/sac/UR5e-Reach-Pose-IK/HerReplayBuffer/model.zip


# Tensorboard
    tensorboard --logdir='directory'




#######################
# IK Absolute Control #
#######################

# Stable-baselines3 - UR5e
    source isaaclab/bin/activate
    cd isaaclab/IsaacLab
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/train_sb3_ppo.py --num_envs 8192 --task UR5e-Reach-Pose-Abs-IK --headless





##########################
# Joint Position Control #
##########################

# Stable-baselines3 - UR5e
    source isaaclab/bin/activate
    cd isaaclab/IsaacLab
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/train_sb3_ppo.py --num_envs 8192 --task UR5e-Reach --headless

    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo.py --task UR5e-Reach --num_envs 4 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/logs/sb3/ppo/UR5e-Reach/2024-11-26_07-31-34/model_204800000_steps.zip


# UR5e Wandb PPO
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

    

# UR5e Wandb PPO with domain randomization
    source isaaclab/bin/activate
    cd /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3

    wandb sweep --project rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e_domain_rand config_sb3_ppo_domain_rand.yaml

    wandb sweep --project rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e_domain_rand_v2 config_sb3_ppo_domain_rand.yaml




# UR5e Wandb SAC
    source isaaclab/bin/activate
    cd /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3
    wandb sweep --project rel_ik_sb3_sac_ur5e_reach_0_05_pose config_sb3_sac.yaml
    wandb sweep --project rel_ik_sb3_sac_ur5e_reach_0_05_pose_2 config_sb3_sac.yaml
    wandb sweep --project rel_ik_sb3_sac_ur5e_reach_0_05_pose_3 config_sb3_sac.yaml
    wandb sweep --project rel_ik_sb3_sac_ur5e_reach_0_05_pose_4 config_sb3_sac.yaml



# UR5e Wandb TD3
    source isaaclab/bin/activate
    cd /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3
    wandb sweep --project rel_ik_sb3_td3_ur5e_reach_0_05_pose config_sb3_td3.yaml

    wandb sweep --project rel_ik_sb3_td3_ur5e_reach_0_05_pose_bayes config_sb3_td3.yaml



# UR5e Wandb DDPG
    source isaaclab/bin/activate
    cd /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3
    wandb sweep --project rel_ik_sb3_ddpg_ur5e_reach_0_05_pose config_sb3_ddpg.yaml


# Remote connection
ssh -I 10.178.107.200 jofa@ia-li-2wqd414.unirobotts.local


# Change ip address
ip link 
sudo ifconfig enp0s31f6 192.168.1.101 up

# Run test_rtde.py
python3 test_rtde.py