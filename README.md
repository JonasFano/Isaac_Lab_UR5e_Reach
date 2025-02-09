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
./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo.py --task UR5e-Reach-Pose-IK --num_envs 4 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/tnecdfk7_UR5e_Hand_E_Reach_Pose_IK/model.zip

./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo.py --task UR5e-Reach-Pose-IK --num_envs 4 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/3crumnxy_UR5e_Hand_E_Reach_Pose_IK/model.zip

./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo.py --task UR5e-Reach-Pose-IK --num_envs 4 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/mirz9884_UR5e_Hand_E_Reach_Pose_IK/model.zip

## Larger pose generation ranges and add Unoise to the observations
./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo.py --task UR5e-Reach-Pose-IK --num_envs 4 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/053yyx3b/model.zip



# v2: Unoise: 0.001 - Robot Reset" "position_range": (0.5, 1.5) - Pose Generation: pos_x=(-0.2, 0.2), pos_y=(0.25, 0.5), pos_z=(0.1, 0.4),



# v3: Unoise: 0.0001 - Robot Reset" "position_range": (0.8, 1.2) - Pose Generation: pos_x=(-0.15, 0.15), pos_y=(0.25, 0.5), pos_z=(0.1, 0.4),



# v4: Unoise: 0.0001 - Robot Reset" "position_range": (0.7, 1.3) - Pose Generation: pos_x=(-0.15, 0.15), pos_y=(0.25, 0.5), pos_z=(0.1, 0.4),



# Record data

./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo_test.py --task UR5e-Reach-Pose-IK --num_envs 1 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/053yyx3b/model.zip



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

wandb sweep --project rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e_final_v4 config_sb3_ppo.yaml

wandb sweep --project rel_ik_sb3_ppo_ur5e_reach_0_05_pose_hand_e_domain_rand config_sb3_ppo.yaml




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