# Isaac_Lab_UR5e_Reach
Utilize Reinforcement Learning in Isaac Lab using the UR5e to reach desired target poses with differential IK control. 



###########################
# Rel IK Relative Control #
###########################

# Stable-baselines3 - UR5e

source isaaclab/bin/activate
cd isaaclab/IsaacLab
./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/train_sb3.py --num_envs 1 --task UR5e-Reach-IK --headless --no_logging
./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/train_sb3.py --num_envs 8192 --task UR5e-Reach-Pose-IK --headless


./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/play_sb3.py --task UR5e-Reach-Pose-IK --num_envs 4 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/logs/sb3/ppo/UR5e-Reach-Pose-IK/2024-12-13_16-04-26/model.zip

# SAC
./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/play_sb3_sac.py --task UR5e-Reach-Pose-IK --num_envs 4 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/models/293fw7iy/model.zip

# Tensorboard
tensorboard --logdir='directory'




#######################
# IK Absolute Control #
#######################

# Stable-baselines3 - UR5e

source isaaclab/bin/activate
cd isaaclab/IsaacLab
./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/train_sb3.py --num_envs 8192 --task UR5e-Reach-Pose-Abs-IK --headless





##########################
# Joint Position Control #
##########################

# Stable-baselines3 - UR5e

source isaaclab/bin/activate
cd isaaclab/IsaacLab
./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/train_sb3.py --num_envs 8192 --task UR5e-Reach --headless

./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/play_sb3.py --task UR5e-Reach --num_envs 4 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/logs/sb3/ppo/UR5e-Reach/2024-11-26_07-31-34/model_204800000_steps.zip


# UR5e Wandb PPO
source isaaclab/bin/activate
cd /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach
wandb sweep --project rel_ik_sb3_ppo_ur5e_reach_0_05_pose config_0_05.yaml
wandb agent jofan23-university-of-southern-denmark/rel_ik_sb3_ppo_ur5e_reach_0_1/za203z2j
wandb agent jofan23-university-of-southern-denmark/rel_ik_sb3_ppo_ur5e_reach_0_1_v2/cs6czrhy

wandb agent jofan23-university-of-southern-denmark/rel_ik_sb3_ppo_ur5e_reach_0_1_corrected/s5h00sc3

wandb agent jofan23-university-of-southern-denmark/rel_ik_sb3_ppo_ur5e_reach_0_05_position/xvaf23h3



# UR5e Wandb SAC
source isaaclab/bin/activate
cd /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach
wandb sweep --project rel_ik_sb3_sac_ur5e_reach_0_05_pose config_sb3_sac.yaml
wandb sweep --project rel_ik_sb3_sac_ur5e_reach_0_05_pose_2 config_sb3_sac.yaml
wandb sweep --project rel_ik_sb3_sac_ur5e_reach_0_05_pose_3 config_sb3_sac.yaml




# UR5e Wandb TD3
source isaaclab/bin/activate
cd /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach
wandb sweep --project rel_ik_sb3_td3_ur5e_reach_0_05_pose config_sb3_td3.yaml


# UR5e Wandb DDPG
source isaaclab/bin/activate
cd /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach
wandb sweep --project rel_ik_sb3_ddpg_ur5e_reach_0_05_pose config_sb3_ddpg.yaml