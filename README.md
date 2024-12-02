# Isaac_Lab_UR5e_Reach
Utilize Reinforcement Learning in Isaac Lab using the UR5e to reach desired target poses with differential IK control. 



#######################
# IK Relative Control #
#######################

# Stable-baselines3 - UR5e

source isaaclab/bin/activate
cd isaaclab/IsaacLab
./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/train_sb3.py --num_envs 4096 --task UR5e-Reach-IK --headless

./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/play_sb3.py --task UR5e-Reach-IK --num_envs 4 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/logs/sb3/ppo/UR5e-Reach-IK/2024-11-25_13-05-41/model_65536000_steps.zip



# Tensorboard
tensorboard --logdir='directory'




##########################
# Joint Position Control #
##########################

# Stable-baselines3 - UR5e

source isaaclab/bin/activate
cd isaaclab/IsaacLab
./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/train_sb3.py --num_envs 8192 --task UR5e-Reach --headless

./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/play_sb3.py --task UR5e-Reach --num_envs 4 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/logs/sb3/ppo/UR5e-Reach/2024-11-26_07-31-34/model_204800000_steps.zip


# UR5e Scale of 0.1
source isaaclab/bin/activate
cd /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach
wandb sweep --project rel_ik_sb3_ppo_ur5e_reach_0_1_corrected_v2 config_0_1.yaml
wandb agent jofan23-university-of-southern-denmark/rel_ik_sb3_ppo_ur5e_reach_0_1/za203z2j
wandb agent jofan23-university-of-southern-denmark/rel_ik_sb3_ppo_ur5e_reach_0_1_v2/cs6czrhy