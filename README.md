# Isaac_Lab_UR5e_Reach
Utilize Reinforcement Learning in Isaac Lab using the UR5e to reach desired target poses with differential IK control. 



#######################
# IK Relative Control #
#######################

# Stable-baselines3 - UR5e

source env_isaacsim/bin/activate
cd env_isaacsim/IsaacLab
./isaaclab.sh -p /home/jofa/Downloads/Isaac_Lab_UR5e_Reach/train_sb3.py --num_envs 4096 --task UR5e-Reach-IK --headless

./isaaclab.sh -p /home/jofa/Downloads/Isaac_Lab_UR5e_Reach/play_sb3.py --task UR5e-Reach-IK --num_envs 4 --checkpoint /home/jofa/Downloads/Isaac_Lab_UR5e_Reach/logs/sb3/ppo/UR5e-Reach-IK/2024-11-22_12-15-31/model_65536000_steps.zip



# Tensorboard
tensorboard --logdir='directory'