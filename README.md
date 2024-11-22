# Isaac_Lab_UR5e_Reach
Utilize Reinforcement Learning in Isaac Lab using the UR5e to reach desired target poses with differential IK control. 



#######################
# IK Relative Control #
#######################

# Stable-baselines3 - UR5e

source env_isaacsim/bin/activate
cd env_isaacsim/IsaacLab
./isaaclab.sh -p /home/jofa/Downloads/Isaac_Lab_UR5e_Reach/train_sb3.py --num_envs 4096 --task UR5e-Reach-IK --headless
