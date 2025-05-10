# Short project description
This project was part of my Master's Thesis for the Master's programme "Robot Systems - Advanced Robotics Technology" at the University of Southern Denmark (SDU). The task was to assess the feasibility of using RL-based robot control for a peg-in-hole task using the advanced physics simulator NVIDIA Isaac Lab.

This Repository includes the implementation to train PPO, DDPG or TD3 agents (from Stable-Baselines3) in Isaac Lab. The considered task includes a UR5e robot and requires the policy to move the robot such that it tracks a target pose with the robot's TCP. The implemented controllers are relative and absolute differential inverse kinematics (IK) control, joint position control, or impedance control (position tracking only).
This simple reach task was used in the thesis project as a benchmark to test the three RL algorithms and optimize their hyperparameters. Additionally, a comparison of relative and absolute control mode, the application of curriculum-based penalty strategies on action rate, action magnitude and end-effector acceleration, as well as the use of simple domain randomization strategies was performed.

The Weights&Biases tool was utilized to automate the hyperparameter search since it allows to extensively visualize the episode reward mean across training runs conducted with different hyperparameter configurations or task setups.




# Requirements
Follow these steps to create a virtual python environment and to install Isaac Sim and Isaac Lab (4.5.0):

https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html

Install requirements:
    
    pip install -r /path/to/requirements.txt 




# Hyperparameter optimization with Weights&Biases
## PPO
### UR5e without gripper - Rel IK

    source /path/to/virtual/environment/bin/activate
    cd /path/to/repository/sb3
    wandb sweep --project rel_ik_sb3_ppo_ur5e_reach_0_05_pose_without_gripper_parameter_optimization config_sb3_ppo.yaml
    wandb sweep --project impedance_ctrl_sb3_ppo_ur5e_reach_0_05_pose_without_gripper config_sb3_ppo.yaml
    

### UR5e without gripper - Rel IK - Domain Randomization

    source /path/to/virtual/environment/bin/activate
    cd /path/to/repository/sb3
    wandb sweep --project rel_ik_sb3_ppo_ur5e_reach_0_05_pose_without_gripper_domain_rand config_sb3_ppo.yaml


### UR5e without gripper - Abs IK

    source /path/to/virtual/environment/bin/activate
    cd /path/to/repository/sb3
    wandb sweep --project abs_ik_sb3_ppo_ur5e_reach_0_05_pose_without_gripper config_sb3_ppo.yaml


### UR5e without gripper - Impedance Control
    source /path/to/virtual/environment/bin/activate
    cd /path/to/repository/sb3
    wandb sweep --project impedance_ctrl_sb3_ppo_ur5e_reach_pose_without_gripper config_sb3_ppo.yaml


## DDPG - UR5e without gripper - Rel IK

    source /path/to/virtual/environment/bin/activate
    cd /path/to/repository/sb3
    wandb sweep --project rel_ik_sb3_ddpg_ur5e_reach_0_05_pose_grid_search config_sb3_ddpg.yaml


## TD3 - UR5e without gripper - Rel IK

    source /path/to/virtual/environment/bin/activate
    cd /path/to/repository/sb3
    wandb sweep --project rel_ik_sb3_td3_ur5e_reach_0_05_pose_grid_search config_sb3_td3.yaml


Notably, optimization run names and the specific environment that is used for training has to be changed inside train_sb3_wandb_ppo.py, train_sb3_wandb_ddpg.py, or train_sb3_wandb_td3.py, respectively. The task options are listed below.



## Train PPO agent without Weights&Biases
Option 1:

    source /path/to/virtual/environment/bin/activate
    cd /path/to/repository/sb3
    python3 train_sb3_ppo.py --num_envs 2048 --task UR5e-Reach-Pose-IK --headless

Option 2:

    source /path/to/virtual/environment/bin/activate
    cd /path/to/isaac/lab/installation/directory
    ./isaaclab.sh -p /path/to/repository/sb3/train_sb3_ppo.py --num_envs 2048 --task UR5e-Reach-Pose-IK --headless

Tensorboard can be used to visualize training results

    tensorboard --logdir='directory'

Note: For this option, the hyperparameters are defined in /gym_env/env/agents/




## Play PPO trained agent

    source /path/to/virtual/environment/bin/activate
    cd /path/to/repository/sb3
    python3 play_sb3_ppo.py --num_envs 4 --task UR5e-Reach-Pose-IK --checkpoint /path/to/trained/model

Note: This repository includes several pre-trained models in sb3/models/. These models were used to obtain the result described in the Master's Thesis.



## Examples to play PPO trained agent
### UR5e without gripper - Abs IK

    source isaaclab/bin/activate
    cd isaaclab/IsaacLab
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo.py --task UR5e-Reach-Pose-Abs-IK --num_envs 4 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/ppo_parameter_optimization/relative_vs_absolute/f414sb65/model.zip


### UR5e without gripper - Rel IK - with obs and rew normalization

    source isaaclab/bin/activate
    cd isaaclab/IsaacLab
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo.py --task UR5e-Reach-Pose-IK --num_envs 4 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/ppo_parameter_optimization/group_g/2zlcpe8a/model.zip


### UR5e without gripper - Rel IK - without obs and rew normalization

    source isaaclab/bin/activate
    cd isaaclab/IsaacLab
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo.py --task UR5e-Reach-Pose-IK --num_envs 4 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/ppo_parameter_optimization/group_b/q229e8ps/model.zip


### UR5e without gripper - Rel IK - with domain randomization

    source isaaclab/bin/activate
    cd isaaclab/IsaacLab
    ./isaaclab.sh -p /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/play_sb3_ppo.py --task UR5e-Domain-Rand-Reach-Pose-IK --num_envs 4 --checkpoint /home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/sb3/models/ppo_domain_rand/gains_0_9/gegtc7pj/model.zip




## Task options (defined in /gym_env/env/__init__.py)
UR5e without Gripper and Relative Differential Inverse Kinematics Action Space

    --task UR5e-Reach-Pose-IK


UR3e without Gripper and Relative Differential Inverse Kinematics Action Space

    --task UR3e-Reach-Pose-IK


UR5e without Gripper, Relative Differential Inverse Kinematics Action Space and Domain Randomization Strategies

    --task UR5e-Domain-Rand-Reach-Pose-IK


UR5e without Gripper and Absolute Differential Inverse Kinematics Action Space

    --task UR5e-Reach-Pose-Abs-IK


UR5e without Gripper, Absolute Differential Inverse Kinematics Action Space and Domain Randomization Strategies

    --task UR5e-Domain-Rand-Reach-Pose-Abs-IK


UR5e without Gripper and Joint Position Action Space

    --task UR5e-Reach


UR5e without Gripper and Impedance Control Action Space (position tracking only)

    --task UR5e-Impedance-Ctrl



# Real-world execution
Every trained model can be transferred to real-world. To run a specific model on the physical robot, connect to the UR robot, select the correct settings in the play_sb3_ppo_real_rtde.py script and run the following commands in a terminal. Note: An Isaac Lab installation is not required for real-world execution

    source /path/to/virtual/environment/bin/activate
    cd /path/to/repository
    python3 play_sb3_ppo_real_rtde.py


# Visualization scripts
Inside the utils-folder, several scripts are provided for analyzing and visualizing the data recorded during real-world deployment (play_sb3_ppo_real_rtde.py) or simulation experiments (play_sb3_ppo_save_observations.py). The CSV files provided in the data-folder were utilized for the Master's Thesis and include both real-world experiment data and simulation test trials.

