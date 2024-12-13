import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Replace with the path to your log file
log_file_path = '/home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/logs/sb3/ppo/UR5e-Reach-Pose-IK/2024-12-03_18-24-40/progress.csv'
# log_file_path = '/home/joefa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/logs/sb3/ppo/UR5e-Reach/2024-11-26_07-31-34/progress.csv'
# log_file_path = "/home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/logs/sb3/ppo/UR5e-Reach-Pose-IK/2024-12-13_16-04-26/progress.csv"

# Read the log file
data = pd.read_csv(log_file_path)

# Extract the relevant columns
timesteps = data['time/total_timesteps']
mean_rewards = data['rollout/ep_rew_mean']

# Remove rows where ep_rew_mean is NaN
valid_indices = ~mean_rewards.isna()
timesteps = timesteps[valid_indices].reset_index(drop=True)
mean_rewards = mean_rewards[valid_indices].reset_index(drop=True)

# Check if we have valid data
if timesteps.empty or mean_rewards.empty:
    print("No valid data available for plotting.")
else:
    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(timesteps, mean_rewards, color='blue', linewidth=2)
    
    # Add titles and labels
    plt.title('Mean Episode Reward Over Timesteps')
    plt.xlabel('Total Timesteps')
    plt.ylabel('Mean Episode Reward')
    plt.grid(True)
    
    # Show the plot
    plt.show()
