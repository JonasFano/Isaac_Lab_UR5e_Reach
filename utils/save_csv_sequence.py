import pandas as pd

filepath = "/home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/data/real_robot/domain_rand_model_random_poses_scale_0_01_seed_24"

# Load the original CSV file
df = pd.read_csv(filepath + ".csv")

# Select the first 25,500 rows
df_subset = df.head(60500)

# Save to a new CSV file
df_subset.to_csv(filepath + "_trim.csv" , index=False)
