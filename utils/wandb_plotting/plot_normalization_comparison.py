# Mainly AI Generated

import pandas as pd
import matplotlib.pyplot as plt

# Load CSV file
df = pd.read_csv("ppo_reach_task_best_performing_configuration.csv")

# Create figure
plt.figure(figsize=(7, 4))

# Plot With Normalization (solid)
plt.plot(df["Step"], df["With Normalization - rollout/ep_rew_mean"],
         linestyle='solid', color='tab:blue', linewidth=2, label="With Normalization")

# Plot Without Normalization (dashed)
plt.plot(df["Step"], df["Without Normalization - rollout/ep_rew_mean"],
         linestyle='dashed', color='tab:green', linewidth=2, label="Without Normalization")

# Labels and legend
plt.xlabel("Training Steps")
plt.ylabel("Mean Episode Reward")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save as PDF
plt.savefig("ppo_reach_task_best_performing_configuration.pdf", format="pdf", bbox_inches="tight")
plt.close()
