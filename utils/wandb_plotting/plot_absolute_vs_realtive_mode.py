# Mainly AI Generated

import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("ppo_reach_task_absolute_mode.csv")

# Create figure
plt.figure(figsize=(7, 4))

# Plot Relative Mode (solid)
plt.plot(df["Step"], df["Relative Mode - rollout/ep_rew_mean"],
         linestyle='solid', color='tab:blue', linewidth=2, label="Relative Mode")

# Plot Absolute Mode (dashed)
plt.plot(df["Step"], df["Absolute Mode - rollout/ep_rew_mean"],
         linestyle='dashed', color='tab:orange', linewidth=2, label="Absolute Mode")

# Labels and legend
plt.xlabel("Training Steps")
plt.ylabel("Mean Episode Reward")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save as PDF
plt.savefig("ppo_reach_task_absolute_mode.pdf", format="pdf", bbox_inches="tight")
plt.close()
