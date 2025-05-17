# Mainly AI Generated

import pandas as pd
import matplotlib.pyplot as plt

# Load CSV file
df = pd.read_csv("td3_best_parameter_configurations.csv")

# Create figure
plt.figure(figsize=(7, 4))

# Plot Run 1 (solid)
plt.plot(df["Step"], df["Run 1 - rollout/ep_rew_mean"],
         linestyle='solid', color='tab:blue', linewidth=2, label="Run 1")

# Plot Run 2 (dashed)
plt.plot(df["Step"], df["Run 2 - rollout/ep_rew_mean"],
         linestyle='dashed', color='tab:green', linewidth=2, label="Run 2")

# Labels and legend
plt.xlabel("Training Steps")
plt.ylabel("Mean Episode Reward")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save as PDF
plt.savefig("td3_best_parameter_configurations.pdf", format="pdf", bbox_inches="tight")
plt.close()
