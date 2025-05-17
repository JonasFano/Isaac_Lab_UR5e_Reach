# Mainly AI Generated

import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("batch_size_learning_curves.csv")

# Define batch sizes and their styles
batch_configs = {
    "Batch Size = 2048": {"style": "solid", "color": "tab:blue"},
    "Batch Size = 8192": {"style": "dashed", "color": "tab:green"},
    "Batch Size = 16384": {"style": "dotted", "color": "tab:orange"},
    "Batch Size = 32768": {"style": "dashdot", "color": "tab:red"},
}

# Create figure
plt.figure(figsize=(7, 4))

# Plot each batch size
for label, cfg in batch_configs.items():
    col = f"{label} - rollout/ep_rew_mean"
    plt.plot(df["Step"], df[col],
             linestyle=cfg["style"],
             color=cfg["color"],
             linewidth=2,
             label=label)

# Labels and legend
plt.xlabel("Training Steps")
plt.ylabel("Mean Episode Reward")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save as PDF
plt.savefig("batch_size_learning_curves.pdf", format="pdf", bbox_inches="tight")
plt.close()
