import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("network_architecture.csv")  # replace with your actual file name

# Define architectures and their styles
arch_configs = {
    "Network = [256,128,64]": {"style": "solid", "color": "tab:blue"},
    "Network = [256,128]": {"style": "dashed", "color": "tab:green"},
    "Network = [128,64]": {"style": "dotted", "color": "tab:orange"},
}

# Create figure
plt.figure(figsize=(7, 4))

# Plot each network architecture
for label, cfg in arch_configs.items():
    base_label = label.split(" = ")[1]  # extract just the [..] key
    col = f"{base_label} - rollout/ep_rew_mean"
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
plt.savefig("network_architecture.pdf", format="pdf", bbox_inches="tight")
plt.close()
