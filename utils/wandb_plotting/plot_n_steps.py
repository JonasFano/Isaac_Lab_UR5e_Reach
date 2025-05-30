import pandas as pd
import matplotlib.pyplot as plt

# Load CSV (replace with your actual filename)
df = pd.read_csv("n_steps.csv")  # <- Adjust filename if needed

# Define n_step configs and their styles
n_step_configs = {
    "n_steps = 128": {"style": "solid", "color": "tab:blue"},
    "n_steps = 64": {"style": "dashed", "color": "tab:green"},
    "n_steps = 32": {"style": "dotted", "color": "tab:orange"},
}

# Create figure
plt.figure(figsize=(7, 4))

# Plot each n_step configuration
for label, cfg in n_step_configs.items():
    key = label.split(" = ")[1]  # extract 128, 64, 32
    col = f"{key} - rollout/ep_rew_mean"
    if col in df.columns:
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
plt.savefig("n_steps.pdf", format="pdf", bbox_inches="tight")
plt.close()
