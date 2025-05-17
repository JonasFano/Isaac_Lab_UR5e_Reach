# Mainly AI Generated

import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV
df = pd.read_csv("action_penalties.csv")

# Mapping from readable labels to exact column name prefixes
sweeps = {
    "Act. Rate/Magnitude (-0.5/-0.02)": "Act. Rate/Magnitude (-0.5/-0.02)",
    "Act. Rate (-1.0)": "Act. Rate (-1.0)",
    "Act. Rate Pos (-1.0)": "Act. Rate Pos (-1.0)",
}

colors = {
    "Act. Rate/Magnitude (-0.5/-0.02)": "tab:blue",
    "Act. Rate (-1.0)": "tab:green",
    "Act. Rate Pos (-1.0)": "tab:orange",
}

line_styles = {
    "Act. Rate/Magnitude (-0.5/-0.02)": "-",     # solid
    "Act. Rate (-1.0)": "--",                   # dashed
    "Act. Rate Pos (-1.0)": ":",                # dotted
}

# Plot setup
plt.figure(figsize=(7, 4))

# Plot each sweep
for label, prefix in sweeps.items():
    col = f"{prefix} - rollout/ep_rew_mean"
    if col in df.columns:
        plt.plot(df["Step"], df[col],
                 label=label,
                 linestyle=line_styles[label],
                 color=colors[label],
                 linewidth=2)
    else:
        print(f"Warning: Column '{col}' not found.")

# Formatting
plt.xlabel("Training Steps")
plt.ylabel("Mean Episode Reward")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save to PDF
plt.savefig("action_penalties.pdf", format="pdf", bbox_inches="tight")
plt.close()
