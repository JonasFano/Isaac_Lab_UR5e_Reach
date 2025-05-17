# Mainly AI Generated

import matplotlib.pyplot as plt

# === Data ===
config_names = [
    "No Penalties",
    "Act. Rate Pos (-1.0)",
    "Act. Rate (-1.0)",
    "Act. Magnitude Pos (-0.8)",
    "Act. Rate/Magnitude (-0.1/-0.01)",
    "Act. Rate/Magnitude (-0.5/-0.02)"
]

mean_mae = [
    0.02888,  # No Penalties
    0.00613,  # Act. Rate Pos -1.0
    0.00436,  # Act. Rate -1.0
    0.00718,  # Act. Magnitude Pos -0.8
    0.01441,  # Combined -0.1 / -0.01
    0.00744   # Combined -0.5 / -0.02
]

# Smoothness = sum of squared accelerations (lower = smoother)
smoothness = [
    70903.19014,  # No Penalties
    25617.38342,  # Act. Rate Pos -1.0
    23975.21756,  # Act. Rate -1.0
    26655.03226,  # Act. Magnitude Pos -0.8
    31019.11917,  # Combined -0.1 / -0.01
    26535.50488   # Combined -0.5 / -0.02
]

avg_success = [
    98.7,  # No Penalties
    98.4,  # Act. Rate Pos -1.0
    95.4,  # Act. Rate -1.0
    99.5,  # Act. Magnitude Pos -0.8
    96.7,  # Combined -0.1 / -0.01
    92.5   # Combined -0.5 / -0.02
]

# === Plot 1: Mean MAE vs Success Rate ===
plt.figure(figsize=(8, 6))
plt.scatter(mean_mae, avg_success, color='blue')

for i, name in enumerate(config_names):
    plt.annotate(name, (mean_mae[i], avg_success[i]), textcoords="offset points", xytext=(5, 5), ha='left', fontsize=10)

plt.xlabel("Mean MAE [m]")
plt.ylabel("Average Success Rate [%]")
# plt.title("Mean MAE vs Average Success Rate")
plt.grid(True)
plt.tight_layout()
plt.savefig("action_penalty_comparison.pdf", format="pdf", bbox_inches="tight")
plt.show()

# === Plot 2: Smoothness vs Success Rate ===
plt.figure(figsize=(8, 6))
plt.scatter(smoothness, avg_success, color='green')

for i, name in enumerate(config_names):
    plt.annotate(name, (smoothness[i], avg_success[i]), textcoords="offset points", xytext=(5, 5), ha='left', fontsize=10)

plt.xlabel("Smoothness (Sum of Squared Accelerations)")
plt.ylabel("Average Success Rate [%]")
# plt.title("Smoothness vs Average Success Rate")
plt.grid(True)
plt.tight_layout()
plt.show()
