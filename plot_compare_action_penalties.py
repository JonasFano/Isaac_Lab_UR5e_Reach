import matplotlib.pyplot as plt

# === Data ===
mean_mae = [
    0.0289,  # Baseline
    # 0.0161,  # Act. Rate Pos -0.1
    0.0061,  # Act. Rate Pos -1.0
    0.0044,  # Act. Rate -1.0 (Normal)
    # 0.0085,  # Act. Rate -0.8
    0.0072,  # Act. Magnitude Pos -0.8
    # 0.0077,  # Act. Magnitude -0.05
    # 0.0141,  # Act. Magnitude -0.01
    0.0144,  # Act. Rate/Magn -0.1/-0.01
    0.0074   # Act. Rate/Magn -0.5/-0.02
]

smoothness = [
    0.0114,  # Baseline
    # 0.0054,  # Act. Rate Pos -0.1
    0.0041,  # Act. Rate Pos -1.0
    0.0038,  # Act. Rate -1.0 (Normal)
    # 0.0041,  # Act. Rate -0.8
    0.0043,  # Act. Magnitude Pos -0.8
    # 0.0078,  # Act. Magnitude -0.05
    # 0.0155,  # Act. Magnitude -0.01
    0.0049,  # Act. Rate/Magn -0.1/-0.01
    0.0042   # Act. Rate/Magn -0.5/-0.02
]

avg_success = [
    99.55,  # Baseline
    # 99.50,  # Act. Rate Pos -0.1
    99.35,  # Act. Rate Pos -1.0
    97.25,  # Act. Rate -1.0 (Normal)
    # 95.95,  # Act. Rate -0.8
    99.65,  # Act. Magnitude Pos -0.8
    # 90.50,  # Act. Magnitude -0.05
    # 96.95,  # Act. Magnitude -0.01
    98.80,  # Act. Rate/Magn -0.1/-0.01
    96.70   # Act. Rate/Magn -0.5/-0.02
]

config_names = [
    "Baseline",
    # "Act. Rate Pos -0.1",
    "Act. Rate Pos -1.0",
    "Act. Rate -1.0",
    # "Act. Rate -0.8",
    "Act. Magnitude Pos -0.8",
    # "Act. Magnitude -0.05",
    # "Act. Magnitude -0.01",
    "Rate/Magn -0.1/-0.01",
    "Rate/Magn -0.5/-0.02"
]

# === Plot 1: Mean MAE vs Success Rate ===
plt.figure(figsize=(8,6))
plt.scatter(mean_mae, avg_success, color='blue')

# Annotate points
for i, name in enumerate(config_names):
    plt.annotate(name, (mean_mae[i], avg_success[i]), textcoords="offset points", xytext=(5,5), ha='left', fontsize=10)

plt.xlabel("Mean MAE [m]")
plt.ylabel("Average Success Rate [%]")
# plt.title("Mean MAE vs Average Success Rate")
plt.grid(True)
plt.tight_layout()
plt.show()

# === Plot 2: Smoothness vs Success Rate ===
plt.figure(figsize=(8,6))
plt.scatter(smoothness, avg_success, color='green')

# Annotate points
for i, name in enumerate(config_names):
    plt.annotate(name, (smoothness[i], avg_success[i]), textcoords="offset points", xytext=(5,5), ha='left', fontsize=10)

plt.xlabel("Smoothness (Acceleration Energy)")
plt.ylabel("Average Success Rate [%]")
plt.title("Smoothness vs Average Success Rate")
plt.grid(True)
plt.tight_layout()
plt.show()
