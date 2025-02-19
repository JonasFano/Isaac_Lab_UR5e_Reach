import gymnasium as gym
from . import agents, ik_rel_env_cfg, joint_pos_env_cfg, ik_rel_env_cfg_pose, ik_abs_env_cfg_pose, ik_rel_env_cfg_pose_domain_rand, ik_rel_env_cfg_pose_ur3e, ik_abs_env_cfg_pose_domain_rand

# Register Gym environments.


# UR5e Reach Position Rel IK
gym.register(
    id="UR5e-Reach-IK",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ik_rel_env_cfg.RelIK_UR5e_ReachEnvCfg,
        "sb3_ppo_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_sac_cfg_entry_point": f"{agents.__name__}:sb3_sac_cfg.yaml",
    },
)

# UR5e Reach Pose Rel IK
gym.register(
    id="UR5e-Reach-Pose-IK",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ik_rel_env_cfg_pose.RelIK_UR5e_ReachEnvCfg,
        "sb3_ppo_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_sac_cfg_entry_point": f"{agents.__name__}:sb3_sac_cfg.yaml",
    },
)

# UR3e Reach Pose Rel IK
gym.register(
    id="UR3e-Reach-Pose-IK",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ik_rel_env_cfg_pose_ur3e.RelIK_UR3e_ReachEnvCfg,
        "sb3_ppo_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_sac_cfg_entry_point": f"{agents.__name__}:sb3_sac_cfg.yaml",
    },
)

# Rel IK UR5e with domain randomization
gym.register(
    id="UR5e-Domain-Rand-Reach-Pose-IK",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ik_rel_env_cfg_pose_domain_rand.RelIK_UR5e_Domain_Rand_ReachEnvCfg,
        "sb3_ppo_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_sac_cfg_entry_point": f"{agents.__name__}:sb3_sac_cfg.yaml",
    },
)

# UR5e Reach Pose Abs IK
gym.register(
    id="UR5e-Reach-Pose-Abs-IK",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ik_abs_env_cfg_pose.AbsIK_UR5e_ReachEnvCfg,
        "sb3_ppo_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_sac_cfg_entry_point": f"{agents.__name__}:sb3_sac_cfg.yaml",
    },
)

# UR5e Reach Pose Abs IK with domain randomization
gym.register(
    id="UR5e-Domain-Rand-Reach-Pose-Abs-IK",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ik_abs_env_cfg_pose_domain_rand.AbsIK_UR5e_Domain_Rand_ReachEnvCfg,
        "sb3_ppo_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_sac_cfg_entry_point": f"{agents.__name__}:sb3_sac_cfg.yaml",
    },
)

# UR5e Reach Position Joint Position Control
gym.register(
    id="UR5e-Reach",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": joint_pos_env_cfg.JointPos_UR5e_ReachEnvCfg,
        "sb3_ppo_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_sac_cfg_entry_point": f"{agents.__name__}:sb3_sac_cfg.yaml",
    },
)
