# go2/sb3_cfg_entry_point.py
#这是Stable-Baselines3框架，如果要用这个框架，当前的train代码是不能用的，要重新写，所以我没改这段
from go2.go2_env import Go2RSLEnvCfg

def sb3_cfg_entry_point():
    # 1. 创建环境配置实例
    env_cfg = Go2RSLEnvCfg()

    # 2. 设置 PPO agent 的训练参数
    agent_cfg = {
        "policy": "MlpPolicy",
        "learning_rate": 3e-4,
        "n_steps": 24,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "normalize_input": True,
        "normalize_value": True,
        "clip_obs": 10.0,
        "n_timesteps": 500_000,   # 会在 main() 中被 CLI 覆盖
        "seed": 42,
    }

    return env_cfg, agent_cfg
