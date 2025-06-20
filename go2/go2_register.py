import gymnasium as gym
from omni.isaac.lab.envs import ManagerBasedRLEnv
from go2.go2_env import Go2RSLEnvCfg

# 注册环境，只执行一次就可以
if "Go2-Flat-SB3-v0" not in gym.registry:
    gym.register(
        id="Go2-Flat-SB3-v0",
        entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
        kwargs={
            "cfg": Go2RSLEnvCfg(),
            "env_cfg_entry_point": "go2.sb3_cfg_entry_point:sb3_cfg_entry_point",    # ✅ 用于 hydra 加载配置
            "agent_cfg_entry_point": "go2.sb3_cfg_entry_point:sb3_cfg_entry_point",  # ✅ 用于 hydra 加载配置
        },
    )