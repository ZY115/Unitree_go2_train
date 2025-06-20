# # train_sb3.py

# import argparse
# import sys
# import os
# from datetime import datetime

# from omni.isaac.lab.app import AppLauncher

# # ========== STEP 1: CLI + AppLauncher ==========
# parser = argparse.ArgumentParser(description="Train Unitree Go2 with SB3 PPO.")
# parser.add_argument("--task", type=str, required=True, help="Hydra task config path, e.g., go2/cfg/sb3/flat")
# parser.add_argument("--num_envs", type=int, default=2)
# parser.add_argument("--max_iterations", type=int, default=1000)
# parser.add_argument("--video", action="store_true")
# parser.add_argument("--video_length", type=int, default=200)
# parser.add_argument("--video_interval", type=int, default=2000)
# AppLauncher.add_app_launcher_args(parser)
# args_cli, unknown_args = parser.parse_known_args()

# if args_cli.video:
#     args_cli.enable_cameras = True

# # 让 hydra 能识别 task 配置路径
# sys.argv = [sys.argv[0]] + unknown_args + [f"task={args_cli.task}"]
# app_launcher = AppLauncher(args_cli)
# simulation_app = app_launcher.app

# # ========== STEP 2: 训练逻辑 ==========
# import gymnasium as gym
# import numpy as np
# from stable_baselines3 import PPO
# from stable_baselines3.common.callbacks import CheckpointCallback
# from stable_baselines3.common.logger import configure
# from stable_baselines3.common.vec_env import VecNormalize
# from omni.isaac.lab.envs import DirectMARLEnv, multi_agent_to_single_agent
# from omni.isaac.lab.utils.io import dump_yaml, dump_pickle
# from omni.isaac.lab.utils.dict import print_dict
# from omni.isaac.lab_tasks.utils.hydra import hydra_task_config
# from omni.isaac.lab_tasks.utils.wrappers.sb3 import Sb3VecEnvWrapper, process_sb3_cfg
# from go2 import go2_ctrl
# from go2 import go2_register

# @hydra_task_config(args_cli.task, "go2.sb3_cfg_entry_point:sb3_cfg_entry_point")
# def main(env_cfg, agent_cfg):
#     # 覆盖部分参数
#     env_cfg.scene.num_envs = args_cli.num_envs
#     env_cfg.seed = agent_cfg["seed"]
#     agent_cfg["n_timesteps"] = args_cli.max_iterations * agent_cfg["n_steps"] * args_cli.num_envs

#     # 日志路径
#     log_dir = os.path.join("logs", "sb3", args_cli.task.replace("/", "_"), datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
#     os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)

#     # 保存配置
#     dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
#     dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
#     dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
#     dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

#     # SB3 agent config
#     agent_cfg = process_sb3_cfg(agent_cfg)
#     policy_arch = agent_cfg.pop("policy")
#     n_timesteps = agent_cfg.pop("n_timesteps")

#     # 环境初始化（确保已注册）
#     env = gym.make("Go2-Flat-SB3-v0", cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

#     # 速度命令控制
#     go2_ctrl.init_base_vel_cmd(env.num_envs)
#     env.unwrapped._event_subs.append(go2_ctrl.sub_keyboard_event)

#     # 视频录制
#     if args_cli.video:
#         video_kwargs = {
#             "video_folder": os.path.join(log_dir, "videos", "train"),
#             "step_trigger": lambda step: step % args_cli.video_interval == 0,
#             "video_length": args_cli.video_length,
#             "disable_logger": True,
#         }
#         print("[INFO] Recording videos during training.")
#         print_dict(video_kwargs, nesting=4)
#         env = gym.wrappers.RecordVideo(env, **video_kwargs)

#     # 单智能体包装
#     if isinstance(env.unwrapped, DirectMARLEnv):
#         env = multi_agent_to_single_agent(env)

#     # SB3 环境包装
#     env = Sb3VecEnvWrapper(env)

#     # 归一化
#     if "normalize_input" in agent_cfg:
#         env = VecNormalize(
#             env,
#             training=True,
#             norm_obs=agent_cfg.pop("normalize_input"),
#             norm_reward=agent_cfg.pop("normalize_value"),
#             clip_obs=agent_cfg.pop("clip_obs"),
#             gamma=agent_cfg["gamma"],
#             clip_reward=np.inf,
#         )

#     # 训练 Agent
#     agent = PPO(policy_arch, env, verbose=1, **agent_cfg)
#     agent.set_logger(configure(log_dir, ["stdout", "tensorboard"]))

#     checkpoint_callback = CheckpointCallback(
#         save_freq=1000,
#         save_path=log_dir,
#         name_prefix="model",
#         verbose=2,
#     )

#     # 训练主循环
#     agent.learn(total_timesteps=n_timesteps, callback=checkpoint_callback)
#     agent.save(os.path.join(log_dir, "model"))

#     env.close()

# if __name__ == "__main__":
#     main()
#     simulation_app.close()


