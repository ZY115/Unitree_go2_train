
from isaacsim import SimulationApp
import os
import math
import time
import torch
import rclpy
import hydra
import gymnasium as gym
from datetime import datetime
from omegaconf import OmegaConf
from argparse import ArgumentParser


@hydra.main(config_path="cfg", config_name="sim", version_base=None)
def main(cfg):
    simulation_app = SimulationApp({
        "headless": True,
        "renderer": "RayTracedLighting",
        "width": cfg.sim_app.width,
        "height": cfg.sim_app.height,
        "anti_aliasing": cfg.sim_app.anti_aliasing,
        "hide_ui": cfg.sim_app.hide_ui,
        "livelink": False,
    })

    from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    from go2.go2_ctrl_cfg import unitree_go2_rough_cfg
    from go2.go2_ctrl import init_base_vel_cmd
    from go2.go2_env import Go2RSLEnvCfg
    from go2.go2_sensors import SensorManager
    from ros2.go2_ros2_bridge import RobotDataManager
    import env.sim_env as sim_env

    env_cfg = Go2RSLEnvCfg()
    env_cfg.scene.num_envs        = cfg.num_envs
    env_cfg.decimation            = math.ceil(1.0 / env_cfg.sim.dt / cfg.freq)
    env_cfg.sim.render_interval   = env_cfg.decimation
    env_cfg.observations.policy.concatenate_terms = True

    if cfg.env_name == "obstacle-dense":
        sim_env.create_obstacle_dense_env()
    elif cfg.env_name == "obstacle-medium":
        sim_env.create_obstacle_medium_env()
    elif cfg.env_name == "obstacle-sparse":
        sim_env.create_obstacle_sparse_env()
    elif cfg.env_name == "warehouse":
        sim_env.create_warehouse_env()
    elif cfg.env_name == "warehouse-forklifts":
        sim_env.create_warehouse_forklifts_env()
    elif cfg.env_name == "warehouse-shelves":
        sim_env.create_warehouse_shelves_env()
    elif cfg.env_name == "full-warehouse":
        sim_env.create_full_warehouse_env()

    unitree_go2_rough_cfg["load_checkpoint"] = ""
    unitree_go2_rough_cfg["load_run"]        = ""
    init_base_vel_cmd(env_cfg.scene.num_envs)

    sm = SensorManager(env_cfg.scene.num_envs)

    from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
    
    ###
    #####################2025 6.25
    import torch.nn as nn
    import torch.nn.functional as F

    class DepthCNNEncoder(nn.Module):
        def __init__(self, input_height=480, input_width=640, output_dim=64):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 16, 5, stride=2, padding=2), nn.ReLU(),  # → [B, 16, H/2, W/2]
                nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(), # → [B, 32, H/4, W/4]
                nn.AdaptiveAvgPool2d((1, 1)),                         # → [B, 32, 1, 1]
                nn.Flatten(),                                         # → [B, 32]
                nn.Linear(32, output_dim)                             # → [B, output_dim]
            )

        def forward(self, depth_flat, H=480, W=640):
            x = depth_flat.view(-1, H, W).unsqueeze(1)  # → [B, 1, H, W]
            return self.encoder(x)
    depth_encoder = DepthCNNEncoder(output_dim=64).to("cuda")
    def depth_term(env_):
        if len(sm.cameras) == 0:
            sm.add_camera(cfg.freq)
        arr = sm.get_depth_obs()
        if arr is None or arr.size == 0:
            return torch.zeros((env_.unwrapped.scene.num_envs, 64), device=env_.device)

        t = torch.tensor(arr, dtype=torch.float32, device=env_.device)  # [B, H*W]
        feat = depth_encoder(t)  # → [B, 64]
        return feat
    #def depth_term(env_):
    #    if len(sm.cameras) == 0:
    #        sm.add_camera(cfg.freq)
    #    arr = sm.get_depth_obs()
    #    if arr is None or arr.size == 0:
    #        H, W = 480, 640
    #        return torch.zeros((env_.unwrapped.scene.num_envs, H*W),
    #                           device=env_.device, dtype=torch.float32)
    #    t = torch.tensor(arr, dtype=torch.float32, device=env_.device)
    #    return t.view(env_.unwrapped.scene.num_envs, -1)

    def lidar_term(env_):
        if len(sm.lidar_annotators) == 0:
            sm.add_rtx_lidar()
        arr = sm.get_lidar_obs()
        if arr is None or arr.size == 0:
            return torch.zeros((env_.unwrapped.scene.num_envs, 1),
                               device=env_.device, dtype=torch.float32)
        t = torch.tensor(arr, dtype=torch.float32, device=env_.device)
        return t.view(env_.unwrapped.scene.num_envs, -1)

    env_cfg.observations.policy.depth = ObsTerm(func=depth_term)
    # env_cfg.observations.policy.lidar = ObsTerm(func=lidar_term)

    env = gym.make("Isaac-Velocity-Rough-Unitree-Go2-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    env.unwrapped.sensor_manager = sm

    rclpy.init()
    ros2_dm = RobotDataManager(env, sm.lidar_annotators, sm.cameras, cfg)

    unitree_go2_rough_cfg["log_dir"] = os.path.join(
        "logs", "rsl_rl_go2_rough_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    os.makedirs(unitree_go2_rough_cfg["log_dir"], exist_ok=True)
    obss = env.reset()
    print("obss:",obss)
    # main_obs = obss[0]
    # print("main_obs.shape:", main_obs.shape)

    # # 结构化观测
    # obs_dict = obss[1]
    # policy_obs = obs_dict['observations']['policy']
    # print("policy_obs.shape:", policy_obs.shape)

    runner = OnPolicyRunner(
        env,
        unitree_go2_rough_cfg,
        log_dir=unitree_go2_rough_cfg["log_dir"],
        device=unitree_go2_rough_cfg["device"],
    )

    sim_step_dt = float(env_cfg.sim.dt * env_cfg.decimation)

    # run!
    for it in range(unitree_go2_rough_cfg["max_iterations"]):
        runner.learn(num_learning_iterations=1)

        ros2_dm.pub_ros2_data()
        rclpy.spin_once(ros2_dm)

        simulation_app.update()
        time.sleep(sim_step_dt)

    ros2_dm.destroy_node()
    rclpy.shutdown()
    simulation_app.close()


if __name__ == "__main__":
    main()
