from isaacsim import SimulationApp
simulation_app = SimulationApp({
    "headless": False,
    "renderer": "RayTracedLighting",
    "width": 1280,
    "height": 720,
    "display_options": 3286,
    "livelink": False,
})
import env.sim_env as sim_env
import os
import math
import torch
import time
import rclpy
from datetime import datetime
from argparse import ArgumentParser
import hydra
from omegaconf import OmegaConf
import gymnasium as gym

from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

from go2.go2_ctrl_cfg import unitree_go2_rough_cfg
from go2.go2_ctrl import init_base_vel_cmd
from go2.go2_env import Go2RSLEnvCfg
from go2.go2_sensors import SensorManager
from ros2.go2_ros2_bridge import RobotDataManager

@hydra.main(config_path="cfg", config_name="sim", version_base=None)
def main(cfg):
    # 创建 Isaac Lab 环境配置
    env_cfg = Go2RSLEnvCfg()
    env_cfg.scene.num_envs = cfg.num_envs
    env_cfg.seed = unitree_go2_rough_cfg["seed"]
    env_cfg.observations.policy.concatenate_terms = True
    env_cfg.observations.policy.concatenate_terms = True

    # env_cfg.observations.policy.terms["depth"] = {
    #     "scale": 1.0,
    #     "type": "sensor",
    #     "params": {
    #         "encoding": "flatten",
    #     },
    # }

    # env_cfg.observations.policy.terms["lidar"] = {
    #     "scale": 1.0,
    #     "type": "sensor",
    #     "params": {
    #         "encoding": "flatten",
    #     },
    # }

    # 设置仿真频率（确保 camera freq 可整除）
    env_cfg.decimation = math.ceil(1.0 / env_cfg.sim.dt / cfg.freq)
    env_cfg.sim.render_interval = env_cfg.decimation

    # 初始化 base_vel_cmd（required for observation term）
    init_base_vel_cmd(env_cfg.scene.num_envs)

    # 创建环境
    env = gym.make("Isaac-Velocity-Rough-Unitree-Go2-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    sim_env.create_obstacle_dense_env()

    # 初始化传感器
    sensor_manager = SensorManager(env_cfg.scene.num_envs)
    lidar_annotators = sensor_manager.add_rtx_lidar()
    cameras = []
    if hasattr(cfg, "sensor") and getattr(cfg.sensor, "enable_camera", False):
        try:
            cameras = sensor_manager.add_camera(freq=cfg.freq)
        except Exception as e:
            print(f"[Camera Init Error] {e}")

    # 初始化 ROS2 连接和数据发布
    rclpy.init()
    data_manager = RobotDataManager(env, lidar_annotators, cameras, cfg)

    unitree_go2_rough_cfg["log_dir"] = os.path.join("logs", "rsl_rl_go2_rough_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(unitree_go2_rough_cfg["log_dir"], exist_ok=True)


    runner = OnPolicyRunner(env, unitree_go2_rough_cfg, log_dir=unitree_go2_rough_cfg["log_dir"],
                            device=unitree_go2_rough_cfg["device"])

    # 主训练循环，每次迭代都发布一次传感器数据
    for i in range(unitree_go2_rough_cfg["max_iterations"]):
        runner.learn(num_learning_iterations=1)

        if lidar_annotators or cameras:
            data_manager.pub_ros2_data()
            rclpy.spin_once(data_manager)

        simulation_app.update()

    # 清理资源
    rclpy.shutdown()
    data_manager.destroy_node()
    simulation_app.close()

if __name__ == "__main__":
    main()
