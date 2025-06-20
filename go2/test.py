import torch
import time

from omni.isaac.lab.app import AppLauncher
from go2_env import Go2RSLEnvCfg, camera_follow
from go2_sensors import SensorManager
import go2_ctrl as ctrl

from omni.isaac.lab.envs import ManagerBasedEnv
import omni.kit.app  # 关键：获取 app 和 input interface

# 启动 Isaac Sim 应用
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

def run_sim():
    cfg = Go2RSLEnvCfg()
    ctrl.init_base_vel_cmd(cfg.scene.num_envs)

    # 初始化环境
    env = ManagerBasedEnv(cfg)
    env.reset()

    # 添加 RTX Lidar
    sensor_mgr = SensorManager(num_envs=cfg.scene.num_envs)
    sensor_mgr.add_rtx_lidar()

    # 订阅键盘事件（替代 carb）
    app = omni.kit.app.get_app()
    input_mgr = app.get_input_interface()
    input_mgr.subscribe_to_keyboard_events(ctrl.sub_keyboard_event)

    print("[INFO] Simulation started. Use W/A/S/D/Z/C to move the robot.")

    # 仿真主循环
    while simulation_app.is_running():
        env.step()
        if cfg.scene.num_envs == 1:
            camera_follow(env)

if __name__ == "__main__":
    run_sim()
    simulation_app.close()
