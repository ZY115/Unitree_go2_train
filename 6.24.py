from isaacsim import SimulationApp
import os
import hydra
import time
import math

FILE_PATH = os.path.join(os.path.dirname(__file__), "cfg")

@hydra.main(config_path=FILE_PATH, config_name="sim", version_base=None)
def run_simulator(cfg):
    # 1. 启动 Omniverse Kit
    simulation_app = SimulationApp({
        "headless":      False,
        "anti_aliasing": cfg.sim_app.anti_aliasing,
        "width":         cfg.sim_app.width,
        "height":        cfg.sim_app.height,
        "hide_ui":       cfg.sim_app.hide_ui,
    })

    # 2. 导入包
    import go2.go2_env as go2_env
    import env.sim_env as sim_env
    import go2.go2_sensors as go2_sensors

    # 3. 配置环境
    env_cfg = go2_env.Go2RSLEnvCfg()
    env_cfg.scene.num_envs      = cfg.num_envs
    env_cfg.decimation          = math.ceil(1.0 / env_cfg.sim.dt / cfg.freq)
    env_cfg.sim.render_interval = env_cfg.decimation

    # 4. 创建环境
    if   cfg.env_name == "obstacle-dense":      sim_env.create_obstacle_dense_env()
    elif cfg.env_name == "obstacle-medium":     sim_env.create_obstacle_medium_env()
    elif cfg.env_name == "obstacle-sparse":     sim_env.create_obstacle_sparse_env()
    elif cfg.env_name == "warehouse":           sim_env.create_warehouse_env()
    elif cfg.env_name == "warehouse-forklifts": sim_env.create_warehouse_forklifts_env()
    elif cfg.env_name == "warehouse-shelves":   sim_env.create_warehouse_shelves_env()
    elif cfg.env_name == "full-warehouse":      sim_env.create_full_warehouse_env()

    # 5. 初始化 SensorManager 并添加 LiDAR
    sm = go2_sensors.SensorManager(cfg.num_envs)
    sm.add_rtx_lidar()  # 必须要先 add
    print("已调用 sm.add_rtx_lidar()")

    sim_step_dt = float(env_cfg.sim.dt * env_cfg.decimation)

    # 6. 跑几帧（让仿真走起来），再输出数据
    for _ in range(20):
        simulation_app.update()
        time.sleep(sim_step_dt)

    # 7. 打印一次 LiDAR 数据的 shape 和部分内容
    arr = sm.get_lidar_obs()
    print("[RESET] LiDAR shape:", arr.shape)
    print("[RESET] LiDAR sample:", arr[:10])

    # 8. 主循环，每隔几帧输出一次
    i = 0
    while simulation_app.is_running() and i < 100:   # 只循环100帧防止跑飞
        for i in range(100):
            simulation_app.update()
            
            arr = sm.get_lidar_obs()

            print(f"[{i}] LiDAR shape:", arr.shape)
            print(f"[{i}] LiDAR sample:", arr[:10])
            time.sleep(sim_step_dt)
            i += 1

    simulation_app.close()

if __name__ == "__main__":
    run_simulator()
