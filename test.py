from isaacsim import SimulationApp
import os
import hydra
import rclpy
import torch
import time
import math

FILE_PATH = os.path.join(os.path.dirname(__file__), "cfg")

@hydra.main(config_path=FILE_PATH, config_name="sim", version_base=None)
def run_simulator(cfg):
    # ─── 1. 启动 Omniverse Kit ───────────────────────────
    simulation_app = SimulationApp({
        "headless":      False,
        "anti_aliasing": cfg.sim_app.anti_aliasing,
        "width":         cfg.sim_app.width,
        "height":        cfg.sim_app.height,
        "hide_ui":       cfg.sim_app.hide_ui,
    })

    # ─── 2. 导入包 ───────────────────────────────────────
    import omni, carb
    import go2.go2_ctrl as go2_ctrl
    from go2.go2_ctrl_cfg       import unitree_go2_rough_cfg
    from go2.go2_env            import Go2RSLEnvCfg, camera_follow
    import env.sim_env          as sim_env
    import go2.go2_sensors      as go2_sensors
    from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm

    # ─── 3. 配置环境 ─────────────────────────────────────
    go2_env_cfg = Go2RSLEnvCfg()
    go2_env_cfg.scene.num_envs       = cfg.num_envs
    go2_env_cfg.decimation           = math.ceil(1.0 / go2_env_cfg.sim.dt / cfg.freq)
    go2_env_cfg.sim.render_interval  = go2_env_cfg.decimation
    go2_env_cfg.observations.policy.concatenate_terms = True

    # ─── 4. 布置场景 ─────────────────────────────────────
    if   cfg.env_name == "obstacle-dense":      sim_env.create_obstacle_dense_env()
    elif cfg.env_name == "obstacle-medium":     sim_env.create_obstacle_medium_env()
    elif cfg.env_name == "obstacle-sparse":     sim_env.create_obstacle_sparse_env()
    elif cfg.env_name == "warehouse":           sim_env.create_warehouse_env()
    elif cfg.env_name == "warehouse-forklifts": sim_env.create_warehouse_forklifts_env()
    elif cfg.env_name == "warehouse-shelves":   sim_env.create_warehouse_shelves_env()
    elif cfg.env_name == "full-warehouse":      sim_env.create_full_warehouse_env()

    # ─── 5. 清空 checkpoint & init base_vel_cmd ────────────
    unitree_go2_rough_cfg["load_checkpoint"] = ""
    unitree_go2_rough_cfg["load_run"]        = ""
    go2_ctrl.init_base_vel_cmd(cfg.num_envs)

    # ─── 6. 先 new SensorManager，不 add annotator ─────────
    sm = go2_sensors.SensorManager(cfg.num_envs)

    # ─── 7. 定义 ObservationTerm（真正的数据会在后面 reset 时拿到）──
    def depth_term(env):
        arr = sm.get_depth_obs()
        if arr is None or arr.size == 0:
            H, W = 480, 640
            return torch.zeros((env.unwrapped.scene.num_envs, H*W),
                                device=env.device, dtype=torch.float32)
        t = torch.tensor(arr, dtype=torch.float32, device=env.device)
        return t.view(env.unwrapped.scene.num_envs, -1)

    def lidar_term(env):
        arr = sm.get_lidar_obs()
        print("arr", arr)
        if arr is None or arr.size == 0:
            return torch.zeros((env.unwrapped.scene.num_envs, 1),
                                device=env.device, dtype=torch.float32)
        t = torch.tensor(arr, dtype=torch.float32, device=env.device)
        return t.view(env.unwrapped.scene.num_envs, -1)

    go2_env_cfg.observations.policy.depth = ObsTerm(func=depth_term)
    go2_env_cfg.observations.policy.lidar = ObsTerm(func=lidar_term)

    # ─── 8. “真正”创建 env + policy ──────────────────────────
    env, policy = go2_ctrl.get_rsl_rough_policy(go2_env_cfg)

    # ─── 9. **此时** SimulationContext 已就绪，才挂传感器 ──────
    sm.add_camera(cfg.freq)
    # 这里给 LiDAR 一个现实的水平分辨率，比如 360 条射线
    sm.add_rtx_lidar()

    # ───10. 如果有 ROS2 Bridge，就 init ───────────────────
    rclpy.init()
    dm = None  # go2_ros2_bridge.RobotDataManager(env, ..., cfg)

    sim_step_dt = float(go2_env_cfg.sim.dt * go2_env_cfg.decimation)

    # ───11. **先跑几帧** update，再 reset ────────────────────
    for _ in range(20):
        simulation_app.update()
        time.sleep(sim_step_dt)

    obs, _ = env.reset()

    # ───12. 打印维度，确认生效 ───────────────────────────────
    print(">>> obs['policy'] shape:", obs["policy"].shape)
    print(">>> depth_term shape:", depth_term(env).shape)
    print(">>> lidar_term shape:", lidar_term(env).shape)

    # ───13. 主循环 ─────────────────────────────────────────
    while simulation_app.is_running():
        t0 = time.time()
        with torch.inference_mode():
            act = policy(obs)
            obs, _, _, _ = env.step(act)
            if dm:
                dm.pub_ros2_data()
                rclpy.spin_once(dm)
            if cfg.camera_follow:
                camera_follow(env)
        dt = time.time() - t0
        if dt < sim_step_dt:
            time.sleep(sim_step_dt - dt)
        rtf = min(1.0, sim_step_dt / dt)
        print(f"\rStep time: {dt*1000:.1f}ms, RTF: {rtf:.2f}", end="", flush=True)

    # ───14. 收尾 ───────────────────────────────────────────
    if dm:
        dm.destroy_node()
    rclpy.shutdown()
    simulation_app.close()

if __name__ == "__main__":
    run_simulator()