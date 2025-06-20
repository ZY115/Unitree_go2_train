# from isaacsim import SimulationApp

# import os
# import hydra
# import rclpy
# import torch
# import time
# import math



# _orig_randint = torch.randint
# def safe_randint(*args, **kwargs):
#     # device="cuda:0" → torch.device("cuda:0")
#     if "device" in kwargs and isinstance(kwargs["device"], str):
#         kwargs["device"] = torch.device(kwargs["device"])
#     return _orig_randint(*args, **kwargs)
# torch.randint = safe_randint

# FILE_PATH = os.path.join(os.path.dirname(__file__), "cfg")
# @hydra.main(config_path=FILE_PATH, config_name="sim", version_base=None)
# def run_simulator(cfg):
#     # launch omniverse app
#     simulation_app = SimulationApp({"headless": False, "anti_aliasing": cfg.sim_app.anti_aliasing,
#                                     "width": cfg.sim_app.width, "height": cfg.sim_app.height, 
#                                     "hide_ui": cfg.sim_app.hide_ui})

#     import omni
#     import carb
#     import go2.go2_ctrl as go2_ctrl
#     import ros2.go2_ros2_bridge as go2_ros2_bridge
#     from go2.go2_env import Go2RSLEnvCfg, camera_follow
#     import env.sim_env as sim_env
#     import go2.go2_sensors as go2_sensors


#     # Go2 Environment setup
#     go2_env_cfg = Go2RSLEnvCfg()
#     go2_env_cfg.scene.num_envs = cfg.num_envs
#     go2_env_cfg.decimation = math.ceil(1./go2_env_cfg.sim.dt/cfg.freq)
#     go2_env_cfg.sim.render_interval = go2_env_cfg.decimation
#     go2_ctrl.init_base_vel_cmd(cfg.num_envs)
#     # env, policy = go2_ctrl.get_rsl_flat_policy(go2_env_cfg)
#     env, policy = go2_ctrl.get_rsl_rough_policy(go2_env_cfg)

#     # Simulation environment
#     if (cfg.env_name == "obstacle-dense"):
#         sim_env.create_obstacle_dense_env() # obstacles dense
#     elif (cfg.env_name == "obstacle-medium"):
#         sim_env.create_obstacle_medium_env() # obstacles medium
#     elif (cfg.env_name == "obstacle-sparse"):
#         sim_env.create_obstacle_sparse_env() # obstacles sparse
#     elif (cfg.env_name == "warehouse"):
#         sim_env.create_warehouse_env() # warehouse
#     elif (cfg.env_name == "warehouse-forklifts"):
#         sim_env.create_warehouse_forklifts_env() # warehouse forklifts
#     elif (cfg.env_name == "warehouse-shelves"):
#         sim_env.create_warehouse_shelves_env() # warehouse shelves
#     elif (cfg.env_name == "full-warehouse"):
#         sim_env.create_full_warehouse_env() # full warehouse

#     # Sensor setup
#     sm = go2_sensors.SensorManager(cfg.num_envs)
#     lidar_annotators = sm.add_rtx_lidar()
#     #####################################
#     #for annotator in lidar_annotators:
#      #   print("leidaxinxi")
#       #  data = annotator.get_data()
#        # print(data.keys())           # 查看有哪些信息
#         #print(data["data"].shape, data["data"].dtype)
#     ############################################
#     cameras = sm.add_camera(cfg.freq)

#     # Keyboard control
#     system_input = carb.input.acquire_input_interface()
#     system_input.subscribe_to_keyboard_events(
#         omni.appwindow.get_default_app_window().get_keyboard(), go2_ctrl.sub_keyboard_event)
    
#     # ROS2 Bridge
#     rclpy.init()
#     dm = go2_ros2_bridge.RobotDataManager(env, lidar_annotators, cameras, cfg)

#     # Run simulation
#     sim_step_dt = float(go2_env_cfg.sim.dt * go2_env_cfg.decimation)
#     obs, _ = env.reset()
#     #################
#     frame_id = 0
#     ##################
#     while simulation_app.is_running():
#         start_time = time.time()
#         with torch.inference_mode():            
#             # control joints
#             actions = policy(obs)


#             # step the environment
#             obs, _, _, _ = env.step(actions)

#             # # ROS2 data
#             dm.pub_ros2_data()
#             rclpy.spin_once(dm)

#             # Camera follow
#             if (cfg.camera_follow):
#                 camera_follow(env)

#             # limit loop time
#             elapsed_time = time.time() - start_time
#             if elapsed_time < sim_step_dt:
#                 sleep_duration = sim_step_dt - elapsed_time
#                 time.sleep(sleep_duration)
#         actual_loop_time = time.time() - start_time
#         rtf = min(1.0, sim_step_dt/elapsed_time)
#         print(f"\rStep time: {actual_loop_time*1000:.2f}ms, Real Time Factor: {rtf:.2f}", end='', flush=True)
#         ##############################################
#         frame_id += 1
#         if frame_id % 20 == 0:
#             for i, annotator in enumerate(lidar_annotators):
#                 data = annotator.get_data()
                
#                 if "data" in data and data["data"].shape[0] > 0:
#                     xyz = data["data"]
#                     beamId = data["beamId"]  
#                     azimuth = data["azimuth"]   
#                     elevation = data["elevation"]

#                     print(f"\n[Env {i}] Lidar point shape: {data['data'].shape}, dtype: {data['data'].dtype}")
#                     print("First 5 Points:\n", xyz[:5], "beamId:", beamId[:5], "azimuth:", azimuth[:5], "elevation:", elevation[:5])
#                     print(type(beamId), beamId.shape if hasattr(beamId, 'shape') else len(beamId), beamId[:5])

#                 else:
#                     print(f"\n[Env {i}] no data")
#         ######################################################
#     dm.destroy_node()
#     rclpy.shutdown()
#     simulation_app.close()

# if __name__ == "__main__":
#     run_simulator()
    



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
    # launch omniverse app
    simulation_app = SimulationApp({"headless": False, "anti_aliasing": cfg.sim_app.anti_aliasing,
                                    "width": cfg.sim_app.width, "height": cfg.sim_app.height, 
                                    "hide_ui": cfg.sim_app.hide_ui})

    import omni
    import carb
    import go2.go2_ctrl as go2_ctrl
    import ros2.go2_ros2_bridge as go2_ros2_bridge
    from go2.go2_env import Go2RSLEnvCfg, camera_follow
    import env.sim_env as sim_env
    import go2.go2_sensors as go2_sensors

    # Go2 Environment setup
    go2_env_cfg = Go2RSLEnvCfg()
    go2_env_cfg.scene.num_envs = cfg.num_envs
    go2_env_cfg.decimation = math.ceil(1./go2_env_cfg.sim.dt/cfg.freq)
    go2_env_cfg.sim.render_interval = go2_env_cfg.decimation
    go2_env_cfg.observations.policy.concatenate_terms = True
    sm = go2_sensors.SensorManager(cfg.num_envs)
    # lidar_annotators = sm.add_rtx_lidar()
    # cameras = sm.add_camera(cfg.freq)
    from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm

    def depth_term(env):
        # 如果还没有相机，第一次进来就创建
        if len(sm.cameras) == 0:
            sm.add_camera(cfg.freq)

        depth_np = sm.get_depth_obs()    
        if depth_np is None or len(depth_np) == 0:
            H, W = 480, 640     
            return torch.zeros((env.unwrapped.scene.num_envs, H*W),
                            device=env.device, dtype=torch.float32)

        depth = torch.tensor(depth_np, dtype=torch.float32, device=env.device)
        return depth.view(env.unwrapped.scene.num_envs, -1)

    go2_env_cfg.observations.policy.depth = ObsTerm(
        func=depth_term
    )


    from go2.go2_ctrl_cfg import unitree_go2_rough_cfg
    unitree_go2_rough_cfg["load_checkpoint"] = "" 
    unitree_go2_rough_cfg["load_run"] = ""

    go2_ctrl.init_base_vel_cmd(cfg.num_envs)
    env, policy = go2_ctrl.get_rsl_rough_policy(go2_env_cfg)

    # Simulation environment
    if (cfg.env_name == "obstacle-dense"):
        sim_env.create_obstacle_dense_env() # obstacles dense
    elif (cfg.env_name == "obstacle-medium"):
        sim_env.create_obstacle_medium_env() # obstacles medium
    elif (cfg.env_name == "obstacle-sparse"):
        sim_env.create_obstacle_sparse_env() # obstacles sparse
    elif (cfg.env_name == "warehouse"):
        sim_env.create_warehouse_env() # warehouse
    elif (cfg.env_name == "warehouse-forklifts"):
        sim_env.create_warehouse_forklifts_env() # warehouse forklifts
    elif (cfg.env_name == "warehouse-shelves"):
        sim_env.create_warehouse_shelves_env() # warehouse shelves
    elif (cfg.env_name == "full-warehouse"):
        sim_env.create_full_warehouse_env() # full warehouse

    # Sensor setup
    # sm = go2_sensors.SensorManager(cfg.num_envs)
    # lidar_annotators = sm.add_rtx_lidar()
    # cameras = sm.add_camera(cfg.freq)


    # ---- 6. 用我们修改过的 cfg 生成 env + 新网络 ------------------------------
    env, policy = go2_ctrl.get_rsl_rough_policy(go2_env_cfg)
        ###################
    from go2.go2_env import ObsTerm
    sm.add_camera(freq=cfg.freq)
    env.unwrapped.sensor_manager = sm
    # go2_env_cfg.observations.policy.lidar = ObsTerm(func=lambda env: sm.get_lidar_obs())
    # go2_env_cfg.observations.policy.depth = ObsTerm(func=lambda env: sm.get_depth_obs())
    ############################

    #这里远程调试先把键盘控制关了
    # Keyboard control
    # system_input = carb.input.acquire_input_interface()
    # system_input.subscribe_to_keyboard_events(
    #     omni.appwindow.get_default_app_window().get_keyboard(), go2_ctrl.sub_keyboard_event)
    
    # ROS2 Bridge
    rclpy.init()
    # dm = go2_ros2_bridge.RobotDataManager(env, lidar_annotators, cameras, cfg)
    dm = None
    # Run simulation
    sim_step_dt = float(go2_env_cfg.sim.dt * go2_env_cfg.decimation)
    # print("`````````````````", go2_env_cfg.observations.policy)


    obs, _ = env.reset()
    # print("obs:", obs)
    print(obs.size())
    print("obs['policy'] size :", obs.shape) 
    print("policy obs len :", obs.shape[-1])
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():            
            # control joints
            actions = policy(obs)

            # step the environment
            obs, _, _, _ = env.step(actions)

            # # ROS2 data
            dm.pub_ros2_data()
            rclpy.spin_once(dm)

            # Camera follow
            if (cfg.camera_follow):
                camera_follow(env)

            # limit loop time
            elapsed_time = time.time() - start_time
            if elapsed_time < sim_step_dt:
                sleep_duration = sim_step_dt - elapsed_time
                time.sleep(sleep_duration)
        actual_loop_time = time.time() - start_time
        rtf = min(1.0, sim_step_dt/elapsed_time)
        print(f"\rStep time: {actual_loop_time*1000:.2f}ms, Real Time Factor: {rtf:.2f}", end='', flush=True)
    
    dm.destroy_node()
    rclpy.shutdown()
    simulation_app.close()

if __name__ == "__main__":
    run_simulator()
    