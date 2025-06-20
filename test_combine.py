from isaacsim import SimulationApp
simulation_app = SimulationApp({
    "headless": False,
    "renderer": "RayTracedLighting",
    "width": 1280,
    "height": 720,
    "display_options": 3286,
    "livelink": False,
})

import omni
import env.sim_env as sim_env
import numpy as np
from pxr import Gf
import omni.replicator.core as rep
from omni.isaac.sensor import Camera
import omni.isaac.core.utils.numpy.rotations as rot_utils

# ==== SensorManager实现（不变） ====
class SensorManager:
    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.lidar_annotators = []
        self.cameras = []

    def add_rtx_lidar(self):
        lidar_annotators = []
        for env_idx in range(self.num_envs):
            _, sensor = omni.kit.commands.execute(
                "IsaacSensorCreateRtxLidar",
                path="/lidar",
                parent=f"/World/envs/env_{env_idx}/Go2/base",
                config="Hesai_XT32_SD10",
                translation=(0.2, 0, 0.2),
                orientation=Gf.Quatd(1.0, 0.0, 0.0, 0.0),
            )
            annotator = rep.AnnotatorRegistry.get_annotator("RtxSensorCpuIsaacCreateRTXLidarScanBuffer")
            hydra_texture = rep.create.render_product(sensor.GetPath(), [1, 1], name="Isaac")
            annotator.attach(hydra_texture.path)
            lidar_annotators.append(annotator)
        self.lidar_annotators = lidar_annotators
        return lidar_annotators

    def add_camera(self, freq):
        from omni.isaac.core.simulation_context import SimulationContext
        sim_dt = SimulationContext.instance().get_physics_dt()
        render_interval = SimulationContext.instance().get_rendering_dt() / sim_dt
        render_freq = 1.0 / (sim_dt * render_interval)

        if render_freq % freq != 0:
            valid_freqs = [f for f in [30, 20, 15, 10, 5, 2, 1] if render_freq % f == 0]
            fallback = valid_freqs[0] if valid_freqs else 1.0
            print(f"[WARN] Camera freq {freq} not compatible with render freq {render_freq:.1f}Hz. Using {fallback}Hz instead.")
            freq = fallback

        cameras = []
        for env_idx in range(self.num_envs):
                    camera = Camera(
                        prim_path=f"/World/envs/env_{env_idx}/Go2/base/front_cam",
                        translation=np.array([0.4, 0.0, 0.2]),
                        frequency=freq,
                        resolution=(640, 480),
                        orientation=rot_utils.euler_angles_to_quats(np.array([0, 0, 0]), degrees=True),
                    )
                    camera.initialize()
                    camera.set_focal_length(1.5)
                    # camera.add_distance_to_camera_to_frame()
                    camera.add_distance_to_image_plane_to_frame()
                    cameras.append(camera)
        self.cameras = cameras
        return cameras

    def get_lidar_obs(self):
        lidar_datas = []
        for annotator in self.lidar_annotators:
            dic_data = annotator.get_data()  # 你实际采集方法
            print(dic_data["data"])
            data = dic_data["data"]
            lidar_datas.append(data)
        return np.stack(lidar_datas, axis=0)

    def get_depth_obs(self):
        depth_datas = []
        for annotator in self.cameras:
            rgba_img = annotator.get_rgba()
            print("rgba_img:", type(rgba_img), getattr(rgba_img, "shape", None))
            depth_img = annotator.get_depth()
            # depth_img = annotator.add_motion_vectors_to_frame()
            print("[DEBUG] depth_img type:", type(depth_img), "content:", depth_img)
            if depth_img is not None and isinstance(depth_img, np.ndarray):
                depth_datas.append(depth_img)
        if len(depth_datas) == 0:
            print("[ERROR] No depth images collected!")
            return None
        return np.stack(depth_datas, axis=0)

# ==== 你的自定义 Go2RSLEnv ====
from omni.isaac.lab.envs import ManagerBasedRLEnv

class Go2RSLEnv(ManagerBasedRLEnv):
    def __init__(self, cfg, sensor_manager=None):
        super().__init__(cfg)
        self.sensor_manager = sensor_manager

    # def compute_observations(self):
    #     obs_dict = self.obs_manager.compute()

    #     # 加入自定义深度相机观测
    #     if self.sensor_manager is not None and hasattr(self.sensor_manager, "get_depth_obs"):
    #         try:
    #             depth_obs = self.sensor_manager.get_depth_obs()
    #             if depth_obs is not None:
    #                 obs_dict["depth"] = np.array(depth_obs).reshape(self.num_envs, -1)
    #         except Exception as e:
    #             print(f"[WARN] Failed to get depth_obs: {e}")
    #     # 加入自定义雷达观测
    #     if self.sensor_manager is not None and hasattr(self.sensor_manager, "get_lidar_obs"):
    #         try:
    #             lidar_obs = self.sensor_manager.get_lidar_obs()
    #             if lidar_obs is not None:
    #                 obs_dict["lidar"] = np.array(lidar_obs).reshape(self.num_envs, -1)
    #         except Exception as e:
    #             print(f"[WARN] Failed to get lidar_obs: {e}")

    #     return obs_dict

    def compute_observations(self):
        #obs_dict = {}
        #这一行没必要，下面的初始obs数据生成了一个完整的初始obs_dict
        ##############
        obs_dict = super().compute_observations()
        ##############
        num_envs = self.cfg.scene.num_envs
        #下面两个就是雷达和深度相机信息了
        if self.sensor_manager is not None and hasattr(self.sensor_manager, "get_depth_obs"):
            try:
                depth_obs = self.sensor_manager.get_depth_obs()
                if depth_obs is not None:
                    obs_dict["depth"] = np.array(depth_obs).reshape(num_envs, -1)
            except Exception as e:
                print(f"[WARN] Failed to get depth_obs: {e}")

        if self.sensor_manager is not None and hasattr(self.sensor_manager, "get_lidar_obs"):
            try:
                lidar_obs = self.sensor_manager.get_lidar_obs()
                if lidar_obs is not None:
                    obs_dict["lidar"] = np.array(lidar_obs).reshape(num_envs, -1)
            except Exception as e:
                print(f"[WARN] Failed to get lidar_obs: {e}")
        print(obs_dict.keys())
        return obs_dict
    

# ==== 测试代码入口 ====
import hydra
from go2.go2_env import Go2RSLEnvCfg
from go2.go2_ctrl_cfg import unitree_go2_rough_cfg
from go2.go2_ctrl import init_base_vel_cmd
#from go2.go2_env import Go2RSLEnv
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper
import env.sim_env as sim_env

@hydra.main(config_path="cfg", config_name="sim", version_base=None)
def main(cfg):
    # 1. 创建环境配置
    env_cfg = Go2RSLEnvCfg()
    env_cfg.scene.num_envs = cfg.num_envs
    env_cfg.seed = unitree_go2_rough_cfg["seed"]

    # 2. 初始化期望速度结构（必须）
    init_base_vel_cmd(env_cfg.scene.num_envs)

    # 3. 创建基础环境
    raw_env = Go2RSLEnv(env_cfg)
    sim_env.create_obstacle_dense_env()

    # 4. 初始化传感器
    # from test_combine import SensorManager  # 如果 SensorManager 和 main 分在两个文件
    sensor_manager = SensorManager(env_cfg.scene.num_envs)
    sensor_manager.add_rtx_lidar()
    sensor_manager.add_camera(freq=5)
    raw_env.sensor_manager = sensor_manager  # 一定要设置到 raw_env，而不是 wrapper

    # 5. 使用包装器初始化环境 → 自动调用 initialize()、构建 obs_manager
    # env = RslRlVecEnvWrapper(raw_env)
    # raw_env.initialize()


    # 6. 执行 reset，准备运行
    raw_env.reset()
    #ready_env = env.envs[0]

    # 7. 推进若干步模拟
    # from isaacsim import simulation_app
    for i in range(100):
        simulation_app.update()
        import time; time.sleep(0.05)

    # 8. 获取观测结果（包含 robot + lidar + depth）
    obs = raw_env.compute_observations()
    print("obs keys:", obs.keys())
    if "robot_state" in obs:
        print("robot_state:", obs["robot_state"].shape)
    if "lidar" in obs:
        print("lidar:", obs["lidar"].shape)
    if "depth" in obs:
        print("depth:", obs["depth"].shape)

    simulation_app.close()

if __name__ == "__main__":
    main()
