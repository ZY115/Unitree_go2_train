import omni
import numpy as np
from pxr import Gf
import omni.replicator.core as rep
from omni.isaac.sensor import Camera
import omni.isaac.core.utils.numpy.rotations as rot_utils

class SensorManager:
    def __init__(self, num_envs):
        self.num_envs = num_envs
        ####
        self.lidar_annotators = []
        self.cameras = []
        ####

    def add_rtx_lidar(self):
        lidar_annotators = []
        for env_idx in range(self.num_envs):
            _, sensor = omni.kit.commands.execute(
                "IsaacSensorCreateRtxLidar",
                path="/lidar",
                parent=f"/World/envs/env_{env_idx}/Go2/base",
                config="Hesai_XT32_SD10",
                # config="Velodyne_VLS128",
                translation=(0.2, 0, 0.2),
                orientation=Gf.Quatd(1.0, 0.0, 0.0, 0.0),  # Gf.Quatd is w,i,j,k
            )

            annotator = rep.AnnotatorRegistry.get_annotator("RtxSensorCpuIsaacCreateRTXLidarScanBuffer")
            hydra_texture = rep.create.render_product(sensor.GetPath(), [1, 1], name="Isaac")
            annotator.attach(hydra_texture.path)
            lidar_annotators.append(annotator)
            ####
        self.lidar_annotators = lidar_annotators
        ####
        return lidar_annotators

    def add_camera(self, freq):
        from omni.isaac.core.simulation_context import SimulationContext
        sim_dt = SimulationContext.instance().get_physics_dt()
        render_interval = SimulationContext.instance().get_rendering_dt() / sim_dt
        render_freq = 1.0 / (sim_dt * render_interval)

        if render_freq % freq != 0:
            # fallback to a compatible frequency
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
            camera.add_distance_to_image_plane_to_frame()
            cameras.append(camera)
        #####
        self.cameras = cameras
        #####
        return cameras
    
    # def get_lidar_obs(self):
    #     lidar_datas = []
    #     for annotator in self.lidar_annotators:
    #         data = annotator.get_data()  # 按照你的 annotator 实际API修改
    #         lidar_datas.append(data)
    #     return np.stack(lidar_datas, axis=0)

    # def get_depth_obs(self):
    #     depth_datas = []
    #     for cam in self.cameras:
    #         depth_img = cam.get_depth_data()  # 按你的 Camera 实现修改
    #         depth_datas.append(depth_img)
    #     return np.stack(depth_datas, axis=0)

    def get_lidar_obs(self):
        lidar_datas = []
        for annotator in self.lidar_annotators:
            pre_data = annotator.get_data()  # 确认API，必要时 flatten
            data = pre_data["data"]
            lidar_datas.append(data.flatten())  # flatten很关键
        return np.concatenate(lidar_datas, axis=0)
    

    #####################2025 6.19
    def get_depth_obs(self):
        depth_datas = []
        for cam in self.cameras:
            depth_img = cam.get_depth() 
            if depth_img is not None:
                depth_datas.append(depth_img.flatten())

        if len(depth_datas) == 0:
            return None                        # 让调用者知道还没数据
        return np.concatenate(depth_datas, axis=0)


