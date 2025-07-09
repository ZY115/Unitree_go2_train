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


########################
    # def add_rtx_lidar(self):
    #     lidar_annotators = []
    #     for env_idx in range(self.num_envs):
    #         lidar_path = f"/World/envs/env_{env_idx}/Go2/base/lidar"
    #         _, sensor = omni.kit.commands.execute(
    #             "IsaacSensorCreateRtxLidar",
    #             path=lidar_path,
    #             parent=f"/World/envs/env_{env_idx}/Go2/base",
    #             config="Hesai_XT32_SD10",
    #             translation=(0.2, 0, 0.2),
    #             orientation=Gf.Quatd(1.0, 0.0, 0.0, 0.0),
    #         )
    #         annotator = rep.AnnotatorRegistry.get_annotator("RtxSensorCpuIsaacCreateRTXLidarScanBuffer")
    #         hydra_texture = rep.create.render_product(sensor.GetPath(), [1, 1], name=f"IsaacLidar{env_idx}")
    #         annotator.attach(hydra_texture.path)
    #         lidar_annotators.append(annotator)
    #     self.lidar_annotators = lidar_annotators
    #     print("add_rtx_lidar called, lidar_annotators:", self.lidar_annotators)
    #     return lidar_annotators
#########################
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
                resolution=(64, 48),
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

    # def get_lidar_obs(self):
    #     if not self.lidar_annotators:
    #         self.add_rtx_lidar()
    #     lidar_datas = []
    #     for annotator in self.lidar_annotators:

    #         pre_data = annotator.get_data()  # 确认API，必要时 flatten
    #         data = pre_data["data"]
    #         pts = pre_data.get("data")
    #         print("LIDAR shape:", pts.shape if pts is not None else None)
    #         if pts is None or pts.size == 0:
    #             return np.array([], dtype=np.float32)
            
    #         lidar_datas.append(pts.reshape(-1).astype(np.float32))
    #         data = pre_data.get("data")
    #         print("雷达数据：", data)
    #         lidar_datas.append(data.flatten())  # flatten很关键
    
    #     return np.concatenate(lidar_datas, axis=0)


    ####################


    def get_lidar_obs(self, num_points=40000):

        if not hasattr(self, "lidar_annotators") or not self.lidar_annotators:
            self.add_rtx_lidar()
        lidar_datas = []
        for annotator in self.lidar_annotators:
            pre_data = annotator.get_data()
            pts = pre_data.get("data")
            print("LIDAR shape:", pts.shape if pts is not None else None)
            if pts is None or pts.size == 0:

                fixed_pts = np.zeros((num_points, 3), dtype=np.float32)
            else:
                pts = pts.astype(np.float32)
                if pts.shape[0] >= num_points:
                    idx = np.random.choice(pts.shape[0], num_points, replace=False)
                    fixed_pts = pts[idx]
                else:
                    fixed_pts = np.zeros((num_points, 3), dtype=np.float32)
                    fixed_pts[:pts.shape[0]] = pts
            lidar_datas.append(fixed_pts.reshape(-1))  # flatten to (num_points*3,)
        lidar_obs = np.concatenate(lidar_datas, axis=0)
        print(f"采样后LIDAR shape: {lidar_obs.shape}")
        print("前10个点:", lidar_obs[:30]) 
        return lidar_obs




    # def get_lidar_obs(self):
    #     if not self.lidar_annotators:
    #         self.add_rtx_lidar()
    #     lidar_datas = []
    #     for annotator in self.lidar_annotators:
    #         pre_data = annotator.get_data()
    #         pts = pre_data.get("data")
    #         print("LIDAR shape:", pts.shape if pts is not None else None)
    #         if pts is not None and pts.size > 0:
    #             lidar_datas.append(pts.reshape(-1).astype(np.float32))
    #     # 注意：一定不能直接 return np.zeros(1)；要用相同 shape
    #     if len(lidar_datas) == 0:
    #         # **永远不要 return 空 array，也不要 return None**
    #         return np.zeros((1,), dtype=np.float32)
    #     arr = np.concatenate(lidar_datas, axis=0)
    #     print("get_lidar_obs called, lidar_annotators:", self.lidar_annotators)

    #     # 如果你希望 obs shape 固定（比如 400729），可以填充
    #     # arr_pad = np.zeros((400729,), dtype=np.float32)
    #     # arr_pad[:arr.size] = arr[:400729]
    #     # return arr_pad
    #     return arr


    #####################2025 6.19
    # def get_depth_obs(self):
        
    #     depth_datas = []
    #     for cam in self.cameras:
    #         depth_img = cam.get_depth() 
    #         # print(depth_img)
    #         # print("深度相机的数据")
    #         if depth_img is not None:
    #             depth_datas.append(depth_img.flatten())
    #     if len(depth_datas) == 0:
    #         return None                        # 让调用者知道还没数据
        
    #     return np.concatenate(depth_datas, axis=0)



    def get_depth_obs(self):
        depth_datas = []
        for cam in self.cameras:
            depth_img = cam.get_depth()
            if depth_img is not None:
                clean_img = np.nan_to_num(depth_img, nan=0.0, posinf=100, neginf=-100)
                depth_datas.append(clean_img.flatten())
        
        if len(depth_datas) == 0:
            return None  # 没有数据，调用者要检查
        
        depth_vec = np.concatenate(depth_datas, axis=0)
        # 再次保障
        depth_vec = np.nan_to_num(depth_vec, nan=0.0, posinf=100, neginf=-100)
        
        assert not np.any(np.isnan(depth_vec)), "NaN in depth obs!"
        assert not np.any(np.isinf(depth_vec)), "Inf in depth obs!"
        
        return depth_vec
