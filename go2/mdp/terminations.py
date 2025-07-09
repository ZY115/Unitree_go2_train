# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`omni.isaac.lab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def terrain_out_of_bounds(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), distance_buffer: float = 3.0
) -> torch.Tensor:
    """Terminate when the actor move too close to the edge of the terrain.

    If the actor moves too close to the edge of the terrain, the termination is activated. The distance
    to the edge of the terrain is calculated based on the size of the terrain and the distance buffer.
    """
    if env.scene.cfg.terrain.terrain_type == "plane":
        return False  # we have infinite terrain because it is a plane
    elif env.scene.cfg.terrain.terrain_type == "generator":
        # obtain the size of the sub-terrains
        terrain_gen_cfg = env.scene.terrain.cfg.terrain_generator
        grid_width, grid_length = terrain_gen_cfg.size
        n_rows, n_cols = terrain_gen_cfg.num_rows, terrain_gen_cfg.num_cols
        border_width = terrain_gen_cfg.border_width
        # compute the size of the map
        map_width = n_rows * grid_width + 2 * border_width
        map_height = n_cols * grid_length + 2 * border_width

        # extract the used quantities (to enable type-hinting)
        asset: RigidObject = env.scene[asset_cfg.name]

        # check if the agent is out of bounds
        x_out_of_bounds = torch.abs(asset.data.root_pos_w[:, 0]) > 0.5 * map_width - distance_buffer
        y_out_of_bounds = torch.abs(asset.data.root_pos_w[:, 1]) > 0.5 * map_height - distance_buffer
        return torch.logical_or(x_out_of_bounds, y_out_of_bounds)
    else:
        raise ValueError("Received unsupported terrain type, must be either 'plane' or 'generator'.")

# def base_contact(env, sensor_cfg: SceneEntityCfg, body_names="base"):
#     contact_sensor = env.scene.sensors[sensor_cfg.name]
#     # base_ids = contact_sensor.find_body_ids(body_names)
#     print("ContactSensor.data keys:",dir(contact_sensor.data))
#     is_contact = contact_sensor.data.in_contact.any(dim=1)
#     return is_contact

def base_contact(env, sensor_cfg: SceneEntityCfg, force_threshold=1.0):
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    force_norm = (contact_sensor.data.net_forces_w ** 2).sum(-1).sqrt()
    is_contact = (force_norm > force_threshold).any(dim=1)
    print("Base contact triggered:", is_contact.cpu().numpy())
    print("force_norm shape:", force_norm.shape)
    print("force_norm value:",force_norm.cpu().numpy())
    return is_contact

def base_fallen(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("unitree_go2"), max_pitch=1.0, max_roll=1.0):

    base = env.scene[asset_cfg.name]
    # 获取四元数
    quat = base.data.root_quat_w[:, :]
    quat_cpu = quat.detach().cpu().numpy()
    # 转欧拉角
    import torch
    from scipy.spatial.transform import Rotation as R
    # 转numpy再转euler
    quat_np = quat.cpu().numpy()
    euler = R.from_quat(quat_np[:, [1, 2, 3, 0]]).as_euler('xyz')  # 注意顺序！
    pitch = euler[:, 1]
    roll = euler[:, 0]
    pitch = torch.from_numpy(pitch).to(quat.device)
    roll = torch.from_numpy(roll).to(quat.device)
    result = (pitch.abs() > max_pitch) | (roll.abs() > max_roll)
    print(f"pitch: {pitch.cpu().numpy()}, roll: {roll.cpu().numpy()}, result: {result.cpu().numpy()}")
    # 超过阈值就True
    return result