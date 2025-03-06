# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg

from . import observations as obs

from isaaclab.envs.mdp import *

import time


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def robot_posture_height_bonus(env: ManagerBasedRLEnv, threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    "Reward for robot posture height, giving a -ve reward if the posture height is not above threshold."
    posture_heights = obs.get_robot_height(env, asset_cfg).squeeze(-1)
    return (posture_heights < threshold).float()

def upright_posture_bonus(
    env: ManagerBasedRLEnv, threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward for maintaining an upright posture."""
    up_proj = obs.base_up_proj(env, asset_cfg).squeeze(-1)
    return (up_proj > threshold).float()

def root_vel(
    env: ManagerBasedRLEnv, target_lin_vel: float, target_ang_vel: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward for moving with more than threshold speed."""
    lin_vel = base_lin_vel(env, asset_cfg).squeeze(-1)
    lin_vel_sq = torch.sum(lin_vel**2, dim=-1)
    return (lin_vel_sq > float(target_lin_vel)**2).float()

def move_to_target_bonus(
    env: ManagerBasedRLEnv,
    threshold: float,
    target_pos: tuple[float, float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for moving to the target heading."""
    heading_proj = obs.base_heading_proj(env, target_pos, asset_cfg).squeeze(-1)
    return torch.where(heading_proj > threshold, 1.0, heading_proj / threshold)

def total_robot_height_below_minimum(
    env: ManagerBasedRLEnv,
    threshold: float,
    feet_distance: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    "Terminate if the total heigh of the robot is below threshold"
    pelvis, lt, rt = obs.get_robot_height(env, asset_cfg)
    posture_heights = torch.max(pelvis - lt, pelvis - rt)

    ret = posture_heights < threshold

    t_dist = lt - rt
    ret &= torch.abs(torch.sum(t_dist, dim=-1)) > feet_distance
    return ret


class progress_reward(ManagerTermBase):
    """Reward for making progress towards the target."""

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        # initialize the base class
        super().__init__(cfg, env)
        # create history buffer
        self.potentials = torch.zeros(env.num_envs, device=env.device)
        self.prev_potentials = torch.zeros_like(self.potentials)

    def reset(self, env_ids: torch.Tensor):
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = self._env.scene["robot"]
        # compute projection of current heading to desired heading vector
        target_pos = torch.tensor(self.cfg.params["target_pos"], device=self.device)
        to_target_pos = target_pos - asset.data.root_pos_w[env_ids, :3]
        # reward terms
        self.potentials[env_ids] = -torch.norm(to_target_pos, p=2, dim=-1) / self._env.step_dt
        self.prev_potentials[env_ids] = self.potentials[env_ids]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        target_pos: tuple[float, float, float],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[asset_cfg.name]
        # compute vector to target
        target_pos = torch.tensor(target_pos, device=env.device)
        to_target_pos = target_pos - asset.data.root_pos_w[:, :3]
        to_target_pos[:, 2] = 0.0
        # update history buffer and compute new potential
        self.prev_potentials[:] = self.potentials[:]
        self.potentials[:] = -torch.norm(to_target_pos, p=2, dim=-1) / env.step_dt

        return self.potentials - self.prev_potentials
    

class cross_legs(ManagerTermBase):

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        super().__init__(env, cfg)
        self.legs = torch.zeros(env.num_envs, device=env.device)
        self.prev_legs = torch.zeros(env.num_envs, device=env.device)
        self.times = torch.zeros(env._num_envs, device=env.device)

    def reset(self, env_ids: torch.Tensor):
        asset: Articulation = self._env.scene["robot"]
        self.times[env_ids] = time.time()
        self.legs[env_ids] = 0.0

    def __call__(
            self,
            env: ManagerBasedRLEnv,
            asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        curr_time = time.time()
        curr_pos = obs.get_robot_y_vals(env, asset_cfg)
        ids = torch.where(self.legs != curr_pos)
        self.times[ids] = curr_time
        self.legs = curr_pos

        return self.times - curr_time - 1.0

