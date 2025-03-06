# import argparse

# from isaaclab.app import AppLauncher

# # add argparse arguments
# parser = argparse.ArgumentParser(description="Training for wheeled quadruped robot.")
# parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")

# # append AppLauncher cli args
# AppLauncher.add_app_launcher_args(parser)
# # parse the arguments
# args_cli = parser.parse_args()

# # launch omniverse app
# app_launcher = AppLauncher(args_cli)
# simulation_app = app_launcher.app


# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ManagerBasedRLEnv
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

import isaaclab_tasks.manager_based.classic.cassie_omniverse.mdp as mdp

from .terrains import PYRAMID_STAIRS

##
# Scene definition
##
def robot_configuration():
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Agility/Cassie/cassie.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=None,
                # retain_accelerations=False,
                # linear_damping=0.0,
                # angular_damping=0.0,
                # max_linear_velocity=1000.0,
                # max_angular_velocity=1000.0,
                max_depenetration_velocity=10.0,
                enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0,sleep_threshold=0.005,
                stabilization_threshold=0.001
            ),
            copy_from_source=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.9),
            joint_pos={
            "hip_abduction_left": 0.1,
            "hip_rotation_left": 0.0,
            "hip_flexion_left": 1.0,
            "thigh_joint_left": -1.8,
            "ankle_joint_left": 1.57,
            "toe_joint_left": -1.57,
            "hip_abduction_right": -0.1,
            "hip_rotation_right": 0.0,
            "hip_flexion_right": 1.0,
            "thigh_joint_right": -1.8,
            "ankle_joint_right": 1.57,
            "toe_joint_right": -1.57,
        },
        ),
        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=["hip_.*", "thigh_.*", "ankle_.*"],
                # effort_limit=200.0,
                # velocity_limit=10.0,
                stiffness={
                    "hip_abduction.*": 20.0,
                    "hip_rotation.*": 20.0,
                    "hip_flexion.*": 1.0,
                    "thigh_joint.*": 20.0,
                    "ankle_joint.*": 5.0,
                },
                damping={
                    "hip_abduction.*": 3.0,
                    "hip_rotation.*": 3.0,
                    "hip_flexion.*": 3.0,
                    "thigh_joint.*": 2.0,
                    "ankle_joint.*": 1.0,
                },
            ),
            "toes": ImplicitActuatorCfg(
                joint_names_expr=["toe_.*"],
                # effort_limit=20.0,
                # velocity_limit=10.0,
                stiffness={
                    "toe_joint.*": 5.0,
                },
                damping={
                    "toe_joint.*": 1.0,
                },
            ),
        },
    )


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a humanoid robot."""

    # terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
        debug_vis=False,
    )

    # robot
    robot = robot_configuration()

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class MyScenePyramidStairsCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a humanoid robot."""

    # terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        max_init_terrain_level=None,
        terrain_type="generator",
        terrain_generator=PYRAMID_STAIRS,
        visual_material=None,
        debug_vis=False,
    )

    # robot
    robot = robot_configuration()

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_effort = mdp.JointEffortActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale={
            "hip_abduction_.*": 45.0,
            "hip_rotation_.*": 67.5,
            "hip_flexion_.*": 150,
            "thigh_joint_.*": 120,
            "ankle_joint_.*": 150,
            "toe_joint_.*": 120
        },
    )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy."""

        base_height = ObsTerm(func=mdp.base_pos_z)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
        base_yaw_roll = ObsTerm(func=mdp.base_yaw_roll)
        base_angle_to_target = ObsTerm(func=mdp.base_angle_to_target, params={"target_pos": (1000.0, 0.0, 0.0)})
        base_up_proj = ObsTerm(func=mdp.base_up_proj)
        base_heading_proj = ObsTerm(func=mdp.base_heading_proj, params={"target_pos": (1000.0, 0.0, 0.0)})
        joint_pos_norm = ObsTerm(func=mdp.joint_pos_limit_normalized)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.1)
        feet_body_forces = ObsTerm(
            func=mdp.body_incoming_wrench,
            scale=0.01,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["left_toe", "right_toe"])},
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {}, "velocity_range": {}},
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.2, 0.2),
            "velocity_range": (-0.1, 0.1),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Reward for moving forward
    progress = RewTerm(func=mdp.progress_reward, weight=1.0, params={"target_pos": (1000.0, 0.0, 0.0)})
    # (2) Stay alive bonus
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # (3) Reward for non-upright posture
    upright = RewTerm(func=mdp.upright_posture_bonus, weight=0.1, params={"threshold": 0.7})
    # (4) Reward for moving in the right direction
    move_to_target = RewTerm(
        func=mdp.move_to_target_bonus, weight=0.1, params={"threshold": 0.8, "target_pos": (1000.0, 0.0, 0.0)}
    )
    velocity_penalty = RewTerm(
        func=mdp.root_vel, weight=-2.0, params={"target_lin_vel": 0.5, "target_ang_vel": None}
    )
    # keep_legs_close = RewTerm(
    #     func=mdp.legs_straight_reward, weight=0.1
    # )
    # terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)


@configclass
class PyramidStairsRewardsCfg:
    """Reward terms for the MDP."""

    # (1) Reward for moving forward
    progress = RewTerm(func=mdp.progress_reward, weight=1.0, params={"target_pos": (1000.0, 0.0, 0.0)})
    # (2) Stay alive bonus
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # (3) Reward for non-upright posture
    upright = RewTerm(func=mdp.robot_posture_height_bonus, weight=-0.5, params={"threshold": 0.7})
    # (4) Reward for moving in the right direction
    move_to_target = RewTerm(
        func=mdp.move_to_target_bonus, weight=0.5, params={"threshold": 0.8, "target_pos": (1000.0, 0.0, 0.0)}
    )
    cross_legs = RewTerm(
        func=mdp.cross_legs, weight = 0.5
    )
    # velocity_penalty = RewTerm(
    #     func=mdp.root_vel, weight=-0.5, params={"target_lin_vel": 0.7, "target_ang_vel": None}
    # )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Terminate if the episode length is exceeded
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Terminate if the robot falls
    torso_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.4})


@configclass
class PyramidStairsTerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Terminate if the episode length is exceeded
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Terminate if the robot falls
    torso_height1 = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.4})
    # (2) Terminate if the robot falls
    torso_height = DoneTerm(func=mdp.total_robot_height_below_minimum, params={"threshold": 0.4, "feet_distance": 0.9})


@configclass
class CassieGroundPlaneEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the MuJoCo-style Humanoid walking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=1024, env_spacing=5.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 16.0
        # simulation settings
        self.sim.dt = 1 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        # default friction material
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.restitution = 0.0

@configclass
class CassiePyramidStairsPlaneEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the MuJoCo-style Humanoid walking environment."""

    # Scene settings
    scene: MyScenePyramidStairsCfg = MyScenePyramidStairsCfg(num_envs=2048, env_spacing=5.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: PyramidStairsTerminationsCfg = PyramidStairsTerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 16.0
        # simulation settings
        self.sim.dt = 1 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        # default friction material
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.restitution = 0.0
