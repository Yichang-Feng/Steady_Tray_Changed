import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from steadytray.tasks import mdp

from .steady_tray_env_cfg import SteadyTrayEnvCfg, TraySceneCfg, TrayTerminationsPlayCfg, TrayRewardsCfg
from .locomotion_env_cfg import ObservationsCfg
from isaaclab.managers import RewardTermCfg as RewTerm

@configclass
class SeparatedTraySceneCfg(TraySceneCfg):
    """Configuration for the separated tray scene."""

    robot_transform: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",  # Reference frame
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Tray",
                name="tray",
            ),
        ],
    )


@configclass
class SeparatedTrayObservationsCfg(ObservationsCfg):
    """Configuration for observations in the separated tray environment."""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        """Observations for policy group."""

        # Add tray observations
        tray_projected_gravity = ObsTerm(func=mdp.projected_gravity, params={"asset_cfg": SceneEntityCfg("tray")}, noise=Unoise(n_min=-0.05, n_max=0.05))
        tray_pos_rel = ObsTerm(func=mdp.object_rel_pos, params={"sensor_cfg": SceneEntityCfg("robot_transform"), "target_frame_name": "tray"}, noise=Unoise(n_min=-0.03, n_max=0.03), clip=(-1.0, 1.0))
        tray_ang_vel_rel = ObsTerm(func=mdp.object_rel_ang_vel, params={"target_asset_cfg": SceneEntityCfg("tray"), "reference_asset_cfg": SceneEntityCfg("robot", body_names="torso_link")}, scale=0.2, clip=(-50.0, 50.0))
        tray_lin_vel_rel = ObsTerm(func=mdp.object_rel_lin_vel, params={"target_asset_cfg": SceneEntityCfg("tray"), "reference_asset_cfg": SceneEntityCfg("robot", body_names="torso_link")}, scale=0.5, clip=(-10.0, 10.0))

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObservationsCfg.CriticCfg):
        """Observations for critic group."""

        # Add tray observations without noise
        tray_projected_gravity = ObsTerm(func=mdp.projected_gravity, params={"asset_cfg": SceneEntityCfg("tray")})
        tray_pos_rel = ObsTerm(func=mdp.object_rel_pos, params={"sensor_cfg": SceneEntityCfg("robot_transform"), "target_frame_name": "tray"}, clip=(-1.0, 1.0))
        tray_ang_vel_rel = ObsTerm(func=mdp.object_rel_ang_vel, params={"target_asset_cfg": SceneEntityCfg("tray"), "reference_asset_cfg": SceneEntityCfg("robot", body_names="torso_link")}, scale=0.2, clip=(-50.0, 50.0))
        tray_lin_vel_rel = ObsTerm(func=mdp.object_rel_lin_vel, params={"target_asset_cfg": SceneEntityCfg("tray"), "reference_asset_cfg": SceneEntityCfg("robot", body_names="torso_link")}, scale=0.5, clip=(-10.0, 10.0))

    critic: CriticCfg = CriticCfg()


@configclass
class SeparatedTrayRewardsCfg(TrayRewardsCfg):
    """Configuration for rewards in the separated tray environment."""
    
    # 奖励左手保持在托盘左侧下方
    left_hand_tray_pos = RewTerm(
        func=mdp.entity_relative_pos_exp,
        weight=1.0,
        params={
            "entity1_cfg": SceneEntityCfg("tray"),
            "entity2_cfg": SceneEntityCfg("robot", body_names="left_rubber_hand"),
            "target_relative_pos": (0.0, 0.20, 0.00), # 托盘坐标系下的位置：Y负方向（左侧）
            "lambda_exp": 10.0,
            "ignore_z": True,
        }
    )
    
    # 奖励右手保持在托盘右侧下方
    right_hand_tray_pos = RewTerm(
        func=mdp.entity_relative_pos_exp,
        weight=1.0,
        params={
            "entity1_cfg": SceneEntityCfg("tray"),
            "entity2_cfg": SceneEntityCfg("robot", body_names="right_rubber_hand"),
            "target_relative_pos": (0.0, -0.20, 0.00), # 托盘坐标系下的位置：Y负方向（右侧）
            "lambda_exp": 10.0,
            "ignore_z": True,
        }
    )
    
    # 奖励双手保持一定距离
    hands_distance = RewTerm(
        func=mdp.hands_distance_exp,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_rubber_hand", "right_rubber_hand"]),
            "target_distance": 0.40,
            "lambda_exp": 5.0,
        }
    )


@configclass
class SteadyTraySeparatedEnvCfg(SteadyTrayEnvCfg):
    """Configuration for the steady separated tray environment."""

    scene: SeparatedTraySceneCfg = SeparatedTraySceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=False)
    observations: SeparatedTrayObservationsCfg = SeparatedTrayObservationsCfg()
    rewards: SeparatedTrayRewardsCfg = SeparatedTrayRewardsCfg()


@configclass
class SteadyTraySeparatedPlayEnvCfg(SteadyTraySeparatedEnvCfg):
    """Configuration for the steady separated tray play environment."""

    terminations: TrayTerminationsPlayCfg = TrayTerminationsPlayCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 5

        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
