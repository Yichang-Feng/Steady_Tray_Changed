import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg
from isaaclab.utils import configclass
from steadytray.tasks import mdp

from .compat import DoneTerm

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from .locomotion_env_cfg import RobotEnvCfg, RobotSceneCfg, EventCfg, RewardsCfg, TerminationsCfg, ObservationsCfg

# Initial positions for the tray and tray holders relative to the robot's pelvis
TRAY_INITIAL_POS = [0.25, 0.0, 0.167]


@configclass
class TraySceneCfg(RobotSceneCfg):
    """Configuration for the tray scene."""


    tray: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Tray",
        spawn=sim_utils.CuboidCfg(
            size=(0.25, 0.500, 0.02),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.4),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.5, 
                dynamic_friction=1.0,
                restitution=0.0
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(),
    )
    tray_contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Tray", 
        track_air_time=True,
        history_length=10,
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/Robot/left_rubber_hand", 
            "{ENV_REGEX_NS}/Robot/right_rubber_hand"
        ],
    )
    robot_transform: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",  # 躯干为参考系
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Tray",
                name="tray",
            ),
        ],
    )


@configclass
class TrayEventCfg(EventCfg):
    """Configuration for the tray events."""

    random_tray_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("tray"),
            "static_friction_range": (1.2, 2.0),
            "dynamic_friction_range": (1.0, 1.8),
            "restitution_range": (0.0, 0.05),
            "num_buckets": 256,
        },
    )

    random_tray_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("tray"),
            "mass_distribution_params": (0.3, 0.7),
            "operation": "abs",
        },
    )

    reset_tray_pos = EventTerm(
        func=mdp.set_rigid_object_relative_to_robot,
        mode="reset",
        params={
            "base_asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
            "target_asset_cfg": SceneEntityCfg("tray"),
            "relative_pose": {
                "x": TRAY_INITIAL_POS[0],
                "y": TRAY_INITIAL_POS[1],
                "z": TRAY_INITIAL_POS[2],
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
            },
            "relative_velocity": {
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
            },
        },
    )

@configclass
class TrayObservationsCfg(ObservationsCfg):
    """Configuration for observations in the steady tray environment (for Residual Adapter)."""

    @configclass
    class EncoderCfg(ObsGroup):
        """喂给残差网络 Transformer 编码器的时间序列观测"""
        
        # 基础机器人观测
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2), clip=(-25.0, 25.0))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5), clip=(-100.0, 100.0))
        last_action = ObsTerm(func=mdp.last_action)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1), clip=(-3.0, 3.0))

        # 【核心输入】：托盘相对于机器人的位姿与姿态投影
        tray_projected_gravity = ObsTerm(func=mdp.projected_gravity, params={"asset_cfg": SceneEntityCfg("tray")}, noise=Unoise(n_min=-0.05, n_max=0.05))
        tray_pos_rel = ObsTerm(func=mdp.object_rel_pos, params={"sensor_cfg": SceneEntityCfg("robot_transform"), "target_frame_name": "tray"}, noise=Unoise(n_min=-0.03, n_max=0.03), clip=(-1.0, 1.0))

        def __post_init__(self):
            self.history_length = 32
            self.enable_corruption = True
            # 【关键】：必须为 False，确保时序维度存在 (Seq_len = 32)
            self.flatten_history_dim = False 

    encoder: EncoderCfg = EncoderCfg()

    @configclass
    class AdaptedCriticCfg(ObsGroup):
        """包含特权信息的评论家网络观测"""
        
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-25.0, 25.0))  
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, clip=(-100.0, 100.0))
        last_action = ObsTerm(func=mdp.last_action)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-3.0, 3.0))

        tray_projected_gravity = ObsTerm(func=mdp.projected_gravity, params={"asset_cfg": SceneEntityCfg("tray")})
        tray_pos_rel = ObsTerm(func=mdp.object_rel_pos, params={"sensor_cfg": SceneEntityCfg("robot_transform"), "target_frame_name": "tray"}, clip=(-1.0, 1.0))
        tray_ang_vel_rel = ObsTerm(func=mdp.object_rel_ang_vel, params={"target_asset_cfg": SceneEntityCfg("tray"), "reference_asset_cfg": SceneEntityCfg("robot", body_names="torso_link")}, scale=0.2, clip=(-50.0, 50.0))
        tray_lin_vel_rel = ObsTerm(func=mdp.object_rel_lin_vel, params={"target_asset_cfg": SceneEntityCfg("tray"), "reference_asset_cfg": SceneEntityCfg("robot", body_names="torso_link")}, scale=0.5, clip=(-10.0, 10.0))
        rubber_hand_contact_forces = ObsTerm(func=mdp.tray_holder_contact_forces, params={"sensor_cfg": SceneEntityCfg("tray_contact_sensor")}, scale=0.1, clip=(-50.0, 50.0))

        def __post_init__(self):
            self.history_length = 5
            self.flatten_history_dim = True

    critic: AdaptedCriticCfg = AdaptedCriticCfg()

@configclass
class TrayRewardsCfg(RewardsCfg):
    """Configuration for the tray rewards."""

    # Release locomotion penalty to allow more natural movement
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_exp,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_.*_joint",
                    ".*_elbow_joint",
                    ".*_wrist_.*",
                ],
            ),
            "lambda_exp": 0.3,
        },    
    )     # MAX = 0.2

    # Reward for keeping the tray stable
    tray_flat_orientation = RewTerm(
        func=mdp.object_upright_bonus_exp,
        weight=0.25,
        params={"object_cfg": SceneEntityCfg('tray'), "lambda_exp": 4.0},
    )   # MAX = 0.25 x 2 = 0.5
    tray_lin_vel = RewTerm(
        func=mdp.object_lin_vel_z_exp, 
        weight=0.2,
        params={"object_cfg": SceneEntityCfg("tray"), "lambda_exp": 2.0}
    )     # MAX = 0.2
    tray_ang_vel = RewTerm(
        func=mdp.object_ang_vel_xy_exp, 
        weight=0.2,
        params={"object_cfg": SceneEntityCfg("tray"), "lambda_exp": 1.0}
    )     # MAX = 0.2

    tray_contact = RewTerm(
        func=mdp.desired_contacts_count,
        weight=0.01,
        params={
            "sensor_cfg": SceneEntityCfg("tray_contact_sensor"),
            "threshold": 0.1,
        },
    )   # MAX = 0.01 x 2 x 10 = 0.2
    tray_contact_force = RewTerm(
        func=mdp.contact_force_exp,
        weight=0.2,
        params={
            "sensor_cfg": SceneEntityCfg("tray_contact_sensor"),
            "lambda_exp": 0.005,
        },
    )   # MAX = 0.2

    left_relative_quat_deviation = RewTerm(
        func=mdp.entity_quat_exp,
        weight=0.5,
        params={
            "entity1_cfg": SceneEntityCfg('tray'),
            "entity2_cfg": SceneEntityCfg('robot', body_names='left_rubber_hand'),
            "lambda_exp": 2.0,
        },
    )
    right_relative_quat_deviation = RewTerm(
        func=mdp.entity_quat_exp,
        weight=0.5,
        params={
            "entity1_cfg": SceneEntityCfg('tray'),
            "entity2_cfg": SceneEntityCfg('robot', body_names='right_rubber_hand'),
            "lambda_exp": 2.0,
        },
    )

    # penalty for using too much torque
    torque_penalty = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-2e-5,
        params={"asset_cfg": SceneEntityCfg("robot")}
    )


@configclass
class TrayTerminationsCfg(TerminationsCfg):
    """Configuration for the tray termination conditions."""

    tray_fallen = DoneTerm(
        func=mdp.link_height_below_minimum,
        params={
            "minimum_height": 0.7,  # Terminate if tray drops below 0.7m (more strict)
            "asset_cfg": SceneEntityCfg("tray"),
        },
        track_only=True,
        track_only_delay=1.0,
    )


@configclass
class SteadyTrayEnvCfg(RobotEnvCfg):
    """Configuration for the steady tray environment."""

    scene: TraySceneCfg = TraySceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)
    events: TrayEventCfg = TrayEventCfg()
    observations: TrayObservationsCfg = TrayObservationsCfg()  
    rewards: TrayRewardsCfg = TrayRewardsCfg()
    terminations: TrayTerminationsCfg = TrayTerminationsCfg()

@configclass
class TrayTerminationsPlayCfg(TerminationsCfg):
    """Configuration for the tray termination conditions."""
    base_height = DoneTerm(
        func=mdp.root_height_below_minimum, 
        params={"minimum_height": 0.4},
        track_only=True,
        track_only_delay=0.0,
    )
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation, 
        params={"limit_angle": 0.7, "asset_cfg": SceneEntityCfg("robot", body_names="torso_link")},
        track_only=True,
        track_only_delay=0.0,
    )
    tray_fallen = DoneTerm(
        func=mdp.link_height_below_minimum,
        params={
            "minimum_height": 0.7,  # Terminate if tray drops below 0.7m (more strict)
            "asset_cfg": SceneEntityCfg("tray"),
        },
        track_only=True,
        track_only_delay=0.0,
    )

@configclass
class SteadyTrayPlayEnvCfg(SteadyTrayEnvCfg):

    terminations: TrayTerminationsPlayCfg = TrayTerminationsPlayCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 5

        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
