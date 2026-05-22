import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from steadytray.tasks import mdp

# 从 locomotion_env_cfg 导入基础的机器人和运动配置
from .locomotion_env_cfg import RobotEnvCfg, RobotSceneCfg, EventCfg, RewardsCfg, TerminationsCfg, ObservationsCfg


@configclass
class ArmPostureSceneCfg(RobotSceneCfg):
    """第一阶段场景配置：彻底移除托盘与物体，只保留机器人本体。"""
    # 继承自 RobotSceneCfg，会自动加载在 g1_delay.py 中定义好的标准 G1 机器人资产
    pass


@configclass
class ArmPostureRewardsCfg(RewardsCfg):
    """第一阶段奖励配置：聚焦于下肢走得稳、上肢姿态定。"""

    # 【关键重写】强制关闭父类中默认的低权重 L1 手臂惩罚，避免与新的指数奖励冲突
    joint_deviation_arms = None

    # 【核心奖励】强力约束手臂维持在默认的平举/手心向上姿态
    arm_posture_tracking = RewTerm(
        func=mdp.joint_deviation_exp,
        weight=2.5,  # 给予非常高的权重，迫使机器人在高优先级下满足控臂约束
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_joint",
                    ".*_wrist_roll_joint",
                    ".*_wrist_pitch_joint",
                    ".*_wrist_yaw_joint",
                ],
            ),
            "lambda_exp": 1.0,  # 对关节偏离的敏感度
        },
    )

    # 辅助惩罚：压制手臂关节的运动速度，防止手臂由于高频抖动带来仿真不稳定
    arm_velocity_penalty = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_.*",
                    ".*_elbow_joint",
                    ".*_wrist_.*",
                ],
            )
        },
    )

    # 惩罚过大的扭矩输出，保证动作顺滑
    torque_penalty = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-2e-5,
        params={"asset_cfg": SceneEntityCfg("robot")}
    )

@configclass
class ArmPostureEnvCfg(RobotEnvCfg):
    """第一阶段训练环境主配置。"""
    
    scene: ArmPostureSceneCfg = ArmPostureSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)
    events: EventCfg = EventCfg()  # 复用基础的出生随机化和重置事件
    rewards: ArmPostureRewardsCfg = ArmPostureRewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()  # 复用摔倒等基础终止条件
    observations: ObservationsCfg = ObservationsCfg()  # 仅包含机器人本体的感受空间


@configclass
class ArmPosturePlayEnvCfg(ArmPostureEnvCfg):
    
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 5

        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
