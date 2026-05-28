import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from steadytray.tasks import mdp
import math
# 从 locomotion_env_cfg 导入基础的机器人和运动配置
from .locomotion_env_cfg import RobotEnvCfg, RobotSceneCfg, EventCfg, RewardsCfg, TerminationsCfg, ObservationsCfg


@configclass
class ArmPostureSceneCfg(RobotSceneCfg):
    """第一阶段场景配置：彻底移除托盘与物体，只保留机器人本体。"""
    
    def __post_init__(self):
        super().__post_init__()
        
        # 确保字典存在
        if self.robot.init_state.joint_pos is None:
            self.robot.init_state.joint_pos = {}
            
        # 1. 找出并移除基础配置中原有的手臂和手腕相关的键，防止正则冲突
        keys_to_remove = [
            k for k in self.robot.init_state.joint_pos.keys() 
            if "shoulder" in k or "elbow" in k or "wrist" in k
        ]
        for k in keys_to_remove:
            self.robot.init_state.joint_pos.pop(k)
        
        # 2. 注入我们新的统一正则规则
        self.robot.init_state.joint_pos.update({
            ".*_shoulder_pitch_joint": 0.0,
            ".*_shoulder_roll_joint": 0.0,
            ".*_shoulder_yaw_joint": 0.0,
            ".*_elbow_joint": 0.0,
            ".*_wrist_.*": 0.0,
        })


@configclass
class ArmPostureRewardsCfg(RewardsCfg):
    """第一阶段奖励配置：下肢走得稳、上肢姿态定。"""

    joint_deviation_arms = None

    gated_arm_posture = RewTerm(
        func=mdp.locomotion_gated_arm_posture_exp,
        weight=0.2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_.*",
                    ".*_elbow_joint",  # 确保正则匹配到肘部
                    ".*_wrist_.*",
                ],
            ),
            "command_name": "base_velocity",
            "std": math.sqrt(0.25),
            "lambda_exp": 1.0,
        },
    )

    # 并且将权重进一步调小，允许手臂网络适度发力对抗重力
    torque_penalty = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-0.00001, 
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_.*",
                    ".*_elbow_.*",
                    ".*_wrist_.*",
                ],
            )
        }
    )

    arm_action_silence = None

@configclass
class ArmPostureEnvCfg(RobotEnvCfg):
    """第一阶段训练环境主配置。"""
    
    scene: ArmPostureSceneCfg = ArmPostureSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)
    events: EventCfg = EventCfg()  # 复用基础的出生随机化和重置事件
    rewards: ArmPostureRewardsCfg = ArmPostureRewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()  # 复用摔倒等基础终止条件
    observations: ObservationsCfg = ObservationsCfg()  # 仅包含机器人本体的感受空间

    def __post_init__(self):
        super().__post_init__()
        # 默认训练时将命令采样范围设置为动作/任务限制范围，
        # 否则默认 ranges 可能过小导致速度命令始终很小（见训练问题）。
        try:
            self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        except Exception:
            # 若配置结构不同或不存在，不抛出错误，仅保留默认行为
            pass


@configclass
class ArmPosturePlayEnvCfg(ArmPostureEnvCfg):
    
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 5

        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
