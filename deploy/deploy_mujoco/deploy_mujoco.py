import time
import argparse
import os
import sys
import mujoco.viewer
import mujoco
import numpy as np
import torch
from collections import deque
import pygame
import cv2
from pupil_apriltags import Detector
import math

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Add parent directory to path for common imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.policy_runner import compute_policy_action, detect_policy_type, detect_encoder_obs_size
from scripts.config import Config
from scipy.spatial.transform import Rotation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


def get_object_pose(data, model, object_half_height=0.05):
    """
    Get object pose in camera frame from MuJoCo simulation.
    
    Args:
        data: MuJoCo data object
        model: MuJoCo model object
        object_half_height: Half of the object height in meters (default: 0.05m = 5cm for half height)
                           This is added to get the top surface position instead of center
    
    Returns:
        object_obs: Position + quaternion observation array (7,)
    """
    # Get camera pose in world frame
    cam_site_id = model.site("d435_camera_frame").id
    cam_pos_world = data.site_xpos[cam_site_id]
    cam_rot_world = data.site_xmat[cam_site_id].reshape(3, 3)

    # Get object pose in world frame (center of object)
    object_body_id = model.body("object").id
    object_pos_world = data.xpos[object_body_id]
    object_rot_world = data.xmat[object_body_id].reshape(3, 3)
    
    # Add half height to get top surface position
    # The offset is in the object's local frame (z-axis points up in object frame)
    top_surface_offset_local = np.array([0, 0, object_half_height], dtype=np.float32)
    top_surface_offset_world = object_rot_world @ top_surface_offset_local
    object_top_pos_world = object_pos_world + top_surface_offset_world

    # Build transformation matrices (using top surface position)
    object_world_transform = np.eye(4)
    object_world_transform[:3, :3] = object_rot_world
    object_world_transform[:3, 3] = object_top_pos_world  # Use top surface position

    camera_world_transform = np.eye(4)
    camera_world_transform[:3, :3] = cam_rot_world
    camera_world_transform[:3, 3] = cam_pos_world

    # Transform object pose to camera frame
    object_camera_transform = np.linalg.inv(camera_world_transform) @ object_world_transform
    object_camera_pos = object_camera_transform[:3, 3]  # This is now the top surface position
    object_camera_rotation = Rotation.from_matrix(object_camera_transform[:3, :3])
    
    # Convert to wxyz quaternion format
    object_camera_quat_xyzw = object_camera_rotation.as_quat()
    object_camera_quat = np.array([object_camera_quat_xyzw[3], object_camera_quat_xyzw[0], 
                                   object_camera_quat_xyzw[1], object_camera_quat_xyzw[2]], dtype=np.float32)

    # Position + quaternion (3+4)
    object_obs = np.concatenate([object_camera_pos, object_camera_quat], axis=0).astype(np.float32)
    
    return object_obs

# 手柄控制器类
class GamepadController:
    def __init__(self, max_vx=1.0, max_vy=0.5, max_yaw=1.0, deadzone=0.1):
        # 强制 SDL2 库在没有窗口焦点时，依然监听手柄事件
        os.environ["SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS"] = "1"
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        
        pygame.init()
        pygame.joystick.init()
        
        self.joystick = None
        self.max_vx = max_vx
        self.max_vy = max_vy
        self.max_yaw = max_yaw
        self.deadzone = deadzone
        self.debug_counter = 0
        
        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            print(f"[Info] 已连接手柄: {self.joystick.get_name()}，共有 {self.joystick.get_numaxes()} 个轴")
        else:
            print("[Warning] 未检测到手柄，将使用默认配置指令。")

    def get_command(self, default_cmd):
        if self.joystick is None:
            return default_cmd

        # 刷新硬件事件队列
        pygame.event.pump()

        raw_vx = -self.joystick.get_axis(1) 
        raw_vy = -self.joystick.get_axis(0) 
        
        num_axes = self.joystick.get_numaxes()
        raw_yaw = -self.joystick.get_axis(3) if num_axes > 3 else 0.0 

        # 调试输出
        self.debug_counter += 1
        if self.debug_counter % 50 == 0:
            print(f"[Debug] 当前输入 -> 前进/后退(vx): {raw_vx:.2f}, 平移(vy): {raw_vy:.2f}, 转向(yaw): {raw_yaw:.2f}")

        # 死区过滤
        vx = raw_vx if abs(raw_vx) > self.deadzone else 0.0
        vy = raw_vy if abs(raw_vy) > self.deadzone else 0.0
        yaw = raw_yaw if abs(raw_yaw) > self.deadzone else 0.0

        cmd = np.array([
            vx * self.max_vx,
            vy * self.max_vy,
            yaw * self.max_yaw
        ], dtype=np.float32)
        
        return cmd

def get_object_pose_from_vision(img_gray, detector, cam_params, tag_size=0.05):
    """
    通过 AprilTag 从图像中推导物体在相机坐标系下的位姿
    """
    # 运行 AprilTag 检测
    detections = detector.detect(img_gray, estimate_tag_pose=True, 
                                 camera_params=cam_params, tag_size=tag_size)
    
    if len(detections) == 0:
        return None  # 视野内无二维码
    
    det = detections[0]
    pos = det.pose_t.flatten() 
    rot_mat = det.pose_R       
    
    # 1. 核心修正：OpenCV 到 MuJoCo Site 坐标系的映射
    # 映射关系推导：X_site = Z_cv, Y_site = -Y_cv, Z_site = X_cv
    T_cv2mj = np.array([
        [0.0,  0.0,  1.0],
        [0.0, -1.0,  0.0],
        [1.0,  0.0,  0.0]
    ], dtype=np.float32)

    # 2. 物体局部坐标系转换 (保持与原物体 Z 轴向上对齐)
    T_tag2obj = np.array([
        [1.0,  0.0,  0.0],
        [0.0, -1.0,  0.0],
        [0.0,  0.0, -1.0]
    ], dtype=np.float32)

    # 计算 MuJoCo 体系下的平移和旋转
    pos_mj = T_cv2mj @ pos
    rot_mj = T_cv2mj @ rot_mat @ np.linalg.inv(T_tag2obj)
    
    # 转换为 wxyz 四元数
    rotation = Rotation.from_matrix(rot_mj)
    quat_xyzw = rotation.as_quat()
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)
    
    object_obs = np.concatenate([pos_mj, quat_wxyz], axis=0).astype(np.float32)
    return object_obs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='g1 deploy mujoco')
    parser.add_argument('--policy', type=str, default="exported/policy_9999.pt",
                       help='Direct path to policy file')
    parser.add_argument('--config', type=str, default="deploy/configs/g1_29dof_walk.yaml",
                       help='Direct path to config file (overrides default config)')
    parser.add_argument('--encoder_seq_len', type=int, default=32,
                       help='Encoder sequence length (number of history frames for distillation policies)')

    # 手柄相关参数
    parser.add_argument('--use_gamepad', action='store_true', 
                       help='Enable gamepad control. If not set, uses config.cmd_init')
    parser.add_argument('--max_vx', type=float, default=1.0, help='Max forward velocity')
    parser.add_argument('--max_vy', type=float, default=0.5, help='Max lateral velocity')
    parser.add_argument('--max_yaw', type=float, default=1.0, help='Max yaw rate')

    args = parser.parse_args()

    # Load configuration using shared Config class
    config = Config(args.config)
    
    # Get policy path
    if args.policy is not None:
        policy_path = args.policy
    else:
        raise ValueError("Policy path must be provided via command line argument --policy")

    if os.path.exists(policy_path):
        print(f"Using policy: {policy_path}")
    else:
        raise FileNotFoundError(f"Policy file not found: {policy_path}")

    # define context variables
    action = np.zeros(config.num_actions, dtype=np.float32)
    obs = np.zeros(config.num_obs, dtype=np.float32)

    counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(config.xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = config.simulation_dt

    # --- 初始化视觉渲染和 AprilTag ---
    cam_name = "d435_camera"
    cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    width, height = 640, 480
    renderer = mujoco.Renderer(m, height=height, width=width)
    
    # 推导相机内参 (fx, fy, cx, cy) 供 AprilTag 库使用
    fovy = m.cam_fovy[cam_id]
    f = 0.5 * height / math.tan(fovy * math.pi / 360)
    cam_params = [f, f, width / 2, height / 2]
    
    detector = Detector(families='tag36h11')
    last_valid_object_obs = np.array([0.312, -0.004, 0.037, 0.746, 0.237, -0.326, -0.531], dtype=np.float32) # 默认位姿缓冲
    # --------------------------------------

    default_angles = config.default_angles[config.policy_to_robot]
    target_dof_pos = default_angles.copy()

    frame_stack = deque(maxlen=5)
    for _ in range(5):
        frame_stack.append(obs.copy())
        mujoco.mj_step(m, d) 


    # Load policy
    policy = torch.jit.load(policy_path)
    policy_type = detect_policy_type(policy)
    print(f"Policy type: {policy_type}")

    # Auto-detect encoder observation size for distillation policies
    encoder_obs_dim = None
    
    if policy_type == 'distillation':
        encoder_obs_dim = detect_encoder_obs_size(policy)

    print(f"Using MuJoCo ground-truth object observations")

    # Initialize encoder frame stack for distillation policies
    encoder_frame_stack = deque(maxlen=args.encoder_seq_len)
    if encoder_obs_dim is not None:
        for _ in range(args.encoder_seq_len):
            encoder_frame_stack.append(np.zeros(encoder_obs_dim, dtype=np.float32))
    current_cmd = config.cmd_init.copy()
    if args.use_gamepad:
        gamepad = GamepadController(
            max_vx=args.max_vx, 
            max_vy=args.max_vy, 
            max_yaw=args.max_yaw
        )

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Set up camera to follow the robot
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        # Track the pelvis/base body (usually body id 1, adjust if needed)
        # You can find the correct body by looking at the robot's URDF/XML structure
        viewer.cam.trackbodyid = 1  # Changed from 0 (world) to 1 (pelvis/base)
        viewer.cam.distance = 2.5   # Distance from robot (increased for better view)
        viewer.cam.elevation = -20  # Camera angle (negative looks down)
        viewer.cam.azimuth = 90     # Side view angle
        viewer.cam.lookat[:] = [0, 0, 0.5]  # Look at point offset
        
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        last_push_time = 0.0
        while viewer.is_running() and time.time() - start < config.simulation_duration:
            step_start = time.time()
            current_sim_time = time.time() - start

            # # --- 添加推力干扰：每隔 4 秒推一次 ---
            # if current_sim_time - last_push_time > 3.0:
            #     # 瞬间给 Y 轴（侧向）增加 0.5 m/s 的速度
            #     d.qvel[1] += 0
            #     # 如果想往前推，可以修改 X 轴： d.qvel[0] += 0.5
            #     print(f"[{current_sim_time:.2f}s] Pushed!! Current velocity after push: {d.qvel[0]:.2f} (forward), {d.qvel[1]:.2f} (sideways)")
            #     last_push_time = current_sim_time
            # # ------------------------------------

            tau = pd_control(target_dof_pos, d.qpos[7:7 + config.num_actions], config.kps, np.zeros_like(config.kds), d.qvel[6:6 + config.num_actions], config.kds)
            d.ctrl[:] = tau
            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)

            counter += 1
            if counter % config.control_decimation == 0:
                # Apply control signal here.

                start_compute = time.time()
                
                # 读取手柄更新指令
                if args.use_gamepad:
                    current_cmd = gamepad.get_command(config.cmd_init)
                else:
                    current_cmd = config.cmd_init

                # Get sensor data (explicitly use float32 for consistency with real robot)
                qj = d.qpos[7:7 + config.num_actions].astype(np.float32)
                dqj = d.qvel[6:6 + config.num_actions].astype(np.float32)
                quat = d.qpos[3:7].astype(np.float32)
                omega = d.qvel[3:6].astype(np.float32)

                # Get object observations from MuJoCo simulation
                object_obs = None
                
                if encoder_obs_dim is not None:
                    # 1. 渲染图像
                    renderer.update_scene(d, camera=cam_name)
                    img_rgb = renderer.render()

                    # # ---保存几帧画面查看相机视角---
                    # control_step = counter // config.control_decimation
                    # if control_step == 10 or control_step == 50:
                    #     # OpenCV 保存图片需要 BGR 格式
                    #     cv2.imwrite(f"debug_vision_view_{control_step}.jpg", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
                    #     print(f"已保存 debug_vision_view_{control_step}.jpg 用于检查相机视角")
                    # # ----------------------------------------------

                    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
                    
                    # 2. 视觉解算
                    vision_obs = get_object_pose_from_vision(img_gray, detector, cam_params, tag_size=0.0465)# tag_size 需要根据实际物体大小调整
                    
                    # 3. 异常处理
                    if vision_obs is not None:
                        object_obs = vision_obs
                        last_valid_object_obs = vision_obs
                    else:
                        # 视野被遮挡或丢失目标时，沿用上一帧数据
                        object_obs = last_valid_object_obs
                    
                # --- 打印视觉解算的物体位姿 ---
                # 计算真值
                gt_obs = get_object_pose(d, m)
                
                if vision_obs is not None:
                    object_obs = vision_obs
                    last_valid_object_obs = vision_obs
                    
                    # 每50步打印一次 GT 与 Vision 的对比
                    control_step = counter // config.control_decimation
                    if control_step % 50 == 0:
                        print(f"[{current_sim_time:.2f}s]")
                        print(f"真实值 : Pos {gt_obs[:3].round(3)} | Quat {gt_obs[3:].round(3)}")
                        print(f"解算值 : Pos {vision_obs[:3].round(3)} | Quat {vision_obs[3:].round(3)}")
                        print("-" * 40)
                else:
                    object_obs = last_valid_object_obs

                # Compute policy action using shared function
                action, target_dof_pos = compute_policy_action(
                    policy=policy,
                    frame_stack=frame_stack,
                    qj=qj,
                    dqj=dqj,
                    quat=quat,
                    omega=omega,
                    cmd=current_cmd,
                    previous_action=action,
                    config=config,
                    object_obs=object_obs,
                    policy_type=policy_type,
                    encoder_frame_stack=encoder_frame_stack
                )
                compute_time = time.time() - start_compute
                if compute_time > config.control_dt:
                    print(f"Warning: Policy compute time {compute_time:.6f} seconds exceeds control_dt {config.control_dt} seconds")

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
