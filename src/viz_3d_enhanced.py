import cv2
from PIL import Image, ImageDraw, ImageFont
import platform
import numpy as np
import time
from viz_vb_data import (CombinedVisualizer, ROBOT_IDS, resize_with_label, 
                         hstack_with_sep, create_combined_image)
import trimesh
import pyrender
from scipy.spatial.transform import Rotation
import os

class Enhanced3DVisualizer(CombinedVisualizer):
    """3D增强可视化器 / Enhanced 3D Visualizer"""
    
    # Language dictionary for bilingual support
    LANG = {
        'zh': {
            'window_title': '机器人监控',
            'platform': '平台',
            'arrow_keys': '方向键',
            'controls': '控制',
            'frame_nav': '前后帧',
            'next_episode': '下一个Episode',
            'prev_episode': '上一个Episode',
            'auto_play': '自动播放（Episode结束后自动跳转下一个）',
            'speed': '速度',
            'cam_rotate_h': '相机水平旋转',
            'cam_rotate_v': '相机俯仰旋转',
            'cam_zoom': '相机缩放',
            'cam_z_translate': '相机Z轴平移（需先按T）',
            'cam_reset': '相机重置',
            'cam_translate_mode': '相机平移模式',
            'screenshot': '截图',
            'show_raw_transform': '显示Robot{0}原始变换',
            'show_gripper_mesh': '显示夹爪网格',
            'show_finger_mesh': '显示手指网格',
            'show_finger_axes': '显示手指坐标轴',
            'switch_language': '切换语言 (中/En)',
            'quit': '退出',
            'ep_complete': 'Episode {0} 完成，自动加载下一个...',
            'play_complete': '播放完成',
            'jump_to_ep': '跳转到 Episode {0}',
            'last_episode': '已经是最后一个Episode',
            'first_episode': '已经是第一个Episode',
            'auto_play_on': '自动播放: 开启',
            'auto_play_off': '自动播放: 关闭',
            'cam_reset_msg': '相机重置',
            'cam_target_z': '相机目标Z: {0:.3f}',
            'cam_translate_on': '相机平移模式: 开启',
            'cam_translate_off': '相机平移模式: 关闭',
            'screenshot_saved': '截图: {0}',
            'raw_transform_show': 'Robot{0}原始变换: 显示',
            'raw_transform_hide': 'Robot{0}原始变换: 隐藏',
            'gripper_mesh_show': '夹爪网格: 显示',
            'gripper_mesh_hide': '夹爪网格: 隐藏',
            'finger_mesh_show': '手指网格: 显示',
            'finger_mesh_hide': '手指网格: 隐藏',
            'finger_axes_show': '手指坐标轴: 显示',
            'finger_axes_hide': '手指坐标轴: 隐藏',
            'exiting': '退出',
            'realtime_trajectory': '实时轨迹',
            '3d_world_view': '3D世界视图',
            'language_switched': '语言已切换: 中文',
            # Panel text labels
            'robot_0_position': 'Robot 0 位置 (m)',
            'robot_1_position': 'Robot 1 位置 (m)',
            'gripper_width': '夹爪宽度 (m)',
            'robot': '机器人',
            'visual': '视觉',
            'left_tactile': '左触觉',
            'right_tactile': '右触觉',
            'ep': '回合',
            'frame': '帧',
        },
        'en': {
            'window_title': 'Robot Monitor',
            'platform': 'Platform',
            'arrow_keys': 'Arrow Keys',
            'controls': 'Controls',
            'frame_nav': 'Frame Navigation',
            'next_episode': 'Next Episode',
            'prev_episode': 'Previous Episode',
            'auto_play': 'Auto Play (auto next episode)',
            'speed': 'Speed',
            'cam_rotate_h': 'Camera Rotate Horizontal',
            'cam_rotate_v': 'Camera Rotate Vertical',
            'cam_zoom': 'Camera Zoom',
            'cam_z_translate': 'Camera Z-axis Translation (T mode)',
            'cam_reset': 'Camera Reset',
            'cam_translate_mode': 'Camera Translation Mode',
            'screenshot': 'Screenshot',
            'show_raw_transform': 'Show Robot{0} Raw Transform',
            'show_gripper_mesh': 'Show Gripper Mesh',
            'show_finger_mesh': 'Show Finger Mesh',
            'show_finger_axes': 'Show Finger Axes',
            'switch_language': 'Switch Language (中/En)',
            'quit': 'Quit',
            'ep_complete': 'Episode {0} complete, loading next...',
            'play_complete': 'Playback complete',
            'jump_to_ep': 'Jump to Episode {0}',
            'last_episode': 'Already at last episode',
            'first_episode': 'Already at first episode',
            'auto_play_on': 'Auto Play: ON',
            'auto_play_off': 'Auto Play: OFF',
            'cam_reset_msg': 'Camera Reset',
            'cam_target_z': 'Camera Target Z: {0:.3f}',
            'cam_translate_on': 'Camera Translation Mode: ON',
            'cam_translate_off': 'Camera Translation Mode: OFF',
            'screenshot_saved': 'Screenshot: {0}',
            'raw_transform_show': 'Robot{0} Raw Transform: SHOW',
            'raw_transform_hide': 'Robot{0} Raw Transform: HIDE',
            'gripper_mesh_show': 'Gripper Mesh: SHOW',
            'gripper_mesh_hide': 'Gripper Mesh: HIDE',
            'finger_mesh_show': 'Finger Mesh: SHOW',
            'finger_mesh_hide': 'Finger Mesh: HIDE',
            'finger_axes_show': 'Finger Axes: SHOW',
            'finger_axes_hide': 'Finger Axes: HIDE',
            'exiting': 'Exiting',
            'realtime_trajectory': 'Real-time Trajectory',
            '3d_world_view': '3D World View',
            'language_switched': 'Language switched: English',
            # Panel text labels
            'robot_0_position': 'Robot 0 Position (m)',
            'robot_1_position': 'Robot 1 Position (m)',
            'gripper_width': 'Gripper Width (m)',
            'robot': 'Robot',
            'visual': 'Visual',
            'left_tactile': 'L-Tact',
            'right_tactile': 'R-Tact',
            'ep': 'Ep',
            'frame': 'Frame',
        }
    }
    
    def __init__(self, *args, **kwargs):
        self.window_name = "Robot Monitor"
        self.playback_speed = 1.0
        self.auto_next_episode = True  # 自动播放下一个Episode
        self._world_scene = None
        self._world_dynamic_nodes = []
        self._world_camera_pose = None
        self._cached_axis_mesh = None
        self._cached_wrist_mesh = None
        self._cached_base_mesh = None
        self._cached_finger_mesh = None  # Legacy single mesh
        self._cached_left_finger_mesh = None  # Separate left finger mesh
        self._cached_right_finger_mesh = None  # Separate right finger mesh
        self._cached_controller_meshes = None
        self._cached_gripper_mesh_left = None
        self._cached_gripper_mesh_right = None
        self._cached_controller_meshes = {}
        self._show_raw_transforms = {0: False, 1: False}
        self._show_controller_mesh = True
        self._show_claw_finger_mesh = True
        self._show_finger_axes = False
        self._world_camera_node = None
        self._world_light_nodes = []
        self._camera_translation_mode = False
        self._camera_yaw = 0.0
        self._camera_pitch = 0.0
        self._camera_distance = 0.6
        self._camera_target = np.zeros(3, dtype=np.float64)
        
        # Language settings (default: Chinese)
        self._language = 'zh'
        
        # Cross-platform arrow key codes
        # OpenCV returns different codes on Windows vs Linux
        self._setup_arrow_key_codes()
        
        super().__init__(*args, **kwargs)
    
    def _t(self, key):
        """Translate text based on current language"""
        return self.LANG[self._language].get(key, key)
    
    def _safe_print(self, text):
        """Print text safely, handling encoding issues on different platforms"""
        try:
            print(text)
        except UnicodeEncodeError:
            # Fallback to ASCII-safe output if terminal doesn't support Unicode
            print(text.encode('ascii', 'replace').decode('ascii'))
    
    def _put_text_unicode(self, img, text, position, font_size=20, color=(255, 255, 255), font_path=None):
        """
        Put Unicode text (including Chinese) on OpenCV image using PIL
        
        Args:
            img: OpenCV image (BGR format)
            text: Text to display (can contain Chinese characters)
            position: (x, y) tuple for text position
            font_size: Font size in pixels
            color: Text color in BGR format (default white)
            font_path: Path to TrueType font file (optional)
        
        Returns:
            Modified image
        """
        # Convert OpenCV BGR to PIL RGB
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        # Try to load a font that supports Chinese
        try:
            if font_path and os.path.exists(font_path):
                font = ImageFont.truetype(font_path, font_size)
            else:
                # Try common Chinese fonts on different platforms
                font_candidates = [
                    # Linux
                    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
                    "/usr/share/fonts/truetype/arphic/uming.ttc",
                    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                    # macOS
                    "/System/Library/Fonts/PingFang.ttc",
                    "/System/Library/Fonts/STHeiti Light.ttc",
                    # Windows (if running under WSL or Wine)
                    "C:/Windows/Fonts/msyh.ttc",
                    "C:/Windows/Fonts/simhei.ttf",
                ]
                
                font = None
                for font_candidate in font_candidates:
                    if os.path.exists(font_candidate):
                        font = ImageFont.truetype(font_candidate, font_size)
                        break
                
                if font is None:
                    # Fallback to default font (doesn't support Chinese well)
                    font = ImageFont.load_default()
        except Exception as e:
            # If all fails, use default font
            font = ImageFont.load_default()
        
        # Draw text
        draw.text(position, text, font=font, fill=color[::-1])  # BGR to RGB
        
        # Convert back to OpenCV BGR
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
    def _switch_language(self):
        """Toggle between Chinese and English"""
        self._language = 'en' if self._language == 'zh' else 'zh'
        self._safe_print(self._t('language_switched'))
        # Re-display controls in new language
        self._display_startup_info()
    
    def _display_startup_info(self):
        """Display startup information with current language"""
        system = platform.system()
        self._safe_print("\n" + "=" * 70)
        self._safe_print(f"  {self._t('window_title')}")
        self._safe_print("=" * 70)
        self._safe_print(f"  {self._t('platform')}: {system}")
        self._safe_print(f"  {self._t('arrow_keys')}: Left={self.KEY_LEFT}, Up={self.KEY_UP}, Right={self.KEY_RIGHT}, Down={self.KEY_DOWN}")
        self._safe_print(f"  {self._t('controls')}:")
        self._safe_print(f"    [A/D]    {self._t('frame_nav')}")
        self._safe_print(f"    [W]      {self._t('next_episode')}")
        self._safe_print(f"    [S]      {self._t('prev_episode')}")
        self._safe_print(f"    [P]      {self._t('auto_play')}")
        self._safe_print(f"    [1-5]    {self._t('speed')}: 0.25x, 0.5x, 1x, 2x, 5x")
        self._safe_print(f"    [←/→]    {self._t('cam_rotate_h')}")
        self._safe_print(f"    [↑/↓]    {self._t('cam_rotate_v')}")
        self._safe_print(f"    [+/-]    {self._t('cam_zoom')}")
        self._safe_print(f"    [./]     {self._t('cam_z_translate')}")
        self._safe_print(f"    [0]      {self._t('cam_reset')}")
        self._safe_print(f"    [T]      {self._t('cam_translate_mode')}")
        self._safe_print(f"    [L]      {self._t('switch_language')}")
        self._safe_print(f"    [C]      {self._t('screenshot')}")
        self._safe_print(f"    [U]      {self._t('show_raw_transform').format(0)}")
        self._safe_print(f"    [I]      {self._t('show_raw_transform').format(1)}")
        self._safe_print(f"    [M]      {self._t('show_gripper_mesh')}")
        self._safe_print(f"    [F]      {self._t('show_finger_mesh')}")
        self._safe_print(f"    [G]      {self._t('show_finger_axes')}")
        self._safe_print(f"    [Q]      {self._t('quit')}")
        self._safe_print("=" * 70 + "\n")
    
    def _setup_arrow_key_codes(self):
        """Setup arrow key codes for cross-platform compatibility"""
        system = platform.system()
        
        if system == 'Windows':
            # Windows arrow key codes (extended keys)
            self.KEY_LEFT = 2424832    # 0x250000
            self.KEY_UP = 2490368      # 0x260000
            self.KEY_RIGHT = 2555904   # 0x270000
            self.KEY_DOWN = 2621440    # 0x280000
        else:
            # Linux/macOS arrow key codes
            self.KEY_LEFT = 81
            self.KEY_UP = 82
            self.KEY_RIGHT = 83
            self.KEY_DOWN = 84

    def setup_renderers(self):
        super().setup_renderers()
        self._init_world_scene_cache()

    def _init_world_scene_cache(self):
        """初始化并缓存世界场景、相机和静态网格"""
        from viz_vb_data import (_axis_mesh, _quest_controller_mesh)

        scene = pyrender.Scene(bg_color=[0.05, 0.08, 0.12, 1.0])

        floor_grid = self._create_floor_grid_solid(size=1.5, step=0.2)
        if floor_grid:
            scene.add(floor_grid)

        coord_axes = self._create_coordinate_axes(length=0.2)
        for axis_mesh in coord_axes:
            scene.add(axis_mesh)

        self._cached_axis_mesh = _axis_mesh(size=0.05)

        wrist = trimesh.creation.cylinder(radius=0.02, height=0.03, sections=16)
        wrist.visual.vertex_colors = np.array([180, 180, 180, 255], dtype=np.uint8)
        self._cached_wrist_mesh = pyrender.Mesh.from_trimesh(wrist, smooth=True)

        base = trimesh.creation.box(extents=[0.08, 0.04, 0.02])
        base.visual.vertex_colors = np.array([200, 200, 200, 255], dtype=np.uint8)
        self._cached_base_mesh = pyrender.Mesh.from_trimesh(base, smooth=False)

        # Legacy single finger mesh (fallback)
        finger_path = 'src/meshes/finger.STL'
        if not os.path.exists(finger_path):
            finger_path = 'src/meshes/gripper.STL'
        if os.path.exists(finger_path):
            try:
                finger_mesh = trimesh.load(finger_path)
                finger_mesh.apply_scale(0.001)
                finger_mesh.visual.vertex_colors = np.array([150, 150, 150, 255], dtype=np.uint8)
                self._cached_finger_mesh = pyrender.Mesh.from_trimesh(finger_mesh, smooth=True)
            except Exception as exc:
                print(f"指尖网格加载失败: {exc}")
                self._cached_finger_mesh = None
        
        # Load separate left/right finger meshes from calibration if available
        self._cached_left_finger_mesh = None
        self._cached_right_finger_mesh = None
        
        # Try to load from robot-specific calibrations
        from viz_vb_data import _load_finger_calibrations_per_robot
        calib_per_robot = _load_finger_calibrations_per_robot()
        
        if calib_per_robot:
            # Use robot 0 (left gripper) meshes
            left_calib = calib_per_robot.get(0)
            if left_calib:
                if left_calib.get("left_finger_mesh") is not None:
                    self._cached_left_finger_mesh = pyrender.Mesh.from_trimesh(
                        left_calib["left_finger_mesh"], smooth=True
                    )
                if left_calib.get("right_finger_mesh") is not None:
                    self._cached_right_finger_mesh = pyrender.Mesh.from_trimesh(
                        left_calib["right_finger_mesh"], smooth=True
                    )

        # load gripper meshes
        left_system_path = 'src/meshes/left_no_finger.STL'
        right_system_path = 'src/meshes/right_no_finger.STL'
        assert os.path.exists(left_system_path), f"缺少左手夹爪STL文件: {left_system_path}"
        assert os.path.exists(right_system_path), f"缺少右手夹爪STL文件: {right_system_path}"

        left_system_mesh = trimesh.load(left_system_path)
        left_system_mesh.apply_scale(0.001)
        left_system_mesh.visual.vertex_colors = np.array([180, 180, 180, 255], dtype=np.uint8)
        self._cached_gripper_mesh_left = pyrender.Mesh.from_trimesh(left_system_mesh, smooth=True)

        right_system_mesh = trimesh.load(right_system_path)
        right_system_mesh.apply_scale(0.001)
        right_system_mesh.visual.vertex_colors = np.array([180, 180, 180, 255], dtype=np.uint8)
        self._cached_gripper_mesh_right = pyrender.Mesh.from_trimesh(right_system_mesh, smooth=True)


        self._cached_controller_meshes = {
            0: _quest_controller_mesh(is_left=False),
            1: _quest_controller_mesh(is_left=True),
        }

        self._world_scene = scene
        self._world_dynamic_nodes = []
        self._reset_camera_controls()

    def _reset_camera_controls(self):
        default_eye = np.array([0.15, -0.3, 0.4], dtype=np.float64)
        self._camera_target = np.zeros(3, dtype=np.float64)
        direction = default_eye - self._camera_target
        self._camera_distance = float(np.linalg.norm(direction))
        if self._camera_distance < 1e-6:
            self._camera_distance = 0.6
            direction = np.array([0.15, -0.3, 0.4], dtype=np.float64)
        self._camera_yaw = float(np.arctan2(direction[1], direction[0]))
        self._camera_pitch = float(np.arcsin(direction[2] / self._camera_distance))
        self._refresh_world_camera()

    def _refresh_world_camera(self):
        if self._world_scene is None:
            return
        from viz_vb_data import _lookat_camera_pose
        if self._world_camera_node is not None:
            self._world_scene.remove_node(self._world_camera_node)
            self._world_camera_node = None
        for node in self._world_light_nodes:
            self._world_scene.remove_node(node)
        self._world_light_nodes = []

        max_pitch = np.deg2rad(89.0)
        self._camera_pitch = float(np.clip(self._camera_pitch, -max_pitch, max_pitch))
        dist = max(self._camera_distance, 0.05)
        eye_offset = np.array([
            dist * np.cos(self._camera_pitch) * np.cos(self._camera_yaw),
            dist * np.cos(self._camera_pitch) * np.sin(self._camera_yaw),
            dist * np.sin(self._camera_pitch),
        ], dtype=np.float64)
        eye = self._camera_target + eye_offset
        cam_pose = _lookat_camera_pose(eye, self._camera_target, [0, 0, 1])
        self._world_camera_pose = cam_pose
        camera = pyrender.PerspectiveCamera(yfov=np.deg2rad(60.0))
        self._world_camera_node = self._world_scene.add(camera, pose=cam_pose)

        main_light = pyrender.DirectionalLight(color=np.ones(3), intensity=5.0)
        self._world_light_nodes.append(self._world_scene.add(main_light, pose=cam_pose))

        fill_pose = cam_pose.copy()
        fill_pose[:3, 3] = -cam_pose[:3, 3]
        fill_light = pyrender.DirectionalLight(color=[0.8, 0.9, 1.0], intensity=2.0)
        self._world_light_nodes.append(self._world_scene.add(fill_light, pose=fill_pose))
    
    
    def _create_floor_grid_solid(self, size=2.0, step=0.2):
        """创建地面网格"""
        meshes = []
        radius = 0.002
        
        for y in np.arange(-size, size + step, step):
            line = trimesh.creation.cylinder(radius=radius, height=size*2, sections=8)
            line.visual.vertex_colors = np.array([100, 120, 140, 255], dtype=np.uint8)
            rot_tf = trimesh.transformations.rotation_matrix(np.pi/2, [0, 1, 0])
            line.apply_transform(rot_tf)
            line.apply_translation([0, y, -0.5])
            meshes.append(line)
        
        for x in np.arange(-size, size + step, step):
            line = trimesh.creation.cylinder(radius=radius, height=size*2, sections=8)
            line.visual.vertex_colors = np.array([100, 120, 140, 255], dtype=np.uint8)
            rot_tf = trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0])
            line.apply_transform(rot_tf)
            line.apply_translation([x, 0, -0.5])
            meshes.append(line)
        
        if meshes:
            combined = trimesh.util.concatenate(meshes)
            return pyrender.Mesh.from_trimesh(combined, smooth=False)
        return None
    
    def _create_coordinate_axes(self, length=0.2):
        """创建RGB坐标轴"""
        meshes = []
        radius = 0.004
        
        # X轴（红色）
        x_cyl = trimesh.creation.cylinder(radius=radius, height=length, sections=12)
        x_cyl.visual.vertex_colors = np.array([255, 0, 0, 255], dtype=np.uint8)
        x_tf = trimesh.transformations.rotation_matrix(np.pi/2, [0, 1, 0])
        x_tf[:3, 3] = [length/2, 0, 0]
        x_cyl.apply_transform(x_tf)
        meshes.append(pyrender.Mesh.from_trimesh(x_cyl, smooth=True))
        
        x_cone = trimesh.creation.cone(radius=radius*2, height=0.02, sections=12)
        x_cone.visual.vertex_colors = np.array([255, 0, 0, 255], dtype=np.uint8)
        x_cone_tf = trimesh.transformations.rotation_matrix(np.pi/2, [0, 1, 0])
        x_cone_tf[:3, 3] = [length + 0.01, 0, 0]
        x_cone.apply_transform(x_cone_tf)
        meshes.append(pyrender.Mesh.from_trimesh(x_cone, smooth=True))
        
        # Y轴（绿色）
        y_cyl = trimesh.creation.cylinder(radius=radius, height=length, sections=12)
        y_cyl.visual.vertex_colors = np.array([0, 255, 0, 255], dtype=np.uint8)
        y_tf = trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0])
        y_tf[:3, 3] = [0, length/2, 0]
        y_cyl.apply_transform(y_tf)
        meshes.append(pyrender.Mesh.from_trimesh(y_cyl, smooth=True))
        
        y_cone = trimesh.creation.cone(radius=radius*2, height=0.02, sections=12)
        y_cone.visual.vertex_colors = np.array([0, 255, 0, 255], dtype=np.uint8)
        y_cone_tf = trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0])
        y_cone_tf[:3, 3] = [0, length + 0.01, 0]
        y_cone.apply_transform(y_cone_tf)
        meshes.append(pyrender.Mesh.from_trimesh(y_cone, smooth=True))
        
        # Z轴（蓝色）
        z_cyl = trimesh.creation.cylinder(radius=radius, height=length, sections=12)
        z_cyl.visual.vertex_colors = np.array([0, 0, 255, 255], dtype=np.uint8)
        z_cyl.apply_translation([0, 0, length/2])
        meshes.append(pyrender.Mesh.from_trimesh(z_cyl, smooth=True))
        
        z_cone = trimesh.creation.cone(radius=radius*2, height=0.02, sections=12)
        z_cone.visual.vertex_colors = np.array([0, 0, 255, 255], dtype=np.uint8)
        z_cone.apply_translation([0, 0, length + 0.01])
        meshes.append(pyrender.Mesh.from_trimesh(z_cone, smooth=True))
        
        return meshes
    
    def render_world_scene(self, current_idx):
        """渲染世界场景"""
        from viz_vb_data import (_line_mesh, _pointcloud_mesh, RenderFlags, _finger_poses_from_width, _axis_mesh)

        if self._world_scene is None:
            self._init_world_scene_cache()

        scene = self._world_scene

        for node in self._world_dynamic_nodes:
            scene.remove_node(node)
        self._world_dynamic_nodes = []

        colors = {0: [1, 0, 0], 1: [0, 1, 0]}
        for r in ROBOT_IDS:
            prefix = f'robot{r}'
            poses = self.data[prefix].get('poses', [])
            if not poses or current_idx >= len(poses):
                continue

            pts = np.array([p[:3, 3] for p in poses[:current_idx + 1]], dtype=np.float64)

            line_mesh = _line_mesh(pts, color=colors[r], radius=0.012)
            if line_mesh:
                self._world_dynamic_nodes.append(scene.add(line_mesh))

            pts_mesh = _pointcloud_mesh(pts, color=colors[r])
            if pts_mesh:
                self._world_dynamic_nodes.append(scene.add(pts_mesh))

            frame_pose = poses[current_idx]
            
            if self._show_raw_transforms.get(r, False):
                self._world_dynamic_nodes.append(scene.add(self._cached_axis_mesh, pose=frame_pose))

            if self._show_controller_mesh:
                gripper_mesh = self._cached_gripper_mesh_left if r == 0 else self._cached_gripper_mesh_right
                self._world_dynamic_nodes.append(scene.add(gripper_mesh, pose=frame_pose))

            gripper = self.data[prefix].get('gripper', [])
            if gripper and current_idx < len(gripper):
                # Get calibration for this specific robot
                calib = self.finger_calibrations.get(r) if getattr(self, "finger_calibrations", None) else None
                if calib:
                    center_pose, left_pose, right_pose = _finger_poses_from_width(
                        float(gripper[current_idx]),
                        calib,
                    )
                else:
                    grip_width = float(gripper[current_idx])
                    offset = grip_width / 2.0
                    center_pose = np.eye(4)
                    center_pose[:3, :3] = Rotation.from_euler('x', 90, degrees=True).as_matrix()
                    center_pose[:3, 3] = [0.05, 0.0, -0.04]
                    left_pose = np.eye(4)
                    right_pose = np.eye(4)
                    left_pose[:3, :3] = center_pose[:3, :3]
                    right_pose[:3, :3] = center_pose[:3, :3]
                    left_pose[:3, 3] = [0.05, - (offset - 0.01), -0.04]
                    right_pose[:3, 3] = [0.05, (offset - 0.01), -0.04]

                if self._show_claw_finger_mesh:
                    # Use separate left/right finger meshes if available
                    if self._cached_left_finger_mesh is not None and self._cached_right_finger_mesh is not None:
                        self._world_dynamic_nodes.append(scene.add(self._cached_left_finger_mesh, pose=frame_pose @ left_pose))
                        self._world_dynamic_nodes.append(scene.add(self._cached_right_finger_mesh, pose=frame_pose @ right_pose))
                    elif self._cached_finger_mesh is not None:
                        # Fallback to legacy single mesh
                        self._world_dynamic_nodes.append(scene.add(self._cached_finger_mesh, pose=frame_pose @ left_pose))
                        self._world_dynamic_nodes.append(scene.add(self._cached_finger_mesh, pose=frame_pose @ right_pose))

                if self._show_finger_axes:
                    axis_mesh = self._cached_axis_mesh or _axis_mesh(size=0.05)
                    self._world_dynamic_nodes.append(scene.add(axis_mesh, pose=frame_pose @ center_pose))
                    self._world_dynamic_nodes.append(scene.add(axis_mesh, pose=frame_pose @ left_pose))
                    self._world_dynamic_nodes.append(scene.add(axis_mesh, pose=frame_pose @ right_pose))
            # else:
            #     self._world_dynamic_nodes.append(scene.add(self._cached_gripper_mesh_right, pose=frame_pose))
            
            if self._show_controller_mesh:
                ctrl_mesh = self._cached_controller_meshes.get(r)
                if ctrl_mesh:
                    ctrl_tf = np.eye(4)
                    ctrl_tf[:3, :3] = Rotation.from_euler('y', 90, degrees=True).as_matrix()
                    ctrl_tf[:3, 3] = [0, 0, 0.03]
                    self._world_dynamic_nodes.append(scene.add(ctrl_mesh, pose=frame_pose @ ctrl_tf))

        color_img, _ = self.renderers['world'].render(scene, flags=RenderFlags.RGBA)
        return color_img[:, :, :3]
    
    def _create_trajectory_plots(self, frame_idx):
        """创建轨迹曲线图面板 - 虚线预览+实线高亮"""
        w, h = 450, 750
        panel = np.zeros((h, w, 3), dtype=np.uint8)
        panel[:] = [25, 30, 40]
        
        cv2.rectangle(panel, (0, 0), (w, 40), (15, 20, 30), -1)
        # Use Unicode-safe text rendering for Chinese
        panel = self._put_text_unicode(
            cv2.cvtColor(panel, cv2.COLOR_RGB2BGR),
            self._t('realtime_trajectory'),
            (15, 5),
            font_size=18,
            color=(220, 220, 220, 220)
        )
        panel = cv2.cvtColor(panel, cv2.COLOR_BGR2RGB)
        
        y_offset = 50
        plot_h = 110
        plot_w = w - 50
        plot_x_start = 40
        
        max_frames = len(self.data['robot0']['poses'])
        
        plots = [
            (self._t('robot_0_position'), 'position', 0, ['robot0'], ['X', 'Y', 'Z'], [(255, 10, 10), (10, 255, 10), (10, 10, 255)]),
            (self._t('robot_1_position'), 'position', 1, ['robot1'], ['X', 'Y', 'Z'], [(255, 10, 10), (10, 255, 10), (10, 10, 255)]),
            (self._t('gripper_width'), 'gripper', None, ['robot0', 'robot1'], ['R0', 'R1'], [(255, 100, 100), (100, 255, 100)])
        ]
        
        for plot_name, plot_type, robot_id, robots, labels, colors in plots:
            # Use Unicode-safe text rendering for plot titles
            panel = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
            panel = self._put_text_unicode(
                panel,
                plot_name,
                (15, y_offset - 5),
                font_size=16,
                color=(200, 200, 200)
            )
            panel = cv2.cvtColor(panel, cv2.COLOR_BGR2RGB)
            y_offset += 20
            
            cv2.rectangle(panel, (plot_x_start, y_offset), (plot_x_start + plot_w, y_offset + plot_h - 20), 
                         (35, 40, 50), -1)
            cv2.rectangle(panel, (plot_x_start, y_offset), (plot_x_start + plot_w, y_offset + plot_h - 20), 
                         (60, 65, 75), 1)
            
            if plot_type == 'position':
                prefix = f'robot{robot_id}'
                poses = self.data[prefix].get('poses', [])
                
                if poses and len(poses) > 0:
                    all_positions = np.array([poses[i][:3, 3] for i in range(len(poses))])
                    
                    for axis_idx, (axis_name, color) in enumerate(zip(labels, colors)):
                        if len(all_positions) > 1:
                            data = all_positions[:, axis_idx]
                            data_min, data_max = data.min(), data.max()
                            data_range = data_max - data_min if data_max > data_min else 1.0
                            
                            # 虚线（整条轨迹）
                            for i in range(1, len(data)):
                                x1 = int(plot_x_start + (i - 1) / max_frames * plot_w)
                                y1 = int(y_offset + (plot_h - 20) - ((data[i-1] - data_min) / data_range) * (plot_h - 30))
                                x2 = int(plot_x_start + i / max_frames * plot_w)
                                y2 = int(y_offset + (plot_h - 20) - ((data[i] - data_min) / data_range) * (plot_h - 30))
                                dark_color = tuple(int(c * 0.3) for c in color)
                                cv2.line(panel, (x1, y1), (x2, y2), dark_color, 1, cv2.LINE_AA)
                            
                            # 实线（当前进度）
                            for i in range(1, min(frame_idx + 1, len(data))):
                                x1 = int(plot_x_start + (i - 1) / max_frames * plot_w)
                                y1 = int(y_offset + (plot_h - 20) - ((data[i-1] - data_min) / data_range) * (plot_h - 30))
                                x2 = int(plot_x_start + i / max_frames * plot_w)
                                y2 = int(y_offset + (plot_h - 20) - ((data[i] - data_min) / data_range) * (plot_h - 30))
                                cv2.line(panel, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
                            
                            # Y轴标签（按颜色区分，位置错开）
                            label_x = 5
                            label_y_up = y_offset + 10 + axis_idx * 12
                            label_y_down = y_offset + plot_h - 25 - axis_idx * 12
                            cv2.putText(panel, f"{data_max:.3f}", (label_x, label_y_up), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)
                            cv2.putText(panel, f"{data_min:.3f}", (label_x, label_y_down), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)
                    
                    # X轴标签
                    cv2.putText(panel, "0", (plot_x_start, y_offset + plot_h - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1, cv2.LINE_AA)
                    cv2.putText(panel, f"{max_frames}", (plot_x_start + plot_w - 20, y_offset + plot_h - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1, cv2.LINE_AA)
                    
                    # 图例
                    legend_x = plot_x_start + 5
                    for axis_name, color in zip(labels, colors):
                        cv2.circle(panel, (legend_x, y_offset + 10), 4, color, -1)
                        cv2.putText(panel, axis_name, (legend_x + 10, y_offset + 13), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)
                        legend_x += 40
            
            elif plot_type == 'gripper':
                all_gripper_series = []
                for robot_prefix in robots:
                    gripper_data = self.data[robot_prefix].get('gripper', [])
                    if gripper_data and len(gripper_data) > 0:
                        all_gripper_series.append(np.array([gripper_data[i] for i in range(len(gripper_data))]))

                if all_gripper_series:
                    all_grippers_concat = np.concatenate(all_gripper_series)
                    data_min, data_max = all_grippers_concat.min(), all_grippers_concat.max()
                    data_range = data_max - data_min if data_max > data_min else 0.01

                    for robot_idx, (robot_prefix, label, color) in enumerate(zip(robots, labels, colors)):
                        gripper_data = self.data[robot_prefix].get('gripper', [])
                        if gripper_data and len(gripper_data) > 0:
                            all_grippers = np.array([gripper_data[i] for i in range(len(gripper_data))])

                            if len(all_grippers) > 1:
                                # 虚线（全部）
                                for i in range(1, len(all_grippers)):
                                    x1 = int(plot_x_start + (i - 1) / max_frames * plot_w)
                                    y1 = int(y_offset + (plot_h - 20) - ((all_grippers[i-1] - data_min) / data_range) * (plot_h - 30))
                                    x2 = int(plot_x_start + i / max_frames * plot_w)
                                    y2 = int(y_offset + (plot_h - 20) - ((all_grippers[i] - data_min) / data_range) * (plot_h - 30))
                                    dark_color = tuple(int(c * 0.3) for c in color)
                                    cv2.line(panel, (x1, y1), (x2, y2), dark_color, 1, cv2.LINE_AA)

                                # 实线（当前）
                                for i in range(1, min(frame_idx + 1, len(all_grippers))):
                                    x1 = int(plot_x_start + (i - 1) / max_frames * plot_w)
                                    y1 = int(y_offset + (plot_h - 20) - ((all_grippers[i-1] - data_min) / data_range) * (plot_h - 30))
                                    x2 = int(plot_x_start + i / max_frames * plot_w)
                                    y2 = int(y_offset + (plot_h - 20) - ((all_grippers[i] - data_min) / data_range) * (plot_h - 30))
                                    cv2.line(panel, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

                    # Y轴标签（统一范围）
                    cv2.putText(panel, f"{data_max:.3f}", (5, y_offset + 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.28, (150, 150, 150), 1, cv2.LINE_AA)
                    cv2.putText(panel, f"{data_min:.3f}", (5, y_offset + plot_h - 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.28, (150, 150, 150), 1, cv2.LINE_AA)
                
                # X轴标签
                cv2.putText(panel, "0", (plot_x_start, y_offset + plot_h - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1, cv2.LINE_AA)
                cv2.putText(panel, f"{max_frames}", (plot_x_start + plot_w - 20, y_offset + plot_h - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1, cv2.LINE_AA)
                
                # 图例
                legend_x = plot_x_start + 5
                for label, color in zip(labels, colors):
                    cv2.circle(panel, (legend_x, y_offset + 10), 4, color, -1)
                    cv2.putText(panel, label, (legend_x + 10, y_offset + 13), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)
                    legend_x += 50
            
            y_offset += plot_h + 15
        
        return panel
    
    def render_frame(self, frame_idx):
        """渲染完整帧"""
        self.frame_idx = frame_idx
        world_image = self.render_world_scene(frame_idx)
        return self._create_complete_layout(frame_idx, {}, world_image)
    
    def _enhance_3d_image(self, img):
        """增强3D图像效果"""
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(img, -1, kernel)
        return np.clip(cv2.addWeighted(img, 0.7, sharpened, 0.3, 0) * 1.15, 0, 255).astype(np.uint8)
    
    def _create_complete_layout(self, frame_idx, pc_images, world_image):
        """创建完整布局"""
        header = self._create_header(frame_idx)
        camera_row = self._create_camera_row(frame_idx)
        world_panel = self._add_panel_frame(world_image if world_image is not None else np.zeros((700, 900, 3), dtype=np.uint8), self._t('3d_world_view'), (100, 200, 255))
        plots_panel = self._create_trajectory_plots(frame_idx)
        
        h = max(world_panel.shape[0], plots_panel.shape[0])
        sep = np.zeros((h, 5, 3), dtype=np.uint8)
        sep[:] = [15, 20, 30]
        bottom_row = np.hstack([self._pad_height(world_panel, h), sep, self._pad_height(plots_panel, h)])
        control_bar = self._create_control_bar(frame_idx)
        max_w = max(header.shape[1], camera_row.shape[1], bottom_row.shape[1], control_bar.shape[1])
        sep_h = np.zeros((3, max_w, 3), dtype=np.uint8)
        sep_h[:] = [15, 20, 30]
        return np.vstack([self._pad_width(header, max_w), sep_h, self._pad_width(camera_row, max_w), sep_h, self._pad_width(bottom_row, max_w), sep_h, self._pad_width(control_bar, max_w)])
    
    def _create_header(self, frame_idx):
        """创建顶部标题栏"""
        w, h = 1600, 60
        header = np.zeros((h, w, 3), dtype=np.uint8)
        header[:] = [20, 25, 35]
        
        # Title with language support using Unicode-safe rendering
        title = self._t('window_title')
        header = self._put_text_unicode(
            cv2.cvtColor(header, cv2.COLOR_RGB2BGR),
            title,
            (20, 8),
            font_size=24,
            color=(255, 255, 255, 255)
        )
        header = cv2.cvtColor(header, cv2.COLOR_BGR2RGB)
        
        cv2.circle(header, (280, 30), 6, (100, 255, 100), -1)
        # Use Unicode-safe text rendering for Speed label
        header = cv2.cvtColor(header, cv2.COLOR_RGB2BGR)
        header = self._put_text_unicode(
            header,
            f"{self._t('speed')}: {self.playback_speed}x",
            (300, 20),
            font_size=18,
            color=(150, 150, 150)
        )
        header = cv2.cvtColor(header, cv2.COLOR_BGR2RGB)
        
        ep_id, max_frames = self.episodes[self.ep_idx], len(self.data['robot0']['poses'])
        # Use Unicode-safe text rendering for Episode/Frame info
        header = cv2.cvtColor(header, cv2.COLOR_RGB2BGR)
        header = self._put_text_unicode(
            header,
            f"{self._t('ep')} {ep_id}/{len(self.episodes)} | {self._t('frame')} {frame_idx}/{max_frames-1}",
            (w - 400, 20),
            font_size=18,
            color=(200, 200, 200)
        )
        header = cv2.cvtColor(header, cv2.COLOR_BGR2RGB)
        return header
    
    def _resize_with_label_unicode(self, img, label, height, color=(255, 255, 255)):
        """Resize image and add label with Unicode support"""
        if img is None:
            return None
        h, w = img.shape[:2]
        resized = cv2.resize(img, (int(w * height / h), height))
        
        # Check if label contains non-ASCII characters (like Chinese)
        has_unicode = any(ord(char) > 127 for char in label)
        
        if has_unicode:
            # Use PIL for Unicode text with outline effect
            # Draw white outline (slightly offset in multiple directions for better outline)
            for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1), (-2, 0), (2, 0), (0, -2), (0, 2)]:
                resized = self._put_text_unicode(
                    cv2.cvtColor(resized, cv2.COLOR_RGB2BGR),
                    label,
                    (10 + dx, 7 + dy),
                    font_size=18,
                    color=(255, 255, 255)
                )
                resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            # Draw colored text on top
            resized = self._put_text_unicode(
                cv2.cvtColor(resized, cv2.COLOR_RGB2BGR),
                label,
                (10, 7),
                font_size=18,
                color=color
            )
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        else:
            # Use original cv2.putText for ASCII text (English) - better rendering
            cv2.putText(resized, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(resized, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        
        return resized
    
    def _create_camera_row(self, frame_idx):
        """创建相机视图行"""
        height, all_images = 250, []
        for r in ROBOT_IDS:
            prefix, color = f'robot{r}', ((100, 100, 255) if r == 0 else (100, 255, 100))
            robot_images = [self._resize_with_label_unicode(self.data[prefix][s][frame_idx].copy(), f"R{r} {l}", height, color) 
                           for s, l in [('visual', self._t('visual')), ('left_tactile', self._t('left_tactile')), ('right_tactile', self._t('right_tactile'))]
                           if s in self.data[prefix] and frame_idx < len(self.data[prefix][s])]
            if robot_images:
                all_images.append(self._add_robot_label(hstack_with_sep(robot_images, 2, 40), f"{self._t('robot')} {r}", color))
        return hstack_with_sep(all_images, 5, 15) if all_images else np.zeros((height, 1600, 3), dtype=np.uint8)
    
    def _add_robot_label(self, img, label, color):
        """添加机器人标签"""
        header_h = 35
        panel = np.zeros((img.shape[0] + header_h, img.shape[1], 3), dtype=np.uint8)
        panel[:] = [25, 30, 40]
        cv2.rectangle(panel, (0, 0), (panel.shape[1], header_h), (15, 20, 30), -1)
        cv2.circle(panel, (15, header_h//2), 5, color, -1)
        # Use Unicode-safe text rendering for robot label
        panel = self._put_text_unicode(
            cv2.cvtColor(panel, cv2.COLOR_RGB2BGR),
            label,
            (30, 5),
            font_size=18,
            color=(220, 220, 220)
        )
        panel = cv2.cvtColor(panel, cv2.COLOR_BGR2RGB)
        panel[header_h:, :] = img
        return panel
    
    def _create_control_bar(self, frame_idx):
        """创建控制栏"""
        w, h = 1600, 80
        bar = np.zeros((h, w, 3), dtype=np.uint8)
        bar[:] = [20, 25, 35]
        max_frames = len(self.data['robot0']['poses'])
        bar_x1, bar_x2, bar_y = 200, w - 450, 30
        
        cv2.rectangle(bar, (bar_x1, bar_y-4), (bar_x2, bar_y+4), (50, 55, 65), -1)
        if max_frames > 1:
            progress = int((frame_idx / (max_frames-1)) * (bar_x2 - bar_x1))
            cv2.rectangle(bar, (bar_x1, bar_y-4), (bar_x1+progress, bar_y+4), (59, 130, 246), -1)
            cv2.circle(bar, (bar_x1+progress, bar_y), 9, (59, 130, 246), -1)
            cv2.circle(bar, (bar_x1+progress, bar_y), 10, (255, 255, 255), 2)
        
        cv2.putText(bar, f"{frame_idx}/{max_frames-1}", (bar_x1, bar_y-15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
        
        # 倍速按钮
        speed_x = bar_x2 + 20
        # Use Unicode-safe text rendering for Speed label
        bar = cv2.cvtColor(bar, cv2.COLOR_RGB2BGR)
        bar = self._put_text_unicode(
            bar,
            self._t('speed') + ":",
            (speed_x, 8),
            font_size=16,
            color=(150, 150, 150)
        )
        bar = cv2.cvtColor(bar, cv2.COLOR_BGR2RGB)
        
        speeds = [0.25, 0.5, 1.0, 2.0, 5.0]
        btn_w, btn_h = 50, 25
        btn_y = 35
        
        for i, speed in enumerate(speeds):
            btn_x = speed_x + i * (btn_w + 5)
            
            if abs(self.playback_speed - speed) < 0.01:
                btn_color = (59, 130, 246)
            else:
                btn_color = (60, 65, 75)
            
            cv2.rectangle(bar, (btn_x, btn_y), (btn_x + btn_w, btn_y + btn_h), 
                         btn_color, -1)
            cv2.rectangle(bar, (btn_x, btn_y), (btn_x + btn_w, btn_y + btn_h), 
                         (80, 85, 95), 1)
            
            label = f"{speed}x" if speed >= 1 else f"{speed}"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0]
            text_x = btn_x + (btn_w - text_size[0]) // 2
            cv2.putText(bar, label, (text_x, btn_y + 17), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1, cv2.LINE_AA)
        
        controls = "[A/D]Frame [W/S]Ep [P]Play [1-5]Speed [0]Reset [T]Move [./]Z [L]Lang [U/I]Raw [M]Grip [F]Fing [G]Axes [Q]Quit"
        cv2.putText(bar, controls, (20, 68), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1, cv2.LINE_AA)
        
        return bar
    
    def _add_panel_frame(self, img, title, color):
        """添加面板框架"""
        border, header_h = 5, 40
        panel = np.zeros((img.shape[0]+header_h+border*2, img.shape[1]+border*2, 3), dtype=np.uint8)
        panel[:] = [25, 30, 40]
        cv2.rectangle(panel, (0, 0), (panel.shape[1], header_h), (15, 20, 30), -1)
        cv2.circle(panel, (15, header_h//2), 5, color, -1)
        
        # Use Unicode-safe text rendering for panel title
        panel = self._put_text_unicode(
            cv2.cvtColor(panel, cv2.COLOR_RGB2BGR),
            title,
            (30, 5),
            font_size=18,
            color=(220, 220, 220, 220)
        )
        panel = cv2.cvtColor(panel, cv2.COLOR_BGR2RGB)
        
        panel[header_h+border:header_h+border+img.shape[0], border:border+img.shape[1]] = img
        return panel
    
    def _pad_width(self, img, target_w):
        """填充宽度"""
        if img.shape[1] >= target_w: return img
        pad = np.zeros((img.shape[0], target_w - img.shape[1], 3), dtype=np.uint8)
        pad[:] = [20, 25, 35]
        return np.hstack([img, pad])
    
    def _pad_height(self, img, target_h):
        """填充高度"""
        if img.shape[0] >= target_h: return img
        pad = np.zeros((target_h - img.shape[0], img.shape[1], 3), dtype=np.uint8)
        pad[:] = [25, 30, 40]
        return np.vstack([img, pad])
    
    def run(self):
        """运行可视化循环"""
        auto_play = False
        frame_counter = 0
        
        # Display startup info
        self._display_startup_info()
        
        while True:
            frame = self.render_frame(self.frame_idx)
            if frame is not None:
                if auto_play:
                    cv2.circle(frame, (frame.shape[1] - 50, 35), 8, (100, 255, 100), -1)
                cv2.imshow(self.window_name, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
            # 自动播放逻辑
            if auto_play:
                frame_counter += self.playback_speed
                if frame_counter >= 1.0:
                    steps = int(frame_counter)
                    frame_counter -= steps
                    
                    max_f = len(self.data['robot0']['poses'])
                    
                    if self.frame_idx + steps < max_f:
                        # 正常前进
                        self.frame_idx += steps
                    else:
                        # 到达当前Episode末尾
                        if self.auto_next_episode and self.ep_idx < len(self.episodes) - 1:
                            # 自动跳转下一个Episode
                            self._safe_print(self._t('ep_complete').format(self.episodes[self.ep_idx]))
                            self.ep_idx += 1
                            self.load_episode()
                            frame_counter = 0
                        else:
                            # 最后一个Episode或关闭自动跳转
                            self.frame_idx = max_f - 1
                            auto_play = False
                            self._safe_print(self._t('play_complete'))
            
            key = cv2.waitKey(30) & 0xFF
            
            if key == ord('d'):
                # D键：下一帧，如果到末尾自动下一个Episode
                max_f = len(self.data['robot0']['poses'])
                if self.frame_idx < max_f - 1:
                    self.frame_idx += 1
                elif self.ep_idx < len(self.episodes) - 1:
                    self.ep_idx += 1
                    self.load_episode()
                    self._safe_print(self._t('jump_to_ep').format(self.episodes[self.ep_idx]))
                    
            elif key == ord('a'):
                # A键：上一帧
                if self.frame_idx > 0:
                    self.frame_idx -= 1
                    
            elif key == ord('w'):
                # W键：下一个Episode
                if self.ep_idx < len(self.episodes) - 1:
                    self.ep_idx += 1
                    self.load_episode()
                    self._safe_print(self._t('jump_to_ep').format(self.episodes[self.ep_idx]))
                else:
                    self._safe_print(self._t('last_episode'))
                    
            elif key == ord('s'):
                # S键：上一个Episode
                if self.ep_idx > 0:
                    self.ep_idx -= 1
                    self.load_episode()
                    self._safe_print(self._t('jump_to_ep').format(self.episodes[self.ep_idx]))
                else:
                    self._safe_print(self._t('first_episode'))
                    
            elif key == ord('p'): 
                auto_play = not auto_play
                frame_counter = 0
                self._safe_print(self._t('auto_play_on') if auto_play else self._t('auto_play_off'))
                
            elif key == ord('1'): self.playback_speed = 0.25; self._safe_print(f"{self._t('speed')}: 0.25x")
            elif key == ord('2'): self.playback_speed = 0.5; self._safe_print(f"{self._t('speed')}: 0.5x")
            elif key == ord('3'): self.playback_speed = 1.0; self._safe_print(f"{self._t('speed')}: 1x")
            elif key == ord('4'): self.playback_speed = 2.0; self._safe_print(f"{self._t('speed')}: 2x")
            elif key == ord('5'): self.playback_speed = 5.0; self._safe_print(f"{self._t('speed')}: 5x")
            elif key in (ord('0'), ord(')')):
                self._reset_camera_controls()
                self._safe_print(self._t('cam_reset_msg'))
            elif key in (self.KEY_LEFT, self.KEY_UP, self.KEY_RIGHT, self.KEY_DOWN):
                step = 0.08
                if self._camera_translation_mode:
                    if key == self.KEY_LEFT:
                        self._camera_target[0] -= step
                    elif key == self.KEY_RIGHT:
                        self._camera_target[0] += step
                    elif key == self.KEY_UP:
                        self._camera_target[1] += step
                    elif key == self.KEY_DOWN:
                        self._camera_target[1] -= step
                else:
                    if key == self.KEY_LEFT:
                        self._camera_yaw -= step
                    elif key == self.KEY_RIGHT:
                        self._camera_yaw += step
                    elif key == self.KEY_UP:
                        self._camera_pitch += step
                    elif key == self.KEY_DOWN:
                        self._camera_pitch -= step
                self._refresh_world_camera()
            elif key in (ord('+'), ord('=')):
                self._camera_distance = max(self._camera_distance * 0.9, 0.05)
                self._refresh_world_camera()
            elif key in (ord('-'), ord('_')):
                self._camera_distance = self._camera_distance * 1.1
                self._refresh_world_camera()
            elif key == ord('.'):
                if self._camera_translation_mode:
                    self._camera_target[2] += 0.08
                    self._refresh_world_camera()
                    self._safe_print(self._t('cam_target_z').format(self._camera_target[2]))
            elif key == ord('/'):
                if self._camera_translation_mode:
                    self._camera_target[2] -= 0.08
                    self._refresh_world_camera()
                    self._safe_print(self._t('cam_target_z').format(self._camera_target[2]))
            elif key == ord('t'):
                self._camera_translation_mode = not self._camera_translation_mode
                self._safe_print(self._t('cam_translate_on') if self._camera_translation_mode else self._t('cam_translate_off'))
            elif key in (ord('l'), ord('L')):
                self._switch_language()
            elif key == ord('c'):
                import datetime
                filename = f"monitor_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                self._safe_print(self._t('screenshot_saved').format(filename))
            elif key == ord('u'):
                self._show_raw_transforms[0] = not self._show_raw_transforms[0]
                msg = self._t('raw_transform_show').format(0) if self._show_raw_transforms[0] else self._t('raw_transform_hide').format(0)
                self._safe_print(msg)
            elif key == ord('i'):
                self._show_raw_transforms[1] = not self._show_raw_transforms[1]
                msg = self._t('raw_transform_show').format(1) if self._show_raw_transforms[1] else self._t('raw_transform_hide').format(1)
                self._safe_print(msg)
            elif key in (ord('m'), ord('M')):
                self._show_controller_mesh = not self._show_controller_mesh
                self._safe_print(self._t('gripper_mesh_show') if self._show_controller_mesh else self._t('gripper_mesh_hide'))
            elif key in (ord('f'), ord('F')):
                self._show_claw_finger_mesh = not self._show_claw_finger_mesh
                self._safe_print(self._t('finger_mesh_show') if self._show_claw_finger_mesh else self._t('finger_mesh_hide'))
            elif key in (ord('g'), ord('G')):
                self._show_finger_axes = not self._show_finger_axes
                self._safe_print(self._t('finger_axes_show') if self._show_finger_axes else self._t('finger_axes_hide'))
            elif key == ord('q'): 
                self._safe_print(f"\n{self._t('exiting')}\n")
                break
        
        cv2.destroyAllWindows()

    def record_episode(self):
        """录制当前episode（计时）"""
        start_time = time.perf_counter()
        super().record_episode()
        elapsed = time.perf_counter() - start_time
        print(f" 录制耗时 {elapsed:.2f}s")

def main():
    import argparse
    from zarr.storage import ZipStore
    import zarr
    from replay_buffer import ReplayBuffer
    
    parser = argparse.ArgumentParser()
    parser.add_argument('zarr_path', nargs='?', default=r'C:\Users\ruich\Downloads\_0118_bi_pick_and_place2.zarr.zip')
    parser.add_argument('--record', '-r', action='store_true')
    parser.add_argument('--record_episode', '-e', type=int, default=1)
    parser.add_argument('--output_video', '-o', type=str, default=None)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--continue_after_record', "-c", action='store_true')
    args = parser.parse_args()
    
    if not os.path.exists(args.zarr_path): 
        return
    
    store = ZipStore(args.zarr_path, mode='r')
    try:
        rb = ReplayBuffer.create_from_group(zarr.open_group(store=store, mode='r'))
        print(f"加载: {rb.n_steps:,} 帧, {rb.n_episodes} episodes\n")
        Enhanced3DVisualizer(rb, np.arange(rb.n_episodes), args.record, args.record_episode, args.output_video, args.fps, args.continue_after_record)
    finally:
        store.close()

if __name__ == "__main__":
    main()
