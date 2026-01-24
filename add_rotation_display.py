"""
在轨迹图中添加rotation (roll/yaw/pitch) 显示
从pose矩阵中提取欧拉角
"""

import re

with open('src/viz_3d_enhanced.py', 'r') as f:
    lines = f.readlines()

# 1. 首先添加scipy的Rotation import（如果还没有）
for i, line in enumerate(lines):
    if 'from scipy.spatial.transform import Rotation' not in ''.join(lines[:20]):
        if 'import scipy' in line or 'from scipy' in line:
            lines.insert(i+1, 'from scipy.spatial.transform import Rotation\n')
            print("✓ 添加Rotation import")
            break

# 2. 找到plots定义，添加rotation plots
for i, line in enumerate(lines):
    if 'plots = [' in line and 'Position (m)' in lines[i+1]:
        # 找到plots列表的结束]
        for j in range(i+1, min(i+20, len(lines))):
            if '("Gripper Width' in lines[j]:
                # 在Gripper之前插入rotation plots
                indent = '            '
                rotation_lines = [
                    indent + '("Left Arm Rotation (deg)", ["left"], ["Roll", "Yaw", "Pitch"], [(255, 150, 150), (150, 255, 150), (150, 150, 255)]),\n',
                    indent + '("Right Arm Rotation (deg)", ["right"], ["Roll", "Yaw", "Pitch"], [(255, 150, 150), (150, 255, 150), (150, 150, 255)]),\n',
                ]
                for k, rot_line in enumerate(rotation_lines):
                    lines.insert(j + k, rot_line)
                print(f"✓ 在第{j+1}行添加rotation plots定义")
                break
        break

# 3. 在绘图逻辑中添加rotation处理
# 找到 if "Position" in plot_name: 的位置
for i, line in enumerate(lines):
    if 'if "Position" in plot_name:' in line:
        # 在这个if块之后添加rotation处理
        # 先找到这个if块的结束位置
        indent_level = len(line) - len(line.lstrip())
        block_end = i
        for j in range(i+1, min(i+100, len(lines))):
            current_indent = len(lines[j]) - len(lines[j].lstrip())
            if current_indent <= indent_level and 'elif' in lines[j]:
                block_end = j
                break
        
        # 在elif之前插入rotation处理代码
        rotation_code = '''            
            elif "Rotation" in plot_name:
                # 从pose矩阵提取roll/yaw/pitch
                arm_name = "left" if "Left" in plot_name else "right"
                robot_id = 0 if arm_name == "left" else 1
                prefix = f'robot{robot_id}'
                poses = self.data[prefix].get('poses', [])
                
                if poses and len(poses) > 0:
                    # 从旋转矩阵提取欧拉角
                    all_rotations = []
                    for pose in poses:
                        rot_matrix = pose[:3, :3]
                        r = Rotation.from_matrix(rot_matrix)
                        euler = r.as_euler('xyz', degrees=True)  # roll, pitch, yaw
                        all_rotations.append([euler[0], euler[2], euler[1]])  # roll, yaw, pitch
                    
                    all_rotations = np.array(all_rotations)
                    
                    for axis_idx, (axis_name, color) in enumerate(zip(labels, colors)):
                        if len(all_rotations) > 1:
                            data = all_rotations[:, axis_idx]
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
                            
                            # Y轴标签
                            cv2.putText(panel, f"{data_max:.2f}", (5, y_offset + 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1, cv2.LINE_AA)
                            cv2.putText(panel, f"{data_min:.2f}", (5, y_offset + plot_h - 25), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1, cv2.LINE_AA)
                    
                    # 图例
                    legend_x = plot_x_start + 5
                    for axis_name, color in zip(labels, colors):
                        cv2.circle(panel, (legend_x, y_offset + 10), 4, color, -1)
                        cv2.putText(panel, axis_name, (legend_x + 10, y_offset + 13), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)
                        legend_x += 55
            
'''
        
        lines.insert(block_end, rotation_code)
        print(f"✓ 在第{block_end+1}行添加rotation绘制代码")
        break

with open('src/viz_3d_enhanced.py', 'w') as f:
    f.writelines(lines)

print("\n" + "="*60)
print("✓ Rotation显示已添加！")
print("="*60)
print("\n现在右边轨迹面板会显示：")
print("  - Left Arm Position (X, Y, Z)")
print("  - Left Arm Rotation (Roll, Yaw, Pitch)")
print("  - Right Arm Position (X, Y, Z)")
print("  - Right Arm Rotation (Roll, Yaw, Pitch)")
print("  - Gripper Width (Left, Right)")
print("="*60)
