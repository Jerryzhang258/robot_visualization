"""
添加Roll/Pitch/Yaw曲线显示
"""

with open('src/viz_3d_enhanced.py', 'r') as f:
    lines = f.readlines()

# 找到plots定义
for i, line in enumerate(lines):
    if 'plots = [' in line:
        # 找到Gripper那行
        for j in range(i, min(i+10, len(lines))):
            if '"Gripper Width (m)"' in lines[j]:
                # 在Gripper之前插入Rotation
                indent = '            '
                rotation_lines = [
                    indent + '("Left Arm Rotation (deg)", ["left"], ["Roll", "Pitch", "Yaw"], [(255, 150, 150), (150, 255, 150), (150, 150, 255)]),\n',
                    indent + '("Right Arm Rotation (deg)", ["right"], ["Roll", "Pitch", "Yaw"], [(255, 150, 150), (150, 255, 150), (150, 150, 255)]),\n',
                ]
                for k, rot_line in enumerate(rotation_lines):
                    lines.insert(j + k, rot_line)
                print(f"✓ 添加Rotation曲线定义")
                break
        break

# 找到Position绘制逻辑，添加Rotation处理
for i, line in enumerate(lines):
    if 'if "Position" in plot_name:' in line:
        # 找到这个if块的结束（下一个elif）
        indent_base = len(line) - len(line.lstrip())
        for j in range(i+1, min(i+100, len(lines))):
            if 'elif "Gripper"' in lines[j]:
                # 在Gripper之前插入Rotation处理
                rotation_code = '''            
            elif "Rotation" in plot_name:
                # 从pose矩阵提取RPY
                arm = "left" if "Left" in plot_name else "right"
                rid = 0 if arm == "left" else 1
                prefix = f'robot{rid}'
                poses = self.data[prefix].get('poses', [])
                
                if poses and len(poses) > 0:
                    from scipy.spatial.transform import Rotation as R
                    all_rpy = []
                    for pose in poses:
                        rot_mat = pose[:3, :3]
                        r = R.from_matrix(rot_mat)
                        rpy = r.as_euler('xyz', degrees=True)  # roll, pitch, yaw
                        all_rpy.append(rpy)
                    
                    all_rpy = np.array(all_rpy)
                    
                    for axis_idx, (axis_name, color) in enumerate(zip(labels, colors)):
                        if len(all_rpy) > 1:
                            data = all_rpy[:, axis_idx]
                            data_min, data_max = data.min(), data.max()
                            data_range = data_max - data_min if data_max > data_min else 1.0
                            
                            # 虚线（全轨迹）
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
                        legend_x += 60
            
'''
                lines.insert(j, rotation_code)
                print(f"✓ 添加Rotation绘制代码")
                break
        break

# 其他优化
for i, line in enumerate(lines):
    # 布局
    if 'w, h = 450, 750' in line:
        lines[i] = line.replace('450', '700')
    # 2位小数
    if '.3f' in line or '.4f' in line:
        lines[i] = line.replace('.3f', '.2f').replace('.4f', '.2f')
    # Left/Right
    if '"Robot 0"' in line:
        lines[i] = line.replace('"Robot 0"', '"Left Arm"')
    if '"Robot 1"' in line:
        lines[i] = line.replace('"Robot 1"', '"Right Arm"')

with open('src/viz_3d_enhanced.py', 'w') as f:
    f.writelines(lines)

print("\n✓ Rotation曲线已添加!")
