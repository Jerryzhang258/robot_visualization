#!/usr/bin/env python3
"""
导出所有episode的曲线图
"""

import sys
import os
import cv2
import numpy as np

# 添加src到路径
sys.path.insert(0, 'src')

# 直接使用主程序的加载方式
import zarr
from zarr.storage import ZipStore
from scipy.spatial.transform import Rotation as R

def load_episode_data(zarr_path, episode_idx):
    """加载单个episode的数据"""
    store = ZipStore(zarr_path, mode='r')
    root = zarr.open_group(store=store, mode='r')
    
    data_group = root['data']
    episodes = sorted([int(k) for k in data_group.keys()])
    
    episode_data = data_group[str(episodes[episode_idx])]
    
    # 提取数据
    result = {}
    for robot_id in [0, 1]:
        prefix = f'robot{robot_id}'
        
        # 位置数据
        pos_data = episode_data[f'{prefix}_eef_pos'][:]
        
        # 从位置+旋转提取RPY
        all_rpy = []
        for i in range(len(pos_data)):
            # 假设有旋转数据，如果没有就跳过
            try:
                rot_data = episode_data[f'{prefix}_eef_rot_axis_angle'][:]
                rot_vec = rot_data[i]
                angle = np.linalg.norm(rot_vec)
                if angle > 0:
                    axis = rot_vec / angle
                    rot = R.from_rotvec(rot_vec)
                    rpy = rot.as_euler('xyz', degrees=True)
                    all_rpy.append(rpy)
                else:
                    all_rpy.append([0, 0, 0])
            except:
                all_rpy.append([0, 0, 0])
        
        result[f'robot{robot_id}_pos'] = pos_data
        result[f'robot{robot_id}_rpy'] = np.array(all_rpy)
        
        # 夹爪数据
        try:
            result[f'robot{robot_id}_gripper'] = episode_data[f'{prefix}_gripper_width'][:]
        except:
            result[f'robot{robot_id}_gripper'] = np.zeros(len(pos_data))
    
    return result, episodes[episode_idx]

def draw_curves(data, episode_id):
    """绘制曲线图"""
    w, h = 1100, 1000
    panel = np.zeros((h, w, 3), dtype=np.uint8)
    panel[:] = [25, 30, 40]
    
    # 标题
    cv2.rectangle(panel, (0, 0), (w, 40), (15, 20, 30), -1)
    cv2.putText(panel, f"Episode {episode_id} - Trajectories", (15, 28), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
    
    y_offset = 50
    plot_h = 120
    plot_w = w - 100
    plot_x_start = 80
    
    max_frames = len(data['robot0_pos'])
    
    # 定义要绘制的图表
    plots = [
        ("Left Arm Position (m)", data['robot0_pos'], ['X', 'Y', 'Z']),
        ("Left Arm Rotation (deg)", data['robot0_rpy'], ['Roll', 'Pitch', 'Yaw']),
        ("Right Arm Position (m)", data['robot1_pos'], ['X', 'Y', 'Z']),
        ("Right Arm Rotation (deg)", data['robot1_rpy'], ['Roll', 'Pitch', 'Yaw']),
        ("Gripper Width (m)", np.column_stack([data['robot0_gripper'], data['robot1_gripper']]), ['Left', 'Right']),
    ]
    
    colors = [(255, 100, 100), (100, 255, 100), (100, 100, 255)]
    
    for plot_name, plot_data, labels in plots:
        # 标题
        cv2.putText(panel, plot_name, (15, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        y_offset += 20
        
        # 绘图区域
        cv2.rectangle(panel, (plot_x_start, y_offset), 
                     (plot_x_start + plot_w, y_offset + plot_h - 20), 
                     (35, 40, 50), -1)
        cv2.rectangle(panel, (plot_x_start, y_offset), 
                     (plot_x_start + plot_w, y_offset + plot_h - 20), 
                     (60, 65, 75), 1)
        
        # 绘制每个轴的数据
        for axis_idx in range(min(len(labels), plot_data.shape[1])):
            color = colors[axis_idx % 3]
            data = plot_data[:, axis_idx]
            
            if len(data) > 1:
                data_min, data_max = data.min(), data.max()
                data_range = data_max - data_min if data_max > data_min else 1.0
                
                # 绘制曲线
                for i in range(1, len(data)):
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
        for label, color in zip(labels, colors[:len(labels)]):
            cv2.circle(panel, (legend_x, y_offset + 10), 4, color, -1)
            cv2.putText(panel, label, (legend_x + 10, y_offset + 13), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)
            legend_x += 70
        
        y_offset += plot_h + 10
    
    return panel

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('zarr_path')
    parser.add_argument('-o', '--output', default='curves_export')
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    # 加载数据
    store = ZipStore(args.zarr_path, mode='r')
    root = zarr.open_group(store=store, mode='r')
    data_group = root['data']
    episodes = sorted([int(k) for k in data_group.keys()])
    
    print(f"总共 {len(episodes)} 个episodes")
    
    for idx, ep_id in enumerate(episodes):
        try:
            data, _ = load_episode_data(args.zarr_path, idx)
            panel = draw_curves(data, ep_id)
            
            output_path = os.path.join(args.output, f'episode_{ep_id:03d}.png')
            cv2.imwrite(output_path, panel)
            
            print(f"  [{idx+1}/{len(episodes)}] Episode {ep_id} -> {output_path}")
        except Exception as e:
            print(f"  [ERROR] Episode {ep_id}: {e}")
    
    print(f"\n✓ 完成！保存到: {args.output}/")
