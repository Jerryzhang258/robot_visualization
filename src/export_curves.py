#!/usr/bin/env python3
"""
批量导出所有episode的曲线图
"""

import sys
import os
import cv2
import numpy as np
import argparse

from viz_vb_data import ReplayBuffer
import zarr
from zarr.storage import ZipStore
from scipy.spatial.transform import Rotation as R

def draw_trajectory_panel(rb, episode_idx):
    """绘制单个episode的轨迹面板"""
    w, h = 1100, 1000
    panel = np.zeros((h, w, 3), dtype=np.uint8)
    panel[:] = [25, 30, 40]
    
    # 加载episode数据
    data = rb.get_episode(episode_idx)
    
    # 标题
    cv2.rectangle(panel, (0, 0), (w, 40), (15, 20, 30), -1)
    cv2.putText(panel, f"Episode {episode_idx} - Trajectories", (15, 28), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
    
    y_offset = 50
    plot_h = 130
    plot_w = w - 100
    plot_x_start = 80
    
    # 提取数据
    robot0_pos = data['robot0_eef_pos'][:]
    robot1_pos = data['robot1_eef_pos'][:]
    robot0_grip = data['robot0_gripper_width'][:]
    robot1_grip = data['robot1_gripper_width'][:]
    
    # 从旋转轴角提取RPY
    def extract_rpy(rot_axis_angle_data):
        rpy_list = []
        for rot_vec in rot_axis_angle_data:
            angle = np.linalg.norm(rot_vec)
            if angle > 0:
                r = R.from_rotvec(rot_vec)
                rpy = r.as_euler('xyz', degrees=True)
                rpy_list.append(rpy)
            else:
                rpy_list.append([0, 0, 0])
        return np.array(rpy_list)
    
    try:
        robot0_rpy = extract_rpy(data['robot0_eef_rot_axis_angle'][:])
        robot1_rpy = extract_rpy(data['robot1_eef_rot_axis_angle'][:])
    except:
        robot0_rpy = np.zeros((len(robot0_pos), 3))
        robot1_rpy = np.zeros((len(robot1_pos), 3))
    
    max_frames = len(robot0_pos)
    
    # 定义图表
    plots = [
        ("Left Arm Position (m)", robot0_pos, ['X', 'Y', 'Z']),
        ("Left Arm Rotation (deg)", robot0_rpy, ['Roll', 'Pitch', 'Yaw']),
        ("Right Arm Position (m)", robot1_pos, ['X', 'Y', 'Z']),
        ("Right Arm Rotation (deg)", robot1_rpy, ['Roll', 'Pitch', 'Yaw']),
        ("Gripper Width (m)", np.column_stack([robot0_grip, robot1_grip]), ['Left', 'Right']),
    ]
    
    colors = [(255, 100, 100), (100, 255, 100), (100, 100, 255)]
    
    for plot_name, plot_data, labels in plots:
        # 绘制标题
        cv2.putText(panel, plot_name, (15, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        y_offset += 20
        
        # 绘图区域背景
        cv2.rectangle(panel, (plot_x_start, y_offset), 
                     (plot_x_start + plot_w, y_offset + plot_h - 20), 
                     (35, 40, 50), -1)
        cv2.rectangle(panel, (plot_x_start, y_offset), 
                     (plot_x_start + plot_w, y_offset + plot_h - 20), 
                     (60, 65, 75), 1)
        
        # 绘制每条曲线
        for axis_idx in range(min(len(labels), plot_data.shape[1])):
            color = colors[axis_idx]
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

def main():
    parser = argparse.ArgumentParser(description='批量导出episode曲线图')
    parser.add_argument('zarr_path', help='zarr数据文件路径')
    parser.add_argument('-o', '--output', default='curves_export', help='输出目录')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    print(f"加载数据: {args.zarr_path}")
    
    # 加载数据
    store = ZipStore(args.zarr_path, mode='r')
    try:
        rb = ReplayBuffer.create_from_group(zarr.open_group(store=store, mode='r'))
        total = rb.n_episodes
        
        print(f"总共 {total} 个episodes\n")
        
        for ep_idx in range(total):
            try:
                # 绘制曲线
                panel = draw_trajectory_panel(rb, ep_idx)
                
                # 保存
                output_path = os.path.join(args.output, f'episode_{ep_idx:03d}.png')
                cv2.imwrite(output_path, panel)
                
                print(f"  [{ep_idx+1:3d}/{total}] Episode {ep_idx:3d} -> {output_path}")
                
            except Exception as e:
                print(f"  [ERROR] Episode {ep_idx:3d}: {e}")
                continue
        
        print(f"\n{'='*60}")
        print(f"✓ 完成！")
        print(f"{'='*60}")
        print(f"保存位置: {args.output}/")
        print(f"共 {total} 张图片")
        print(f"\n快速查看：")
        print(f"  open {args.output}/")
        print(f"{'='*60}")
        
    finally:
        store.close()

if __name__ == '__main__':
    main()
