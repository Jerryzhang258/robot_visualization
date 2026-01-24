"""
在主程序中添加批量导出功能
"""

with open('src/viz_3d_enhanced.py', 'r') as f:
    lines = f.readlines()

# 添加批量导出方法到类中
export_method = '''
    def export_all_curves(self, output_dir='curves_export'):
        """导出所有episode的曲线图"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        total = len(self.episodes)
        print(f"\\n开始导出 {total} 个episodes的曲线图...")
        
        for idx in range(total):
            try:
                self.ep_idx = idx
                ep_id = self.episodes[idx]
                self._load_episode(idx)
                
                # 获取最后一帧（完整轨迹）
                max_frame = len(self.data['robot0']['poses']) - 1
                
                # 只生成曲线面板
                curves = self._create_trajectory_plots(max_frame)
                
                # 保存
                output_path = os.path.join(output_dir, f'ep_{ep_id:03d}.png')
                cv2.imwrite(output_path, curves)
                
                print(f"  [{idx+1}/{total}] Episode {ep_id} -> {output_path}")
            except Exception as e:
                print(f"  [ERROR] Episode {idx}: {e}")
        
        print(f"\\n✓ 完成！所有曲线图保存在: {output_dir}/")
    
'''

# 找到run方法之前插入
for i, line in enumerate(lines):
    if 'def run(self):' in line:
        lines.insert(i, export_method)
        print(f"✓ 在第{i}行添加导出方法")
        break

with open('src/viz_3d_enhanced.py', 'w') as f:
    f.writelines(lines)

print("✓ 批量导出功能已添加")
