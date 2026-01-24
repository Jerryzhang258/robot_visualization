#!/usr/bin/env python3
"""
随机录制5个episode
"""

import subprocess
import random
import os

# 总共169个episodes (0-168)
total_episodes = 169

# 随机选择5个
random_eps = random.sample(range(total_episodes), 5)
random_eps.sort()

print(f"随机选择的episodes: {random_eps}\n")

# 创建输出目录
os.makedirs('recorded_videos', exist_ok=True)

# 依次录制
for i, ep in enumerate(random_eps, 1):
    output = f'recorded_videos/episode_{ep:03d}.mp4'
    print(f"[{i}/5] 录制 Episode {ep} -> {output}")
    
    cmd = [
        'python', 'src/viz_3d_enhanced.py',
        'data/_0115_bi_pick_and_place_2ver.zarr.zip',
        '-r',
        '--record_episode', str(ep),
        '--output_video', output
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  ✓ 完成")
        else:
            print(f"  ✗ 失败: {result.stderr[:100]}")
    except Exception as e:
        print(f"  ✗ 错误: {e}")

print(f"\n✓ 录制完成！视频保存在: recorded_videos/")
print(f"随机episodes: {random_eps}")
