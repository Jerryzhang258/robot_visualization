"""
修复布局，消除右侧空白
调整各面板宽度使其更平衡
"""

with open('src/viz_3d_enhanced.py', 'r') as f:
    content = f.read()

# 1. 增加轨迹面板宽度 550 -> 700
content = content.replace('w, h = 550, 750', 'w, h = 700, 900')
print("✓ 轨迹面板: 550 -> 700宽, 750 -> 900高")

# 2. 调整plot的内部宽度比例
content = content.replace('plot_w = w - 50', 'plot_w = w - 80')
print("✓ 调整图表内边距")

# 3. 检查3D窗口大小
if '900 *' in content and '), 900)' in content:
    print("✓ 3D窗口已是900px")

with open('src/viz_3d_enhanced.py', 'w') as f:
    f.write(content)

print("\n" + "="*60)
print("布局优化完成！")
print("="*60)
print("\n新布局：")
print("  左侧：3D World View (900px宽)")
print("  右侧：Trajectories (700px宽)")
print("  总宽度：~1600px，更加平衡")
print("="*60)
