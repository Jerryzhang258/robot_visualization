"""
确保：
1. 所有坐标都是2位小数
2. 布局铺满全屏
"""

with open('src/viz_3d_enhanced.py', 'r') as f:
    content = f.read()

# 1. 强制所有小数格式为.2f
content = content.replace('.3f', '.2f')
content = content.replace('.4f', '.2f')
print("✓ 所有小数格式改为.2f")

# 2. 最大化布局
# 轨迹面板
if 'w, h = 700, 1000' in content:
    content = content.replace('w, h = 700, 1000', 'w, h = 1100, 1000')
elif 'w, h = 450, 750' in content:
    content = content.replace('w, h = 450, 750', 'w, h = 1100, 1000')
else:
    # 查找当前值
    import re
    match = re.search(r'w, h = (\d+), (\d+)', content)
    if match:
        current_w = match.group(1)
        content = content.replace(f'w, h = {current_w},', 'w, h = 1100,')

print("✓ 轨迹面板宽度: 1100px")

# 3D窗口大小
content = content.replace('(int(700 *', '(int(1000 *')
content = content.replace('), 700)', '), 1000)')
print("✓ 3D窗口高度: 1000px")

# 调整相机行高度以匹配
content = content.replace('height, all_images = 250, []', 'height, all_images = 280, []')
print("✓ 相机行高度: 280px")

with open('src/viz_3d_enhanced.py', 'w') as f:
    f.write(content)

print("\n" + "="*60)
print("✓ 完成！")
print("="*60)
print("布局：")
print("  - 3D窗口: 1000px高")
print("  - 轨迹面板: 1100px宽 x 1000px高")
print("  - 相机行: 280px高")
print("  - 总宽度: ~2100px，应该铺满全屏")
print("  - 所有坐标: 2位小数")
print("="*60)
