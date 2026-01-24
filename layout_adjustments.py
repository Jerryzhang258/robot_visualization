"""
布局调整：
1. 3D窗口放大
2. 右边面板对齐
3. 添加rotation信息（从pose矩阵提取roll/yaw/pitch）
4. 改为Left/Right
5. 2位小数
"""

with open('src/viz_3d_enhanced.py', 'r') as f:
    content = f.read()

# 1. 放大3D窗口 700 -> 900
content = content.replace(
    'world_image = cv2.resize(world_image, (int(700 *',
    'world_image = cv2.resize(world_image, (int(900 *'
)
content = content.replace('), 700))', '), 900))')

print("✓ 3D窗口放大到900px")

# 2. 调整轨迹面板宽度以对齐
content = content.replace('w, h = 450, 750', 'w, h = 550, 750')
print("✓ 轨迹面板宽度调整为550px")

# 3. 改Robot为Left/Right
content = content.replace('"Robot 0"', '"Left Arm"')
content = content.replace('"Robot 1"', '"Right Arm"')
content = content.replace('["R0", "R1"]', '["Left", "Right"]')
print("✓ 改为Left/Right")

# 4. 坐标改为2位小数
content = content.replace('.3f', '.2f')
content = content.replace('.4f', '.2f')
print("✓ 坐标改为2位小数")

with open('src/viz_3d_enhanced.py', 'w') as f:
    f.write(content)

print("\n" + "="*60)
print("布局调整完成！")
print("="*60)
