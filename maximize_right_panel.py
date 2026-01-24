"""
最大化右侧面板，消除空白
"""

with open('src/viz_3d_enhanced.py', 'r') as f:
    content = f.read()

# 1. 大幅增加轨迹面板宽度
# 当前 w, h = 850, 1000
# 改为更宽
content = content.replace('w, h = 850, 1000', 'w, h = 1100, 1000')
print("✓ 轨迹面板宽度: 850 -> 1100")

# 2. 调整图表区域比例
content = content.replace('plot_w = w - 60', 'plot_w = w - 100')
print("✓ 调整图表左边距")

# 3. 调整plot_x_start给图例留更多空间
content = content.replace('plot_x_start = 40', 'plot_x_start = 80')
print("✓ 图表起始位置右移")

with open('src/viz_3d_enhanced.py', 'w') as f:
    f.write(content)

print("\n" + "="*60)
print("右侧面板最大化完成！")
print("="*60)
print("\n新尺寸：")
print("  左侧 3D窗口: ~900px")
print("  右侧轨迹面板: 1100px")
print("  应该能铺满屏幕了")
print("="*60)
