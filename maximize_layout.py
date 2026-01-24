"""
最大化布局，铺满屏幕
"""

with open('src/viz_3d_enhanced.py', 'r') as f:
    content = f.read()

# 1. 进一步增加轨迹面板宽度和高度
# 当前是 w, h = 700, 900
content = content.replace('w, h = 700, 900', 'w, h = 850, 1000')
print("✓ 轨迹面板: 700->850宽, 900->1000高")

# 2. 增加3D窗口高度匹配
content = content.replace(
    'np.zeros((700, 900, 3)',
    'np.zeros((900, 900, 3)'
)
print("✓ 3D窗口默认尺寸: 900x900")

# 3. 调整图表内边距，使图表更宽
content = content.replace('plot_w = w - 80', 'plot_w = w - 60')
print("✓ 减少图表内边距")

# 4. 增加图表高度
content = content.replace('plot_h = 110', 'plot_h = 130')
print("✓ 图表高度: 110->130")

with open('src/viz_3d_enhanced.py', 'w') as f:
    f.write(content)

print("\n" + "="*60)
print("布局最大化完成！")
print("="*60)
print("\n新布局：")
print("  左侧：3D窗口 900x900")
print("  右侧：轨迹面板 850x1000")
print("  总宽度：~1755px")
print("  应该基本铺满屏幕")
print("="*60)
