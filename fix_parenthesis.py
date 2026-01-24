with open('src/viz_3d_enhanced.py', 'r') as f:
    content = f.read()

# 修复括号
content = content.replace(
    '(int(900 * world_image.shape[1] / world_image.shape[0]), 900)',
    '(int(900 * world_image.shape[1] / world_image.shape[0]), 900))'
)

with open('src/viz_3d_enhanced.py', 'w') as f:
    f.write(content)

print("✓ 已修复括号")
