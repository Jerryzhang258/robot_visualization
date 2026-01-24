"""
修改Y轴标签：按颜色错开显示，避免重叠
"""

with open('src/viz_3d_enhanced.py', 'r') as f:
    content = f.read()

# 替换Y轴标签的绘制方式
# 旧的方式（所有轴共用一个位置）
old_y_labels = '''                            # Y轴标签
                            cv2.putText(panel, f"{data_max:.2f}", (5, y_offset + 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1, cv2.LINE_AA)
                            cv2.putText(panel, f"{data_min:.2f}", (5, y_offset + plot_h - 25), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1, cv2.LINE_AA)'''

# 新的方式（按颜色错开）
new_y_labels = '''                            # Y轴标签（按颜色区分，位置错开）
                            label_x = 5
                            label_y_up = y_offset + 10 + axis_idx * 12
                            label_y_down = y_offset + plot_h - 25 - axis_idx * 12
                            cv2.putText(panel, f"{data_max:.2f}", (label_x, label_y_up), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)
                            cv2.putText(panel, f"{data_min:.2f}", (label_x, label_y_down), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)'''

content = content.replace(old_y_labels, new_y_labels)

with open('src/viz_3d_enhanced.py', 'w') as f:
    f.write(content)

print("✓ Y轴标签已改为错开显示")
print("  - 每个轴用对应颜色")
print("  - 上下错开12像素")
print("  - 不会重叠")
