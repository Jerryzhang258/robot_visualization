"""
同步修改export_curves.py的Y轴标签
"""

with open('src/export_curves.py', 'r') as f:
    content = f.read()

# 同样的替换
old_y = '''                # Y轴标签
                cv2.putText(panel, f"{data_max:.2f}", (5, y_offset + 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1, cv2.LINE_AA)
                cv2.putText(panel, f"{data_min:.2f}", (5, y_offset + plot_h - 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1, cv2.LINE_AA)'''

new_y = '''                # Y轴标签（按颜色错开）
                label_x = 5
                label_y_up = y_offset + 10 + axis_idx * 12
                label_y_down = y_offset + plot_h - 25 - axis_idx * 12
                cv2.putText(panel, f"{data_max:.2f}", (label_x, label_y_up), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)
                cv2.putText(panel, f"{data_min:.2f}", (label_x, label_y_down), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)'''

content = content.replace(old_y, new_y)

with open('src/export_curves.py', 'w') as f:
    f.write(content)

print("✓ export_curves.py的Y轴标签也已修正")
