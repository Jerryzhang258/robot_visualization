with open('src/viz_3d_enhanced.py', 'r') as f:
    lines = f.readlines()

count = 0
new_lines = []
for line in lines:
    # 只替换绘制文本的cv2.putText（不是绘制图像的）
    if 'cv2.putText(' in line and ('panel' in line or 'header' in line or 'control_bar' in line):
        # 替换函数名
        modified = line.replace('cv2.putText(', 'put_text(')
        # 移除cv2.FONT_HERSHEY_SIMPLEX及其参数
        import re
        # 移除字体参数
        modified = re.sub(r', cv2\.FONT_HERSHEY_SIMPLEX, ([0-9.]+)', r', int(\1*30)', modified)
        # 移除thickness和line_type参数
        modified = re.sub(r', 1, cv2\.LINE_AA', '', modified)
        
        new_lines.append(modified)
        count += 1
    else:
        new_lines.append(line)

with open('src/viz_3d_enhanced.py', 'w') as f:
    f.writelines(new_lines)

print(f"✓ 已替换{count}处cv2.putText为put_text")
