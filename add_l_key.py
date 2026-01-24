with open('src/viz_3d_enhanced.py', 'r') as f:
    lines = f.readlines()

# 找到键盘处理的地方，在q键之前添加L键
for i, line in enumerate(lines):
    if "elif key == ord('q')" in line:
        indent = len(line) - len(line.lstrip())
        # 在q键之前插入L键处理
        insert_lines = [
            ' ' * indent + "elif key == ord('l') or key == ord('L'):\n",
            ' ' * (indent + 4) + "self.lang.toggle()\n",
            ' ' * (indent + 4) + "continue\n",
        ]
        for j, new_line in enumerate(insert_lines):
            lines.insert(i + j, new_line)
        print(f"✓ 在第{i+1}行之前添加L键处理")
        break

with open('src/viz_3d_enhanced.py', 'w') as f:
    f.writelines(lines)

print("完成！按L键可以切换语言")
