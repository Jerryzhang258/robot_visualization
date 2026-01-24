with open('src/viz_3d_enhanced.py', 'r') as f:
    lines = f.readlines()

# 找到if args.export_curves并移到正确位置
for i, line in enumerate(lines):
    if 'if args.export_curves:' in line and i > 750:
        # 检查这几行
        # 应该把这3行移到vis创建之后，try块内
        export_lines = [lines[i], lines[i+1], lines[i+2]]  # if, export, return
        
        # 删除这3行
        del lines[i:i+3]
        
        # 找到vis创建的那行
        for j in range(max(0, i-10), i):
            if 'vis = Enhanced3DVisualizer(rb' in lines[j]:
                # 在这行后面插入（try块内）
                indent = '        '
                lines.insert(j+1, '\n')
                lines.insert(j+2, indent + '# 导出模式\n')
                lines.insert(j+3, indent + 'if hasattr(args, "export_curves") and args.export_curves:\n')
                lines.insert(j+4, indent + '    self.export_all_curves("curves_export")\n')
                lines.insert(j+5, indent + '    store.close()\n')
                lines.insert(j+6, indent + '    return\n')
                print(f"✓ 将导出逻辑移到try块内（第{j+2}行）")
                break
        break

with open('src/viz_3d_enhanced.py', 'w') as f:
    f.writelines(lines)
