"""
添加--export-curves命令行参数
"""

with open('src/viz_3d_enhanced.py', 'r') as f:
    content = f.read()

# 在argparse中添加参数
content = content.replace(
    "parser.add_argument('--continue_after_record'",
    "parser.add_argument('--export-curves', action='store_true', help='导出所有episode曲线图后退出')\n    parser.add_argument('--continue_after_record'"
)

# 在main函数中添加导出逻辑
content = content.replace(
    "    vis = Enhanced3DVisualizer(rb, np.arange(rb.n_episodes)",
    '''    vis = Enhanced3DVisualizer(rb, np.arange(rb.n_episodes)'''
)

# 添加导出调用
content = content.replace(
    "    Enhanced3DVisualizer(rb, np.arange(rb.n_episodes), args.record, args.record_episode, args.output_video, args.fps, args.continue_after_record)",
    '''    vis = Enhanced3DVisualizer(rb, np.arange(rb.n_episodes), args.record, args.record_episode, args.output_video, args.fps, args.continue_after_record)
    
    # 如果是导出模式
    if args.export_curves:
        vis.export_all_curves('curves_export')
        return'''
)

with open('src/viz_3d_enhanced.py', 'w') as f:
    f.write(content)

print("✓ 添加--export-curves参数")
