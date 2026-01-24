with open('src/viz_3d_enhanced.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 修复：字典值应该直接是字符串，不应该调用self.t()
# 找到并替换错误的字典定义
wrong_patterns = [
    'self.lang.t("monitor")',
    'self.lang.t("world")',
    'self.lang.t("traj")',
]

# 如果发现这些错误模式，说明字典定义有问题
# 需要完全重新定义字典部分
if any(pattern in content for pattern in wrong_patterns):
    # 直接替换整个字典定义
    old_dict = '''            "EN": {
                "monitor": self.lang.t("monitor"),'''
    
    new_dict = '''            "EN": {
                "monitor": "Robot Monitor",'''
    
    content = content.replace(old_dict, new_dict)

# 清理其他可能的self.lang.t调用
content = content.replace('self.lang.t("', '"')
content = content.replace('")', '"')

with open('src/viz_3d_enhanced.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✓ 已修复字典定义")
