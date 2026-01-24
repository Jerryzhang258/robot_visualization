with open('src/viz_3d_enhanced.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 清理字典中的self.lang.t()调用
# 这些都是错误的递归调用
wrong_calls = [
    'self.lang.t("monitor")',
    'self.lang.t("left")',
    'self.lang.t("right")',
    'self.lang.t("world")',
    'self.lang.t("traj")',
    'self.lang.t("pos")',
    'self.lang.t("grip")',
    'self.lang.t("rot")',
    'self.lang.t("vis")',
    'self.lang.t("ep")',
    'self.lang.t("frame")',
    'self.lang.t("speed")',
]

for wrong in wrong_calls:
    # 把self.lang.t("xxx")替换回"xxx"
    content = content.replace(wrong, wrong.replace('self.lang.t(', '').replace(')', ''))

with open('src/viz_3d_enhanced.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✓ 已清理字典中的递归调用")
