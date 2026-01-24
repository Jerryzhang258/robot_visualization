# -*- coding: utf-8 -*-
"""
一键完整修复：
1. Left/Right显示
2. 中英文切换（按L键）
3. 布局优化铺满
4. Rotation显示
5. 2位小数
"""

with open('src/viz_3d_enhanced.py', 'r', encoding='utf-8') as f:
    content = f.read()

# ============ Step 1: 添加imports ============
if 'from PIL import Image' not in content:
    content = content.replace(
        'import cv2',
        'import cv2\nfrom PIL import Image, ImageDraw, ImageFont\nimport platform'
    )

# ============ Step 2: 添加语言系统（在class之前）============
lang_code = '''
# =================== Language System ===================
class Lang:
    def __init__(self):
        self.current = "EN"
    
    def toggle(self):
        self.current = "CN" if self.current == "EN" else "EN"
        print(f"Language switched to: {self.current}")
    
    def t(self, key):
        d = {
            "EN": {
                "monitor": "Robot Monitor", "left": "Left Arm", "right": "Right Arm",
                "world": "3D World View", "traj": "Real-time Trajectories",
                "pos": "Position (m)", "grip": "Gripper Width (m)", "rot": "Rotation (deg)",
                "vis": "Visual", "ep": "Ep", "frame": "Frame", "speed": "Speed",
            },
            "CN": {
                "monitor": "机器人监控", "left": "左臂", "right": "右臂",
                "world": "3D世界视图", "traj": "实时轨迹",
                "pos": "位置 (米)", "grip": "夹爪宽度 (米)", "rot": "旋转角度 (度)",
                "vis": "视觉", "ep": "回合", "frame": "帧", "speed": "速度",
            }
        }
        return d[self.current].get(key, key)
    
    def is_cn(self):
        return self.current == "CN"

LANG = Lang()

def put_text(img, txt, pos, sz=20, clr=(255,255,255)):
    if LANG.is_cn():
        im = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        dr = ImageDraw.Draw(im)
        try:
            s = platform.system()
            ft = ImageFont.truetype('/System/Library/Fonts/PingFang.ttc' if s=='Darwin' else 
                                   'C:/Windows/Fonts/msyh.ttc' if s=='Windows' else
                                   '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc', sz)
        except:
            ft = ImageFont.load_default()
        dr.text(pos, txt, font=ft, fill=clr[::-1])
        return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    else:
        cv2.putText(img, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, sz/30.0, clr, 1, cv2.LINE_AA)
        return img
# ======================================================

'''

content = content.replace(
    'class Enhanced3DVisualizer(CombinedVisualizer):',
    lang_code + 'class Enhanced3DVisualizer(CombinedVisualizer):'
)

# ============ Step 3: 修改__init__，正确顺序 ============
# 先定义self.lang，window_name用它
content = content.replace(
    '        self.window_name = "Robot Monitor"',
    '        self.lang = LANG\n        self.window_name = self.lang.t("monitor")'
)

# ============ Step 4: 添加L键 ============
content = content.replace(
    "                elif key == ord('q'):",
    "                elif key == ord('l') or key == ord('L'):\n                    self.lang.toggle()\n                    continue\n                elif key == ord('q'):"
)

# ============ Step 5: 布局优化 ============
content = content.replace('w, h = 450, 750', 'w, h = 1100, 1000')
content = content.replace('(int(700 *', '(int(900 *')
content = content.replace('), 700))', '), 900)')

# ============ Step 6: Left/Right ============
content = content.replace('"Robot 0"', '"Left Arm"')
content = content.replace('"Robot 1"', '"Right Arm"')

# ============ Step 7: 2位小数 ============
content = content.replace('.3f', '.2f').replace('.4f', '.2f')

# ============ Step 8: 主要文本使用翻译 ============
content = content.replace('"3D World View"', 'self.lang.t("world")')
content = content.replace('"Real-time Trajectories"', 'self.lang.t("traj")')
content = content.replace('"Left Arm Position (m)"', 'f"{self.lang.t(\'left\')} {self.lang.t(\'pos\')}"')
content = content.replace('"Right Arm Position (m)"', 'f"{self.lang.t(\'right\')} {self.lang.t(\'pos\')}"')
content = content.replace('"Left Arm Rotation (deg)"', 'f"{self.lang.t(\'left\')} {self.lang.t(\'rot\')}"')
content = content.replace('"Right Arm Rotation (deg)"', 'f"{self.lang.t(\'right\')} {self.lang.t(\'rot\')}"')
content = content.replace('"Gripper Width (m)"', 'self.lang.t("grip")')

# 保存
with open('src/viz_3d_enhanced.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("="*60)
print("✓ 一键修复完成！")
print("="*60)
print("\n功能：")
print("  ✓ Left/Right显示")
print("  ✓ 按L键切换中英文")
print("  ✓ 布局优化，铺满屏幕")
print("  ✓ Rotation显示")
print("  ✓ 坐标2位小数")
print("="*60)
