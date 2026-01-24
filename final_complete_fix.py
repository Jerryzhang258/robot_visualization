# -*- coding: utf-8 -*-
"""
最终完整修复 - 一键搞定
"""

with open('src/viz_3d_enhanced.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. 添加PIL import
if 'from PIL import Image' not in content:
    content = content.replace(
        'import cv2',
        'import cv2\nfrom PIL import Image, ImageDraw, ImageFont\nimport platform'
    )

# 2. 添加完整的语言和中文绘制系统
lang_system = '''
# =================== Language & Chinese Support ===================
class LanguageSystem:
    def __init__(self):
        self.lang = "EN"
        self.trans = {
            "EN": {"monitor": "Robot Monitor", "left": "Left", "right": "Right",
                   "world": "3D World View", "traj": "Real-time Trajectories",
                   "pos": "Position (m)", "grip": "Gripper Width (m)", "rot": "Rotation (deg)"},
            "CN": {"monitor": "机器人监控", "left": "左臂", "right": "右臂",
                   "world": "3D世界视图", "traj": "实时轨迹",
                   "pos": "位置 (米)", "grip": "夹爪宽度 (米)", "rot": "旋转角度 (度)"}
        }
    
    def toggle(self):
        self.lang = "CN" if self.lang == "EN" else "EN"
        print(f"Language: {self.lang}")
    
    def t(self, k):
        return self.trans[self.lang].get(k, k)
    
    def is_cn(self):
        return self.lang == "CN"

_LANG = LanguageSystem()

def cv2_put_text_cn(img, text, pos, font_scale=0.5, color=(255,255,255), thickness=1):
    """支持中英文的putText"""
    if _LANG.is_cn():
        im = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        dr = ImageDraw.Draw(im)
        try:
            s = platform.system()
            fpath = '/System/Library/Fonts/PingFang.ttc' if s=='Darwin' else 'C:/Windows/Fonts/msyh.ttc' if s=='Windows' else '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'
            ft = ImageFont.truetype(fpath, int(font_scale*30))
        except:
            ft = ImageFont.load_default()
        dr.text(pos, text, font=ft, fill=color[::-1])
        return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    else:
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
        return img
# ================================================================

'''

content = content.replace(
    'class Enhanced3DVisualizer(CombinedVisualizer):',
    lang_system + 'class Enhanced3DVisualizer(CombinedVisualizer):'
)

# 3. 修改__init__
content = content.replace(
    '        self.window_name = "Robot Monitor"',
    '        self.lang = _LANG\n        self.window_name = self.lang.t("monitor")'
)

# 4. 添加L键（在q键之前）
if "elif key == ord('l')" not in content:
    content = content.replace(
        "                elif key == ord('q'):",
        "                elif key == ord('l') or key == ord('L'):\n" +
        "                    self.lang.toggle()\n" +
        "                    continue\n" +
        "                elif key == ord('q'):"
    )

# 5. 替换所有cv2.putText为cv2_put_text_cn
import re
content = re.sub(
    r'cv2\.putText\((panel|header|control_bar), ',
    r'\1 = cv2_put_text_cn(\1, ',
    content
)

# 6. 替换主要文本
content = content.replace('"Robot Monitor"', 'self.lang.t("monitor")')
content = content.replace('"3D World View"', 'self.lang.t("world")')
content = content.replace('"Real-time Trajectories"', 'self.lang.t("traj")')
content = content.replace('"Left Arm Position (m)"', 'f"{self.lang.t(\'left\')} {self.lang.t(\'pos\')}"')
content = content.replace('"Right Arm Position (m)"', 'f"{self.lang.t(\'right\')} {self.lang.t(\'pos\')}"')
content = content.replace('"Gripper Width (m)"', 'self.lang.t("grip")')
content = content.replace('"Left Arm"', 'self.lang.t("left")')
content = content.replace('"Right Arm"', 'self.lang.t("right")')

# 7. 布局和格式优化
content = content.replace('w, h = 450, 750', 'w, h = 1100, 1000')
content = content.replace('.3f', '.2f').replace('.4f', '.2f')

with open('src/viz_3d_enhanced.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("="*60)
print("✓ 一键修复完成！")
print("="*60)
print("功能：")
print("  - 按L键切换中英文")
print("  - 真正的中文汉字显示")
print("  - 布局优化")
print("="*60)
