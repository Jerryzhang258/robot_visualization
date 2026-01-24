# -*- coding: utf-8 -*-

with open('src/viz_3d_enhanced.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 添加imports
content = content.replace(
    'import cv2\n',
    'import cv2\nfrom PIL import Image, ImageDraw, ImageFont\nimport platform\n'
)

# 完整的语言系统（手写，没有bug）
lang_code = '''
# ================= Language System =================
class LangSys:
    def __init__(self):
        self.lang = "EN"
    
    def toggle(self):
        self.lang = "CN" if self.lang == "EN" else "EN"
        print(f"Language: {self.lang}")
    
    def t(self, key):
        trans = {
            "EN": {
                "monitor": "Robot Monitor",
                "left": "Left Arm",
                "right": "Right Arm",
                "world": "3D World View",
                "traj": "Real-time Trajectories",
                "pos": "Position (m)",
                "grip": "Gripper Width (m)",
            },
            "CN": {
                "monitor": "机器人监控",
                "left": "左臂",
                "right": "右臂",
                "world": "3D世界视图",
                "traj": "实时轨迹",
                "pos": "位置 (米)",
                "grip": "夹爪宽度 (米)",
            }
        }
        return trans[self.lang].get(key, key)
    
    def is_cn(self):
        return self.lang == "CN"

LS = LangSys()

def ptxt(img, txt, pos, sz=0.5, clr=(255,255,255)):
    if LS.is_cn():
        im = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        dr = ImageDraw.Draw(im)
        try:
            s = platform.system()
            fp = '/System/Library/Fonts/PingFang.ttc' if s=='Darwin' else 'C:/Windows/Fonts/msyh.ttc' if s=='Windows' else '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'
            ft = ImageFont.truetype(fp, int(sz*30))
        except:
            ft = ImageFont.load_default()
        dr.text(pos, txt, font=ft, fill=clr[::-1])
        return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    cv2.putText(img, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, sz, clr, 1, cv2.LINE_AA)
    return img
# ===================================================

'''

content = content.replace(
    'class Enhanced3DVisualizer(CombinedVisualizer):',
    lang_code + 'class Enhanced3DVisualizer(CombinedVisualizer):'
)

# 修改__init__
content = content.replace(
    '        self.window_name = "Robot Monitor"\n        self.playback_speed',
    '        self.lang = LS\n        self.window_name = LS.t("monitor")\n        self.playback_speed'
)

# 添加L键
content = content.replace(
    "                elif key == ord('q'):",
    "                elif key == ord('l') or key == ord('L'):\n                    LS.toggle()\n                    continue\n                elif key == ord('q'):"
)

# 只替换header和panel的文本，保持简单
content = content.replace('"Robot Monitor"', 'LS.t("monitor")')
content = content.replace('"Left Arm"', 'LS.t("left")')
content = content.replace('"Right Arm"', 'LS.t("right")')
content = content.replace('"3D World View"', 'LS.t("world")')
content = content.replace('"Real-time Trajectories"', 'LS.t("traj")')

# 简单的putText包装（不改函数名，只包装返回值）
import re
lines = content.split('\n')
new_lines = []
for line in lines:
    if 'cv2.putText(panel,' in line or 'cv2.putText(header,' in line:
        # 在行首添加变量赋值
        if 'cv2.putText(panel,' in line:
            line = '                panel = ptxt(panel,' + line.split('cv2.putText(panel,')[1]
            line = re.sub(r', cv2\.FONT_HERSHEY_SIMPLEX, ([0-9.]+), \([^)]+\), \d+, cv2\.LINE_AA\)', r', \1, \2)', line)
        elif 'cv2.putText(header,' in line:
            line = '        header = ptxt(header,' + line.split('cv2.putText(header,')[1]
            line = re.sub(r', cv2\.FONT_HERSHEY_SIMPLEX, ([0-9.]+), \([^)]+\), \d+, cv2\.LINE_AA\)', r', \1, \2)', line)
    new_lines.append(line)

content = '\n'.join(new_lines)

# 布局优化
content = content.replace('w, h = 450, 750', 'w, h = 1100, 1000')
content = content.replace('.3f', '.2f').replace('.4f', '.2f')

with open('src/viz_3d_enhanced.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✓ 完成!")
