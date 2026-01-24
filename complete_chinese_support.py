# -*- coding: utf-8 -*-
"""
完整的中文支持 - 完全重写
"""

# 读取原始文件
with open('src/viz_3d_enhanced.py', 'r', encoding='utf-8') as f:
    original = f.read()

# 准备要插入的完整代码
imports_addition = '''from PIL import Image, ImageDraw, ImageFont
import platform
'''

language_system = '''
# =================== 语言系统 ===================
class LangSystem:
    def __init__(self):
        self.lang = "EN"
        
    def toggle(self):
        self.lang = "CN" if self.lang == "EN" else "EN"
        print(f"Language: {self.lang}")
    
    def t(self, key):
        translations = {
            "EN": {
                "monitor": "Robot Monitor",
                "left": "Left Arm",
                "right": "Right Arm",
                "world": "3D World View",
                "traj": "Real-time Trajectories",
                "pos": "Position (m)",
                "grip": "Gripper Width (m)",
                "rot": "Rotation (deg)",
                "vis": "Visual",
                "ep": "Ep",
                "frame": "Frame",
                "speed": "Speed",
            },
            "CN": {
                "monitor": "机器人监控",
                "left": "左臂",
                "right": "右臂",
                "world": "3D世界视图",
                "traj": "实时轨迹",
                "pos": "位置 (米)",
                "grip": "夹爪宽度 (米)",
                "rot": "旋转角度 (度)",
                "vis": "视觉",
                "ep": "回合",
                "frame": "帧",
                "speed": "速度",
            }
        }
        return translations[self.lang].get(key, key)
    
    def is_cn(self):
        return self.lang == "CN"

LANG = LangSystem()

def put_text(img, text, pos, size=20, color=(255,255,255)):
    """智能文本绘制：中文用PIL，英文用cv2"""
    if LANG.is_cn():
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        try:
            if platform.system() == 'Darwin':
                font = ImageFont.truetype('/System/Library/Fonts/PingFang.ttc', size)
            elif platform.system() == 'Windows':
                font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', size)
            else:
                font = ImageFont.truetype('/usr/share/fonts/truetype/wqy/wqy-microhei.ttc', size)
        except:
            font = ImageFont.load_default()
        draw.text(pos, text, font=font, fill=color[::-1])
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    else:
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, size/30.0, color, 1, cv2.LINE_AA)
        return img
# ================================================

'''

# 1. 添加imports
original = original.replace('import cv2\n', 'import cv2\n' + imports_addition)

# 2. 添加语言系统
original = original.replace(
    'class Enhanced3DVisualizer(CombinedVisualizer):',
    language_system + 'class Enhanced3DVisualizer(CombinedVisualizer):'
)

# 3. 在__init__添加self.lang
original = original.replace(
    'self.window_name = "Robot Monitor"',
    'self.window_name = "Robot Monitor"\n        self.lang = LANG'
)

# 4. 添加L键
original = original.replace(
    "                elif key == ord('q'):",
    "                elif key == ord('l') or key == ord('L'):\n" +
    "                    self.lang.toggle()\n" +
    "                    continue\n" +
    "                elif key == ord('q'):"
)

# 5. 布局调整
original = original.replace('w, h = 450, 750', 'w, h = 1100, 1000')
original = original.replace('world_image = cv2.resize(world_image, (int(700 *', 'world_image = cv2.resize(world_image, (int(900 *')
original = original.replace('), 700))', '), 900))')
original = original.replace('.3f', '.2f').replace('.4f', '.2f')

# 6. 改为Left/Right
original = original.replace('"Robot 0"', '"Left Arm"')
original = original.replace('"Robot 1"', '"Right Arm"')

# 7. 替换文本为翻译（只替换主要的）
original = original.replace('"Robot Monitor"', 'self.lang.t("monitor")')
original = original.replace('"3D World View"', 'self.lang.t("world")')
original = original.replace('"Real-time Trajectories"', 'self.lang.t("traj")')
original = original.replace('"Left Arm Position (m)"', 'f"{self.lang.t(\'left\')} {self.lang.t(\'pos\')}"')
original = original.replace('"Right Arm Position (m)"', 'f"{self.lang.t(\'right\')} {self.lang.t(\'pos\')}"')

# 保存
with open('src/viz_3d_enhanced.py', 'w', encoding='utf-8') as f:
    f.write(original)

print("✓ 完整版本创建成功！")
