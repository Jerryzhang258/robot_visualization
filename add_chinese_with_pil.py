# -*- coding: utf-8 -*-
"""
使用PIL实现真正的中文支持
"""

with open('src/viz_3d_enhanced.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. 添加PIL import
if 'from PIL import Image' not in content:
    content = content.replace(
        'import cv2',
        'import cv2\nfrom PIL import Image, ImageDraw, ImageFont\nimport platform'
    )
    print("✓ 添加PIL import")

# 2. 在class定义之前添加中文支持系统
chinese_system = '''
# =================== 中文支持系统 ===================
class LanguageSystem:
    def __init__(self):
        self.current = "EN"  # "EN" 或 "CN"
        self.translations = {
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
                "ltact": "L-Tact",
                "rtact": "R-Tact",
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
                "ltact": "左触觉",
                "rtact": "右触觉",
                "ep": "回合",
                "frame": "帧",
                "speed": "速度",
            }
        }
    
    def toggle(self):
        self.current = "CN" if self.current == "EN" else "EN"
        print(f"\\n语言切换: {self.current}")
    
    def t(self, key):
        return self.translations[self.current].get(key, key)
    
    def is_chinese(self):
        return self.current == "CN"

LANG_SYS = LanguageSystem()

def put_chinese_text(img, text, position, font_size=20, color=(255, 255, 255)):
    """
    在OpenCV图像上绘制文本（支持中文）
    """
    if not LANG_SYS.is_chinese():
        # 英文直接用cv2
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                   font_size/30.0, color, 1, cv2.LINE_AA)
        return img
    
    # 中文用PIL
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # 加载中文字体
    try:
        system = platform.system()
        if system == 'Darwin':  # macOS
            font = ImageFont.truetype('/System/Library/Fonts/PingFang.ttc', font_size)
        elif system == 'Windows':
            font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', font_size)
        else:  # Linux
            font = ImageFont.truetype('/usr/share/fonts/truetype/wqy/wqy-microhei.ttc', font_size)
    except:
        font = ImageFont.load_default()
    
    # BGR -> RGB for PIL
    rgb_color = color[::-1] if len(color) == 3 else color
    draw.text(position, text, font=font, fill=rgb_color)
    
    # 转回OpenCV格式
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
# ==================================================

'''

content = content.replace(
    'class Enhanced3DVisualizer(CombinedVisualizer):',
    chinese_system + 'class Enhanced3DVisualizer(CombinedVisualizer):'
)

# 3. 在__init__添加语言系统引用
content = content.replace(
    'self.window_name = "Robot Monitor"',
    'self.window_name = "Robot Monitor"\n        self.lang = LANG_SYS'
)

# 4. 添加L键切换
content = content.replace(
    "                elif key == ord('q'):",
    '''                elif key == ord('l') or key == ord('L'):
                    self.lang.toggle()
                    continue
                elif key == ord('q'):'''
)

# 5. 替换主要文本为翻译
replacements = {
    '"Robot Monitor"': 'self.lang.t("monitor")',
    '"3D World View"': 'self.lang.t("world")',
    '"Real-time Trajectories"': 'self.lang.t("traj")',
    '"Left Arm Position (m)"': 'f"{self.lang.t(\'left\')} {self.lang.t(\'pos\')}"',
    '"Right Arm Position (m)"': 'f"{self.lang.t(\'right\')} {self.lang.t(\'pos\')}"',
    '"Left Arm Rotation (deg)"': 'f"{self.lang.t(\'left\')} {self.lang.t(\'rot\')}"',
    '"Right Arm Rotation (deg)"': 'f"{self.lang.t(\'right\')} {self.lang.t(\'rot\')}"',
    '"Gripper Width (m)"': 'self.lang.t("grip")',
}

for old, new in replacements.items():
    content = content.replace(old, new)

# 6. 替换所有cv2.putText为put_chinese_text
# 只替换panel, header, control_bar上的文本
content = content.replace('cv2.putText(panel,', 'panel = put_chinese_text(panel,')
content = content.replace('cv2.putText(header,', 'header = put_chinese_text(header,')
content = content.replace('cv2.putText(control_bar,', 'control_bar = put_chinese_text(control_bar,')

# 移除cv2.FONT_HERSHEY_SIMPLEX参数（put_chinese_text不需要）
import re
content = re.sub(r', cv2\.FONT_HERSHEY_SIMPLEX, ([0-9.]+)', r', \1*30', content)
content = re.sub(r', \d+, cv2\.LINE_AA\)', ')', content)

with open('src/viz_3d_enhanced.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("="*60)
print("✓ 真正的中文支持已添加！")
print("="*60)
print("\n功能：")
print("  - 按 [L] 键切换中英文")
print("  - 中文显示真正的汉字")
print("  - 英文保持原样")
print("\n显示对照：")
print("  EN: Robot Monitor, Left Arm, Right Arm")
print("  CN: 机器人监控, 左臂, 右臂")
print("="*60)
