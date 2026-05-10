"""
系统常量定义
"""

# ==================== 颜色定义 (BGR格式) ====================
COLOR_MAP = {
    "NORMAL": (0, 255, 0),      # 绿色
    "MILD": (0, 255, 255),      # 黄色
    "MODERATE": (0, 128, 255),  # 橙色
    "SEVERE": (0, 0, 255)       # 红色
}

# ==================== 注意力等级 ====================
class AttentionLevel:
    NORMAL = "NORMAL"
    MILD = "MILD"
    MODERATE = "MODERATE"
    SEVERE = "SEVERE"


# ==================== YOLO关键点索引 (COCO 17点) ====================
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# 关键点索引
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16

# ==================== 角度阈值 ====================
HEAD_DOWN_THRESHOLD = 15  # 低头角度阈值（度）
MAX_NORMAL_LEAN = 30       # 正常前倾角度（度）
ANGLE_CHANGE_THRESHOLD = 10  # 角度突变阈值（度）

# ==================== RFAC权重配置 ====================
RFAC_WEIGHTS = {
    "apathy": 0.55,
    "fatigue": 0.25,
    "rushing": 0.15,
    "frustration": 0.05
}

# ==================== 归一化参数 ====================
NORMALIZATION_PARAMS = {
    "body_sway_scale": 20.0,
    "torso_velocity_scale": 50.0,
    "gesture_activity_scale": 100.0,
    "angular_velocity_scale": 30.0,
    "acceleration_scale": 200.0,
    "instability_scale": 15.0
}
