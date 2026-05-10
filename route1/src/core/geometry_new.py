"""
修复后的头部俯仰角计算

问题：原计算方式得到 -97.7° 的异常值
原因：arctan2(dy, dx) 计算的是向量角度，不是真正的俯仰角

修正方案：使用鼻子相对于眼睛中心的垂直偏移来估算
"""
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import constants as const


def calculate_head_pitch_v2(keypoints: np.ndarray) -> float:
    """
    计算头部俯仰角（修正版本）
    
    原理：
    - 使用鼻子相对于眼睛中心的垂直位置来判断低头
    - 正常正视时，鼻子约在眼睛下方 0.3-0.5 倍眼距
    - 低头时，鼻子相对眼睛向下移动更多
    
    Args:
        keypoints: 关键点数组 (17, 3)
        
    Returns:
        俯仰角（度）: 正值=低头, 负值=抬头, 0=正视
    """
    left_eye = keypoints[const.LEFT_EYE]
    right_eye = keypoints[const.RIGHT_EYE]
    nose = keypoints[const.NOSE]
    
    if left_eye[0] == 0 or right_eye[0] == 0 or nose[0] == 0:
        return 0.0
    
    # 计算眼睛中心
    eye_center_y = (left_eye[1] + right_eye[1]) / 2
    eye_center_x = (left_eye[0] + right_eye[0]) / 2
    
    # 计算眼距（用于归一化）
    eye_distance = abs(right_eye[0] - left_eye[0])
    if eye_distance < 1e-6:
        return 0.0
    
    # 鼻子相对眼睛中心的垂直偏移（正值=鼻子在眼睛下方=低头）
    nose_offset_y = nose[1] - eye_center_y
    
    # 归一化：相对于眼距
    normalized_offset = nose_offset_y / eye_distance
    
    # 正常正视时，normalized_offset 约为 0.3-0.5
    # 转换为角度估算：
    # - normalized_offset < 0.3: 抬头 (< 0°)
    # - normalized_offset ≈ 0.4: 正视 (≈ 0°)
    # - normalized_offset > 0.6: 低头 (> 15°)
    
    # 线性映射：将 0.2-0.8 映射到 -30° 到 +45°
    pitch = (normalized_offset - 0.4) * 100  # 粗略估算
    
    return pitch


if __name__ == "__main__":
    # 测试
    test_keypoints = np.array([
        [100, 100, 1],   # nose - 正常位置
        [90, 80, 1],     # left_eye
        [110, 80, 1],    # right_eye
    ])
    
    pitch = calculate_head_pitch_v2(test_keypoints)
    print(f"测试俯仰角: {pitch:.1f}°")
