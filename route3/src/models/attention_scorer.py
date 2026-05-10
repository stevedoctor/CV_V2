"""
注意力评分模块
基于几何约束计算注意力评分
"""
import numpy as np
from typing import Tuple
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Settings, constants as const
from core import GeometryCalculator


class AttentionScorer:
    """注意力评分器"""
    
    def __init__(self, 
                 eye_ratio: float = None,
                 nose_ratio: float = None):
        """
        初始化注意力评分器
        
        Args:
            eye_ratio: 眼睛对称性评分系数
            nose_ratio: 鼻子偏移评分系数
        """
        settings = Settings()
        self.eye_ratio = eye_ratio or settings.eye_score_ratio
        self.nose_ratio = nose_ratio or settings.nose_score_ratio
    
    def calculate_geometry_score(self, keypoints: np.ndarray) -> Tuple[float, float, float]:
        """
        计算几何约束评分
        
        Args:
            keypoints: 关键点坐标 (17, 3)
            
        Returns:
            (eye_score, nose_score, geometry_score)
        """
        # 使用 GeometryCalculator 进行计算
        return GeometryCalculator.calculate_geometry_score(
            keypoints, 
            self.eye_ratio, 
            self.nose_ratio
        )
    
    def calculate_current_score(self, keypoints: np.ndarray) -> float:
        """
        计算当前帧的姿态评分
        
        Args:
            keypoints: 关键点坐标 (17, 3)
            
        Returns:
            姿态评分 (0-1)
        """
        l_eye = keypoints[const.LEFT_EYE]
        r_eye = keypoints[const.RIGHT_EYE]
        l_shoulder = keypoints[const.LEFT_SHOULDER]
        
        # 检查有效性
        if l_eye[0] == 0 or r_eye[0] == 0 or l_shoulder[0] == 0:
            return 0.2
        
        # 使用 GeometryCalculator 的自适应评分（支持遮挡处理）
        return GeometryCalculator.calculate_occlusion_adaptive_score(
            keypoints, 
            self.eye_ratio, 
            self.nose_ratio
        )
