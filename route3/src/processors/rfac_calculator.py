"""
RFAC四维度计算模块
计算注意力涣散、疲劳、匆忙、情绪失控四个维度
"""
import numpy as np
from typing import Dict, List, Tuple
from collections import deque
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Settings, constants as const
from core import GeometryCalculator
from .person_tracker import PersonState


class RFACScore:
    """RFAC综合评分结果"""
    
    def __init__(self):
        # 四维度评分 (0-1, 越高表示异常程度越高)
        self.apathy_score: float = 0.0
        self.fatigue_score: float = 0.0
        self.rushing_score: float = 0.0
        self.frustration_score: float = 0.0
        
        # 详细指标
        self.forward_rate: float = 0.0
        self.head_down_ratio: float = 0.0
        self.torso_velocity: float = 0.0
        self.gesture_activity: float = 0.0
        self.limb_antagonism: float = 0.0
        self.posture_instability: float = 0.0
        
        # 综合评分
        self.overall_score: float = 0.0
        self.attention_level: str = const.AttentionLevel.NORMAL
    
    def __str__(self) -> str:
        return (f"RFAC(Apathy={self.apathy_score:.2f}, "
                f"Fatigue={self.fatigue_score:.2f}, "
                f"Rushing={self.rushing_score:.2f}, "
                f"Frustration={self.frustration_score:.2f}, "
                f"Overall={self.overall_score:.2f})")


class RFACCalculator:
    """RFAC四维度计算器"""
    
    def __init__(self):
        self.settings = Settings()
        self.forward_threshold = self.settings.forward_threshold
    
    def calculate(self, person_state: PersonState, dt: float = 0.05) -> RFACScore:
        """
        计算RFAC综合评分
        
        Args:
            person_state: 人员状态追踪器
            dt: 时间间隔
            
        Returns:
            RFACScore对象
        """
        score = RFACScore()
        
        # 1. Apathy（注意力涣散）
        if person_state.keypoints_history:
            latest_kps = list(person_state.keypoints_history)[-1]
            apathy_raw = self._calculate_apathy(latest_kps)
            score.apathy_score = apathy_raw
            score.forward_rate = person_state.get_forward_rate()
        
        # 2. Fatigue（疲劳状态）
        score.fatigue_score = self._calculate_fatigue(person_state)
        score.head_down_ratio = self._calculate_head_down_ratio(person_state)
        score.posture_instability = self._calculate_posture_instability(person_state.keypoints_history)
        
        # 3. Rushing（匆忙状态）
        score.rushing_score = self._calculate_rushing(person_state, dt)
        
        # 更新躯干速度
        if len(person_state.torso_positions) >= 2:
            current = person_state.torso_positions[-1]
            prev = person_state.torso_positions[-2]
            scale = 1.0
            if person_state.keypoints_history:
                latest_kps = list(person_state.keypoints_history)[-1]
                scale = GeometryCalculator.calculate_scale_factor(latest_kps)
            score.torso_velocity = GeometryCalculator.calculate_torso_velocity(current, prev, dt, scale)
        
        # 更新手势活跃度
        if person_state.wrist_positions and person_state.keypoints_history:
            latest_kps = list(person_state.keypoints_history)[-1]
            shoulder = GeometryCalculator.get_shoulder_center(latest_kps)
            scale = GeometryCalculator.calculate_scale_factor(latest_kps)
            score.gesture_activity = GeometryCalculator.calculate_gesture_activity(
                person_state.wrist_positions, shoulder, scale)
        
        # 4. Frustration（情绪失控）
        score.frustration_score = self._calculate_frustration(person_state)
        score.limb_antagonism = GeometryCalculator.calculate_wrist_acceleration(
            person_state.wrist_positions, dt)
        
        # 5. 综合评分
        score.overall_score = (
            const.RFAC_WEIGHTS["apathy"] * score.apathy_score +
            const.RFAC_WEIGHTS["fatigue"] * score.fatigue_score +
            const.RFAC_WEIGHTS["rushing"] * score.rushing_score +
            const.RFAC_WEIGHTS["frustration"] * score.frustration_score
        )
        
        # 6. 注意力等级判定
        score.attention_level = self._get_attention_level(score.overall_score)
        
        return score
    
    def _calculate_apathy(self, keypoints: np.ndarray) -> float:
        """计算注意力涣散评分"""
        # 使用 GeometryCalculator 计算几何评分，然后取反得到涣散评分
        _, _, geometry_score = GeometryCalculator.calculate_geometry_score(
            keypoints,
            self.settings.eye_score_ratio,
            self.settings.nose_score_ratio
        )
        return 1.0 - geometry_score
    
    def _calculate_fatigue(self, person_state: PersonState) -> float:
        """计算疲劳状态综合评分"""
        head_down_ratio = self._calculate_head_down_ratio(person_state)
        body_sway = GeometryCalculator.calculate_body_sway(
            list(person_state.keypoints_history) if person_state.keypoints_history else [])
        return min(0.7 * head_down_ratio + 0.3 * body_sway, 1.0)
    
    def _calculate_head_down_ratio(self, person_state: PersonState) -> float:
        """计算低头时间占比"""
        if person_state.total_frames == 0:
            return 0.0
        return person_state.head_down_frames / person_state.total_frames
    
    def _calculate_rushing(self, person_state: PersonState, dt: float) -> float:
        """计算匆忙状态综合评分"""
        torso_velocity = 0.0
        if len(person_state.torso_positions) >= 2:
            current = person_state.torso_positions[-1]
            prev = person_state.torso_positions[-2]
            scale = 1.0
            if person_state.keypoints_history:
                latest_kps = list(person_state.keypoints_history)[-1]
                scale = GeometryCalculator.calculate_scale_factor(latest_kps)
            torso_velocity = GeometryCalculator.calculate_torso_velocity(current, prev, dt, scale)
        
        gesture_activity = 0.0
        if len(person_state.wrist_positions) >= 2 and person_state.keypoints_history:
            latest_kps = list(person_state.keypoints_history)[-1]
            shoulder = GeometryCalculator.get_shoulder_center(latest_kps)
            scale = GeometryCalculator.calculate_scale_factor(latest_kps)
            gesture_activity = GeometryCalculator.calculate_gesture_activity(
                person_state.wrist_positions, shoulder, scale)
        
        return min(0.4 * torso_velocity + 0.4 * gesture_activity + 0.2 * torso_velocity * gesture_activity, 1.0)
    
    
    def _calculate_frustration(self, person_state: PersonState) -> float:
        """计算情绪失控综合评分"""
        limb_antagonism = GeometryCalculator.calculate_wrist_acceleration(
            person_state.wrist_positions)
        forward_lean = 0.0
        if person_state.keypoints_history and len(person_state.keypoints_history) >= 2:
            kps_list = list(person_state.keypoints_history)
            forward_lean = self._calculate_forward_lean(kps_list[-1], kps_list[-2])
        posture_instability = GeometryCalculator.calculate_posture_instability(
            list(person_state.keypoints_history) if person_state.keypoints_history else [])
        
        return min(0.4 * limb_antagonism + 0.3 * forward_lean + 0.3 * posture_instability, 1.0)
    
    def _calculate_forward_lean(self, keypoints: np.ndarray, prev_keypoints: np.ndarray = None) -> float:
        """计算身体前倾程度"""
        lean_angle = GeometryCalculator.calculate_body_lean_angle(keypoints)
        
        angle_change = 0.0
        if prev_keypoints is not None:
            prev_angle = GeometryCalculator.calculate_body_lean_angle(prev_keypoints)
            angle_change = GeometryCalculator.calculate_angle_change(lean_angle, prev_angle)
        
        lean_score = min(lean_angle / const.MAX_NORMAL_LEAN, 1.0)
        change_score = min(angle_change / const.ANGLE_CHANGE_THRESHOLD, 1.0)
        
        return 0.6 * lean_score + 0.4 * change_score
    
    def _calculate_posture_instability(self, keypoints_history: deque) -> float:
        """计算姿态不稳定度"""
        return GeometryCalculator.calculate_posture_instability(list(keypoints_history) if keypoints_history else [])
    
    def _get_attention_level(self, score: float) -> str:
        """根据评分获取注意力等级"""
        thresholds = self.settings.level_thresholds
        if score < thresholds["NORMAL"]:
            return const.AttentionLevel.NORMAL
        elif score < thresholds["MILD"]:
            return const.AttentionLevel.MILD
        elif score < thresholds["MODERATE"]:
            return const.AttentionLevel.MODERATE
        else:
            return const.AttentionLevel.SEVERE
