"""
人员追踪模块
追踪每个人员的姿态状态历史
"""
import numpy as np
from collections import deque
from typing import Dict, Tuple
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Settings, constants as const
from core import GeometryCalculator


class PersonState:
    """人员状态追踪器"""
    
    def __init__(self, person_id: int, history_size: int = 10):
        self.person_id = person_id
        self.score_history = deque(maxlen=history_size)
        self.keypoints_history = deque(maxlen=history_size)
        self.torso_positions = deque(maxlen=30)
        self.wrist_positions = deque(maxlen=30)
        self.head_pitch_history = deque(maxlen=30)
        self.timestamps = deque(maxlen=history_size)
        
        # 累计统计
        self.total_frames = 0
        self.head_down_frames = 0
        self.forward_frames = 0
    
    def add_frame(self, score: float, keypoints: np.ndarray, timestamp: float, 
                  forward_threshold: float = 0.4):
        """添加一帧数据"""
        self.score_history.append(score)
        self.keypoints_history.append(keypoints.copy())
        self.timestamps.append(timestamp)
        self.total_frames += 1
        
        if score > forward_threshold:
            self.forward_frames += 1
    
    def update_torso_position(self, torso_center: Tuple[float, float]):
        """更新躯干位置"""
        self.torso_positions.append(torso_center)
    
    def update_wrist_positions(self, left_wrist: Tuple[float, float], 
                             right_wrist: Tuple[float, float]):
        """更新手腕位置"""
        self.wrist_positions.append((left_wrist, right_wrist))
    
    def update_head_pitch(self, pitch: float):
        """更新头部俯仰角"""
        self.head_pitch_history.append(pitch)
        if pitch > const.HEAD_DOWN_THRESHOLD:
            self.head_down_frames += 1
    
    def get_forward_rate(self) -> float:
        """获取正视率"""
        if self.total_frames == 0:
            return 0.0
        return self.forward_frames / self.total_frames


class PersonTracker:
    """人员追踪器"""
    
    def __init__(self, history_size: int = 10):
        self.person_states: Dict[int, PersonState] = {}
        self.person_histories: Dict[int, deque] = {}
        self.history_size = history_size
    
    def get_or_create_person(self, person_id: int) -> PersonState:
        """获取或创建人员状态"""
        if person_id not in self.person_states:
            self.person_states[person_id] = PersonState(person_id=person_id, history_size=self.history_size)
            self.person_histories[person_id] = deque(maxlen=self.history_size)
        return self.person_states[person_id]
    
    def update_person(self, person_id: int, score: float, keypoints: np.ndarray, 
                     dt: float, forward_threshold: float = 0.4):
        """更新人员状态"""
        person_state = self.get_or_create_person(person_id)
        
        # 添加帧数据
        person_state.add_frame(score, keypoints, 0, forward_threshold)
        
        # 更新历史
        history = self.person_histories.setdefault(person_id, deque(maxlen=self.history_size))
        history.append(score)
        
        # 更新躯干位置 - 使用 GeometryCalculator
        torso_center = GeometryCalculator.get_torso_center(keypoints)
        person_state.update_torso_position((torso_center[0], torso_center[1]))
        
        # 更新手腕位置
        left_wrist = (keypoints[const.LEFT_WRIST][0], keypoints[const.LEFT_WRIST][1]) if keypoints[const.LEFT_WRIST][0] > 0 else (0, 0)
        right_wrist = (keypoints[const.RIGHT_WRIST][0], keypoints[const.RIGHT_WRIST][1]) if keypoints[const.RIGHT_WRIST][0] > 0 else (0, 0)
        person_state.update_wrist_positions(left_wrist, right_wrist)
        
        # 更新头部俯仰角 - 使用 GeometryCalculator
        head_pitch = GeometryCalculator.calculate_head_pitch(keypoints)
        person_state.update_head_pitch(head_pitch)
    
    def get_average_score(self, person_id: int) -> float:
        """获取滑动平均评分"""
        person_state = self.person_states.get(person_id)
        if person_state and len(person_state.score_history) > 0:
            return sum(person_state.score_history) / len(person_state.score_history)
        return 0.0
