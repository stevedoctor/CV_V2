"""
几何计算核心模块
集中管理所有几何计算逻辑，消除代码重复
"""
import numpy as np
from typing import Tuple, Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import constants as const


class GeometryCalculator:
    """几何计算工具类"""
    
    # ==================== 尺度因子计算 ====================
    
    @staticmethod
    def calculate_scale_factor(keypoints: np.ndarray) -> float:
        """
        计算归一化尺度因子（基于肩宽）
        
        Args:
            keypoints: 关键点数组 (17, 3)
            
        Returns:
            尺度因子
        """
        left_shoulder = keypoints[const.LEFT_SHOULDER]
        right_shoulder = keypoints[const.RIGHT_SHOULDER]
        shoulder_width = np.abs(right_shoulder[0] - left_shoulder[0])
        return shoulder_width if shoulder_width > 1e-6 else 1.0
    
    @staticmethod
    def calculate_pixel_to_metric(pixel_distance: float, scale_factor: float) -> float:
        """
        像素距离转换为归一化度量距离
        
        Args:
            pixel_distance: 像素距离
            scale_factor: 尺度因子
            
        Returns:
            归一化距离
        """
        return pixel_distance / (scale_factor + 1e-6)
    
    # ==================== 中心点计算 ====================
    
    @staticmethod
    def get_shoulder_center(keypoints: np.ndarray) -> np.ndarray:
        """
        获取肩膀中心点
        
        Args:
            keypoints: 关键点数组 (17, 3)
            
        Returns:
            肩膀中心点坐标
        """
        left_shoulder = keypoints[const.LEFT_SHOULDER]
        right_shoulder = keypoints[const.RIGHT_SHOULDER]
        return (left_shoulder + right_shoulder) / 2
    
    @staticmethod
    def get_hip_center(keypoints: np.ndarray) -> np.ndarray:
        """
        获取髋部中心点
        
        Args:
            keypoints: 关键点数组 (17, 3)
            
        Returns:
            髋部中心点坐标
        """
        left_hip = keypoints[const.LEFT_HIP]
        right_hip = keypoints[const.RIGHT_HIP]
        return (left_hip + right_hip) / 2
    
    @staticmethod
    def get_torso_center(keypoints: np.ndarray) -> np.ndarray:
        """
        获取躯干中心点（肩部和髋部的中心）
        
        Args:
            keypoints: 关键点数组 (17, 3)
            
        Returns:
            躯干中心点坐标
        """
        shoulder_center = GeometryCalculator.get_shoulder_center(keypoints)
        hip_center = GeometryCalculator.get_hip_center(keypoints)
        return (shoulder_center + hip_center) / 2
    
    @staticmethod
    def get_eye_center(keypoints: np.ndarray) -> Optional[np.ndarray]:
        """
        获取眼睛中心点
        
        Args:
            keypoints: 关键点数组 (17, 3)
            
        Returns:
            眼睛中心点坐标，如果眼睛不可见则返回None
        """
        left_eye = keypoints[const.LEFT_EYE]
        right_eye = keypoints[const.RIGHT_EYE]
        
        if left_eye[0] == 0 or right_eye[0] == 0:
            return None
        
        return (left_eye + right_eye) / 2
    
    @staticmethod
    def get_wrist_center(keypoints: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        获取双手腕中心点
        
        Args:
            keypoints: 关键点数组 (17, 3)
            
        Returns:
            (左腕中心, 右腕中心)
        """
        left_wrist = keypoints[const.LEFT_WRIST]
        right_wrist = keypoints[const.RIGHT_WRIST]
        
        l_center = left_wrist if left_wrist[0] > 0 else None
        r_center = right_wrist if right_wrist[0] > 0 else None
        
        return l_center, r_center
    
    # ==================== 距离计算 ====================
    
    @staticmethod
    def calculate_shoulder_width(keypoints: np.ndarray) -> float:
        """
        计算肩宽
        
        Args:
            keypoints: 关键点数组 (17, 3)
            
        Returns:
            肩宽（像素）
        """
        left_shoulder = keypoints[const.LEFT_SHOULDER]
        right_shoulder = keypoints[const.RIGHT_SHOULDER]
        return np.abs(right_shoulder[0] - left_shoulder[0])
    
    @staticmethod
    def calculate_eye_distance(keypoints: np.ndarray) -> float:
        """
        计算双眼距离
        
        Args:
            keypoints: 关键点数组 (17, 3)
            
        Returns:
            双眼距离（像素）
        """
        left_eye = keypoints[const.LEFT_EYE]
        right_eye = keypoints[const.RIGHT_EYE]
        return np.abs(right_eye[0] - left_eye[0])
    
    @staticmethod
    def calculate_ear_distance(keypoints: np.ndarray) -> float:
        """
        计算双耳距离
        
        Args:
            keypoints: 关键点数组 (17, 3)
            
        Returns:
            双耳距离（像素）
        """
        left_ear = keypoints[const.LEFT_EAR]
        right_ear = keypoints[const.RIGHT_EAR]
        return np.abs(right_ear[0] - left_ear[0])
    
    @staticmethod
    def calculate_point_distance(p1: np.ndarray, p2: np.ndarray) -> float:
        """
        计算两点之间的欧几里得距离
        
        Args:
            p1: 点1坐标 (x, y) 或 (x, y, conf)
            p2: 点2坐标 (x, y) 或 (x, y, conf)
            
        Returns:
            距离
        """
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return np.sqrt(dx**2 + dy**2)
    
    @staticmethod
    def calculate_nose_offset(keypoints: np.ndarray) -> float:
        """
        计算鼻子相对于眼睛中心的偏移
        
        Args:
            keypoints: 关键点数组 (17, 3)
            
        Returns:
            鼻子X轴偏移量（像素）
        """
        nose = keypoints[const.NOSE]
        eye_center = GeometryCalculator.get_eye_center(keypoints)
        
        if eye_center is None:
            return float('inf')
        
        return np.abs(nose[0] - eye_center[0])
    
    # ==================== 角度计算 ====================
    
    @staticmethod
    def calculate_head_pitch(keypoints: np.ndarray) -> float:
        """
        计算头部俯仰角（度数）- 修正版
        
        使用鼻子相对于眼睛中心的垂直位置来判断低头程度。
        正值表示低头，负值表示抬头，0表示正视。
        
        原版问题：使用肩膀中心计算导致角度约 -90°
        修正方案：使用眼睛中心作为参考点
        
        Args:
            keypoints: 关键点数组 (17, 3)
            
        Returns:
            俯仰角（度）：典型范围 -30° ~ +45°
        """
        left_eye = keypoints[const.LEFT_EYE]
        right_eye = keypoints[const.RIGHT_EYE]
        nose = keypoints[const.NOSE]
        
        # 有效性检查
        if left_eye[0] == 0 or right_eye[0] == 0 or nose[0] == 0:
            return 0.0
        
        # 计算眼睛中心Y坐标
        eye_center_y = (left_eye[1] + right_eye[1]) / 2
        
        # 计算眼距（用于归一化）
        eye_distance = abs(right_eye[0] - left_eye[0])
        if eye_distance < 10:  # 眼距太小，可能是检测异常
            return 0.0
        
        # 鼻子相对眼睛中心的垂直偏移
        # 正值=鼻子在眼睛下方=低头
        nose_offset_y = nose[1] - eye_center_y
        
        # 归一化：相对于眼距
        normalized_offset = nose_offset_y / eye_distance
        
        # 限制异常值：正常范围约 0.1 ~ 1.2
        normalized_offset = max(-0.5, min(1.5, normalized_offset))
        
        # 转换为角度估算
        # - normalized_offset < 0.3: 抬头 (< 0°)
        # - normalized_offset ≈ 0.4: 正视 (≈ 0°)
        # - normalized_offset > 0.6: 低头 (> 20°)
        pitch = (normalized_offset - 0.4) * 90
        
        return pitch
    
    @staticmethod
    def calculate_body_lean_angle(keypoints: np.ndarray) -> float:
        """
        计算身体前倾角度
        
        Args:
            keypoints: 关键点数组 (17, 3)
            
        Returns:
            前倾角度（度）
        """
        shoulder_center = GeometryCalculator.get_shoulder_center(keypoints)
        hip_center = GeometryCalculator.get_hip_center(keypoints)
        
        dx = shoulder_center[0] - hip_center[0]
        dy = shoulder_center[1] - hip_center[1]
        
        return np.abs(np.degrees(np.arctan2(dx, dy)))
    
    @staticmethod
    def calculate_angle_change(current: float, previous: float) -> float:
        """
        计算角度变化量
        
        Args:
            current: 当前角度
            previous: 之前角度
            
        Returns:
            角度变化量
        """
        return np.abs(current - previous)
    
    # ==================== 注意力评分计算 ====================
    
    @staticmethod
    def calculate_eye_symmetry_score(
        keypoints: np.ndarray,
        ratio: float = 0.5
    ) -> float:
        """
        计算眼睛对称性评分
        
        Args:
            keypoints: 关键点数组 (17, 3)
            ratio: 眼睛评分系数
            
        Returns:
            眼睛对称性评分 (0-1)
        """
        left_shoulder = keypoints[const.LEFT_SHOULDER]
        right_shoulder = keypoints[const.RIGHT_SHOULDER]
        left_eye = keypoints[const.LEFT_EYE]
        right_eye = keypoints[const.RIGHT_EYE]
        
        if left_eye[0] == 0 or right_eye[0] == 0 or left_shoulder[0] == 0:
            return 0.0
        
        shoulder_dist = np.abs(right_shoulder[0] - left_shoulder[0])
        eye_dist = np.abs(right_eye[0] - left_eye[0])
        
        if shoulder_dist > 0:
            return min(1.0, eye_dist / (shoulder_dist * ratio))
        return 0.0
    
    @staticmethod
    def calculate_nose_offset_score(
        keypoints: np.ndarray,
        ratio: float = 0.6
    ) -> float:
        """
        计算鼻子偏移评分
        
        Args:
            keypoints: 关键点数组 (17, 3)
            ratio: 鼻子偏移评分系数
            
        Returns:
            鼻子偏移评分 (0-1)
        """
        nose = keypoints[const.NOSE]
        eye_center = GeometryCalculator.get_eye_center(keypoints)
        eye_dist = GeometryCalculator.calculate_eye_distance(keypoints)
        
        if eye_center is None or eye_dist == 0:
            return 0.0
        
        nose_offset = np.abs(nose[0] - eye_center[0])
        return max(0.0, 1.0 - (nose_offset / (eye_dist * ratio)))
    
    @staticmethod
    def calculate_ear_based_score(
        keypoints: np.ndarray,
        shoulder_width: float,
        ratio: float = 0.4
    ) -> float:
        """
        基于耳朵的评分（用于眼睛被遮挡时）
        
        Args:
            keypoints: 关键点数组 (17, 3)
            shoulder_width: 肩宽
            ratio: 耳朵评分系数
            
        Returns:
            耳朵评分 (0-1)
        """
        ear_dist = GeometryCalculator.calculate_ear_distance(keypoints)
        
        if shoulder_width > 0:
            return min(1.0, ear_dist / (shoulder_width * ratio))
        return 0.0
    
    @staticmethod
    def calculate_geometry_score(
        keypoints: np.ndarray,
        eye_ratio: float = 0.5,
        nose_ratio: float = 0.6
    ) -> Tuple[float, float, float]:
        """
        计算综合几何评分
        
        Args:
            keypoints: 关键点数组 (17, 3)
            eye_ratio: 眼睛对称性评分系数
            nose_ratio: 鼻子偏移评分系数
            
        Returns:
            (eye_score, nose_score, geometry_score)
        """
        eye_score = GeometryCalculator.calculate_eye_symmetry_score(keypoints, eye_ratio)
        nose_score = GeometryCalculator.calculate_nose_offset_score(keypoints, nose_ratio)
        
        # 综合几何评分：眼睛权重60%，鼻子权重40%
        geometry_score = 0.6 * eye_score + 0.4 * nose_score
        
        return eye_score, nose_score, geometry_score
    
    # ==================== 遮挡处理评分 ====================
    
    @staticmethod
    def calculate_occlusion_adaptive_score(
        keypoints: np.ndarray,
        eye_ratio: float = 0.5,
        nose_ratio: float = 0.6
    ) -> float:
        """
        自适应遮挡处理的注意力评分
        
        当眼睛置信度较低时，使用耳朵代替
        
        Args:
            keypoints: 关键点数组 (17, 3)
            eye_ratio: 眼睛对称性评分系数
            nose_ratio: 鼻子偏移评分系数
            
        Returns:
            注意力评分 (0-1)
        """
        left_eye = keypoints[const.LEFT_EYE]
        right_eye = keypoints[const.RIGHT_EYE]
        left_ear = keypoints[const.LEFT_EAR]
        right_ear = keypoints[const.RIGHT_EAR]
        left_shoulder = keypoints[const.LEFT_SHOULDER]
        right_shoulder = keypoints[const.RIGHT_SHOULDER]
        
        # 检查有效性
        if left_eye[0] == 0 or right_eye[0] == 0 or left_shoulder[0] == 0:
            return 0.2
        
        shoulder_width = GeometryCalculator.calculate_shoulder_width(keypoints)
        
        # 眼睛置信度检查
        if left_eye[2] < 0.5 or right_eye[2] < 0.5:
            if left_ear[2] > 0.3 and right_ear[2] > 0.3:
                ear_score = GeometryCalculator.calculate_ear_based_score(
                    keypoints, shoulder_width, 0.4
                )
                return ear_score * 0.8
            else:
                return 0.2
        else:
            # 使用几何评分
            _, _, geometry_score = GeometryCalculator.calculate_geometry_score(
                keypoints, eye_ratio, nose_ratio
            )
            return geometry_score
    
    # ==================== 历史数据分析 ====================
    
    @staticmethod
    def calculate_torso_velocity(
        current: Tuple[float, float],
        previous: Tuple[float, float],
        dt: float,
        scale: float
    ) -> float:
        """
        计算躯干移动速度
        
        Args:
            current: 当前躯干位置
            previous: 之前躯干位置
            dt: 时间间隔
            scale: 尺度因子
            
        Returns:
            归一化速度 (0-1)
        """
        if dt <= 0:
            return 0.0
        
        dx = current[0] - previous[0]
        dy = current[1] - previous[1]
        displacement = np.sqrt(dx**2 + dy**2)
        velocity = displacement / dt
        
        return min(velocity / (scale + 1e-6) / const.NORMALIZATION_PARAMS["torso_velocity_scale"], 1.0)
    
    @staticmethod
    def calculate_body_sway(keypoints_history: list) -> float:
        """
        计算身体晃动程度
        
        Args:
            keypoints_history: 关键点历史列表
            
        Returns:
            晃动程度 (0-1)
        """
        if len(keypoints_history) < 2:
            return 0.0
        
        torso_centers = []
        for kps in keypoints_history:
            if kps is not None:
                torso_center = GeometryCalculator.get_torso_center(kps)
                torso_centers.append(torso_center[:2])
        
        if len(torso_centers) < 2:
            return 0.0
        
        torso_centers = np.array(torso_centers)
        std_x = np.std(torso_centers[:, 0])
        std_y = np.std(torso_centers[:, 1])
        sway = np.sqrt(std_x**2 + std_y**2)
        
        return min(sway / const.NORMALIZATION_PARAMS["body_sway_scale"], 1.0)
    
    @staticmethod
    def calculate_gesture_activity(
        wrist_positions: list,
        shoulder_center: np.ndarray,
        scale: float
    ) -> float:
        """
        计算手势活跃度
        
        Args:
            wrist_positions: 手腕位置历史 [(left, right), ...]
            shoulder_center: 肩部中心
            scale: 尺度因子
            
        Returns:
            手势活跃度 (0-1)
        """
        if len(wrist_positions) < 2:
            return 0.0
        
        relative_positions = []
        for (lw, rw) in wrist_positions:
            if lw[0] > 0:
                rel_l = ((lw[0] - shoulder_center[0]) / (scale + 1e-6),
                        (lw[1] - shoulder_center[1]) / (scale + 1e-6))
                relative_positions.append(rel_l)
            if rw[0] > 0:
                rel_r = ((rw[0] - shoulder_center[0]) / (scale + 1e-6),
                        (rw[1] - shoulder_center[1]) / (scale + 1e-6))
                relative_positions.append(rel_r)
        
        if len(relative_positions) < 2:
            return 0.0
        
        relative_positions = np.array(relative_positions)
        variance = np.var(relative_positions, axis=0)
        activity = np.sqrt(variance[0]**2 + variance[1]**2)
        
        return min(activity / const.NORMALIZATION_PARAMS["gesture_activity_scale"], 1.0)
    
    @staticmethod
    def calculate_wrist_acceleration(
        wrist_positions: list,
        dt: float = 0.05
    ) -> float:
        """
        计算手腕加速度
        
        Args:
            wrist_positions: 手腕位置历史 [(left, right), ...]
            dt: 时间间隔
            
        Returns:
            加速度 (0-1)
        """
        if len(wrist_positions) < 3:
            return 0.0
        
        positions = []
        for (lw, rw) in list(wrist_positions)[-5:]:
            center_x = 0
            center_y = 0
            count = 0
            if lw[0] > 0:
                center_x += lw[0]
                center_y += lw[1]
                count += 1
            if rw[0] > 0:
                center_x += rw[0]
                center_y += rw[1]
                count += 1
            if count > 0:
                positions.append((center_x / count, center_y / count))
        
        if len(positions) < 3:
            return 0.0
        
        velocities = []
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            v = np.sqrt(dx**2 + dy**2) / (dt + 1e-6)
            velocities.append(v)
        
        if len(velocities) < 1:
            return 0.0
        
        accelerations = []
        for i in range(1, len(velocities)):
            acc = abs(velocities[i] - velocities[i-1])
            accelerations.append(acc)
        
        if not accelerations:
            return 0.0
        
        return min(np.mean(accelerations) / const.NORMALIZATION_PARAMS["acceleration_scale"], 1.0)
    
    @staticmethod
    def calculate_posture_instability(keypoints_history: list) -> float:
        """
        计算姿态不稳定度
        
        Args:
            keypoints_history: 关键点历史列表
            
        Returns:
            不稳定度 (0-1)
        """
        if len(keypoints_history) < 3:
            return 0.0
        
        keypoints_list = list(keypoints_history)[-10:]
        all_stabilities = []
        
        for kp_idx in range(17):
            positions = []
            for kps in keypoints_list:
                if kps is not None and kps[kp_idx, 0] != 0:
                    positions.append(kps[kp_idx, :2])
            
            if len(positions) >= 2:
                positions = np.array(positions)
                std = np.std(positions)
                all_stabilities.append(std)
        
        if not all_stabilities:
            return 0.0
        
        return min(np.mean(all_stabilities) / const.NORMALIZATION_PARAMS["instability_scale"], 1.0)
