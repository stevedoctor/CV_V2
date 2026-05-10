"""
规则阈值分级引擎

将RFAC四维度指标映射为0-3级异常等级：
- 0级: 正常
- 1级: 轻微异常
- 2级: 中度异常
- 3级: 重度异常

设计原则：
1. 每个维度独立判定
2. 分级阈值可配置
3. 支持综合评分聚合
"""
from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np


@dataclass
class RuleThresholds:
    """
    规则阈值配置
    
    命名规则：{维度}_{指标}_{等级}
    等级说明：l0(正常) < l1(轻微) < l2(中度) < l3(重度)
    """
    
    # Rushing（匆忙状态）: 躯干速度阈值（归一化值）
    rushing_velocity_l0: float = 0.1
    rushing_velocity_l1: float = 0.3
    rushing_velocity_l2: float = 0.5
    rushing_velocity_l3: float = 0.7
    
    # Fatigue（疲劳状态）: 低头时间占比
    fatigue_head_down_l0: float = 0.2
    fatigue_head_down_l1: float = 0.4
    fatigue_head_down_l2: float = 0.6
    fatigue_head_down_l3: float = 0.8
    
    # Apathy（注意力涣散）: 正视率（越高越好）
    apathy_forward_l0: float = 0.7
    apathy_forward_l1: float = 0.5
    apathy_forward_l2: float = 0.3
    apathy_forward_l3: float = 0.2
    
    # Frustration（情绪失控）: 手势活跃度
    frustration_activity_l0: float = 0.15
    frustration_activity_l1: float = 0.35
    frustration_activity_l2: float = 0.55
    frustration_activity_l3: float = 0.75
    
    # 综合评分阈值
    overall_score_l0: float = 0.3
    overall_score_l1: float = 0.5
    overall_score_l2: float = 0.7


@dataclass
class RuleResult:
    """
    规则判定结果
    """
    rushing_level: int = 0
    fatigue_level: int = 0
    apathy_level: int = 0
    frustration_level: int = 0
    overall_level: int = 0
    
    rushing_value: float = 0.0
    fatigue_value: float = 0.0
    apathy_value: float = 0.0
    frustration_value: float = 0.0
    overall_value: float = 0.0
    
    def __str__(self) -> str:
        level_names = ["正常", "轻微", "中度", "重度"]
        return (f"RuleResult("
                f"Rushing={level_names[self.rushing_level]}, "
                f"Fatigue={level_names[self.fatigue_level]}, "
                f"Apathy={level_names[self.apathy_level]}, "
                f"Frustration={level_names[self.frustration_level]}, "
                f"Overall={level_names[self.overall_level]})")


class RuleEngine:
    """
    规则判定引擎
    
    职责：
    1. 将连续指标值转换为离散异常等级
    2. 提供可配置的阈值
    3. 计算综合异常等级
    """
    
    def __init__(self, thresholds: Optional[RuleThresholds] = None):
        """
        初始化规则引擎
        
        Args:
            thresholds: 阈值配置对象，None则使用默认值
        """
        self.thresholds = thresholds or RuleThresholds()
    
    def evaluate_rushing(self, torso_velocity: float) -> int:
        """
        评估匆忙等级
        
        Args:
            torso_velocity: 躯干速度（归一化值 0-1）
            
        Returns:
            等级 0-3
        """
        t = self.thresholds
        if torso_velocity < t.rushing_velocity_l0:
            return 0
        elif torso_velocity < t.rushing_velocity_l1:
            return 1
        elif torso_velocity < t.rushing_velocity_l2:
            return 2
        else:
            return 3
    
    def evaluate_fatigue(self, head_down_ratio: float) -> int:
        """
        评估疲劳等级
        
        Args:
            head_down_ratio: 低头时间占比（0-1）
            
        Returns:
            等级 0-3
        """
        t = self.thresholds
        if head_down_ratio < t.fatigue_head_down_l0:
            return 0
        elif head_down_ratio < t.fatigue_head_down_l1:
            return 1
        elif head_down_ratio < t.fatigue_head_down_l2:
            return 2
        else:
            return 3
    
    def evaluate_apathy(self, forward_rate: float) -> int:
        """
        评估注意力涣散等级
        
        注意：forward_rate越高越好，所以阈值逻辑相反
        
        Args:
            forward_rate: 正视率（0-1）
            
        Returns:
            等级 0-3
        """
        t = self.thresholds
        if forward_rate >= t.apathy_forward_l0:
            return 0
        elif forward_rate >= t.apathy_forward_l1:
            return 1
        elif forward_rate >= t.apathy_forward_l2:
            return 2
        else:
            return 3
    
    def evaluate_frustration(self, gesture_activity: float) -> int:
        """
        评估情绪失控等级
        
        Args:
            gesture_activity: 手势活跃度（0-1）
            
        Returns:
            等级 0-3
        """
        t = self.thresholds
        if gesture_activity < t.frustration_activity_l0:
            return 0
        elif gesture_activity < t.frustration_activity_l1:
            return 1
        elif gesture_activity < t.frustration_activity_l2:
            return 2
        else:
            return 3
    
    def evaluate_all(self, rfac_score) -> RuleResult:
        """
        综合评估所有维度
        
        Args:
            rfac_score: RFACScore对象（来自processors.rfac_calculator）
            
        Returns:
            RuleResult对象
        """
        result = RuleResult()
        
        # 保存原始值
        result.torso_velocity = rfac_score.torso_velocity
        result.head_down_ratio = rfac_score.head_down_ratio
        result.forward_rate = rfac_score.forward_rate
        result.gesture_activity = rfac_score.gesture_activity
        
        # 计算各维度等级
        result.rushing_level = self.evaluate_rushing(rfac_score.torso_velocity)
        result.fatigue_level = self.evaluate_fatigue(rfac_score.head_down_ratio)
        result.apathy_level = self.evaluate_apathy(rfac_score.forward_rate)
        result.frustration_level = self.evaluate_frustration(rfac_score.gesture_activity)
        
        # 计算综合评分和等级
        result.overall_value = rfac_score.overall_score
        result.overall_level = self._calculate_overall_level(rfac_score.overall_score)
        
        return result
    
    def _calculate_overall_level(self, overall_score: float) -> int:
        """
        从综合评分计算异常等级
        
        Args:
            overall_score: 综合评分（0-1）
            
        Returns:
            等级 0-3
        """
        t = self.thresholds
        if overall_score < t.overall_score_l0:
            return 0
        elif overall_score < t.overall_score_l1:
            return 1
        elif overall_score < t.overall_score_l2:
            return 2
        else:
            return 3
    
    def get_level_description(self, level: int) -> str:
        """获取等级描述"""
        descriptions = {
            0: "正常 - 无需关注",
            1: "轻微 - 建议观察",
            2: "中度 - 需要关注",
            3: "重度 - 建议干预"
        }
        return descriptions.get(level, "未知")
    
    def get_intervention_suggestion(self, result: RuleResult) -> str:
        """
        获取干预建议
        
        Args:
            result: RuleResult对象
            
        Returns:
            干预建议字符串
        """
        if result.overall_level == 0:
            return "状态良好，继续保持"
        
        suggestions = []
        
        if result.fatigue_level >= 2:
            suggestions.append("建议休息片刻，适当活动")
        
        if result.apathy_level >= 2:
            suggestions.append("注意力可能分散，建议调整坐姿或参与互动")
        
        if result.rushing_level >= 2:
            suggestions.append("动作较匆忙，建议放缓节奏")
        
        if result.frustration_level >= 2:
            suggestions.append("情绪可能有波动，建议关注心理状态")
        
        return "; ".join(suggestions) if suggestions else "继续观察"


def create_rule_engine(**threshold_overrides) -> RuleEngine:
    """
    创建规则引擎实例
    
    Args:
        **threshold_overrides: 阈值覆盖参数
        
    Returns:
        RuleEngine实例
    """
    thresholds = RuleThresholds()
    
    for key, value in threshold_overrides.items():
        if hasattr(thresholds, key):
            setattr(thresholds, key, value)
    
    return RuleEngine(thresholds=thresholds)
