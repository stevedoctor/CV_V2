"""
模拟VLM模块

用于测试和开发，无需API调用
"""
import numpy as np
from typing import List, Dict, Any
import random
from .base_vlm import BaseVLM


class MockVLM(BaseVLM):
    """
    模拟VLM实现
    
    用于无API环境下的测试
    """
    
    def __init__(self, seed: int = 42):
        """
        初始化模拟VLM
        
        Args:
            seed: 随机种子
        """
        random.seed(seed)
    
    def analyze_frames(self, 
                       frames: List[np.ndarray],
                       prompt: str) -> Dict[str, Any]:
        """
        模拟分析帧
        
        返回随机但合理的评分
        """
        # 生成随机评分
        apathy = random.uniform(0.2, 0.5)
        fatigue = random.uniform(0.1, 0.4)
        rushing = random.uniform(0.0, 0.2)
        frustration = random.uniform(0.3, 0.6)
        
        overall = (
            0.55 * apathy +
            0.25 * fatigue +
            0.15 * rushing +
            0.05 * frustration
        )
        
        # 确定等级
        if overall < 0.3:
            level = "NORMAL"
        elif overall < 0.5:
            level = "MILD"
        elif overall < 0.7:
            level = "MODERATE"
        else:
            level = "SEVERE"
        
        return {
            "apathy_score": round(apathy, 3),
            "fatigue_score": round(fatigue, 3),
            "rushing_score": round(rushing, 3),
            "frustration_score": round(frustration, 3),
            "overall_score": round(overall, 3),
            "attention_level": level,
            "reasoning": "【模拟分析】基于视频帧的注意力状态分析",
            "suggestions": self._generate_suggestions(level, apathy, fatigue),
            "frame_count": len(frames)
        }
    
    def _generate_suggestions(self, level: str, apathy: float, fatigue: float) -> str:
        """生成干预建议"""
        suggestions = []
        
        if level == "NORMAL":
            return "状态良好，继续保持专注"
        
        if apathy > 0.4:
            suggestions.append("注意力可能分散，建议集中精力")
        
        if fatigue > 0.3:
            suggestions.append("可能有疲劳迹象，建议适当休息")
        
        if suggestions:
            return "; ".join(suggestions)
        else:
            return "继续观察"
    
    def get_provider_name(self) -> str:
        return "mock"
