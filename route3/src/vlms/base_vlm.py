"""
VLM基类模块

定义视觉语言模型的抽象接口
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np


class BaseVLM(ABC):
    """
    视觉语言模型抽象基类
    
    所有VLM实现（通义千问VL、GPT-4V、Ollama等）都需要继承此类
    """
    
    @abstractmethod
    def analyze_frames(self, 
                       frames: List[np.ndarray],
                       prompt: str) -> Dict[str, Any]:
        """
        分析多帧图像
        
        Args:
            frames: 图像帧列表 (BGR格式)
            prompt: 分析提示词
            
        Returns:
            分析结果字典，包含：
            - apathy_score: 注意力涣散评分 (0-1)
            - fatigue_score: 疲劳评分 (0-1)
            - rushing_score: 匆忙评分 (0-1)
            - frustration_score: 情绪评分 (0-1)
            - overall_score: 综合评分 (0-1)
            - attention_level: 注意力等级
            - reasoning: 分析推理过程
            - suggestions: 干预建议
        """
        pass
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """
        获取提供者名称
        
        Returns:
            提供者名称字符串
        """
        pass
    
    def supports_streaming(self) -> bool:
        """
        是否支持流式输出
        
        Returns:
            是否支持
        """
        return False
    
    def get_max_frames(self) -> int:
        """
        获取支持的最大帧数
        
        Returns:
            最大帧数
        """
        return 10
