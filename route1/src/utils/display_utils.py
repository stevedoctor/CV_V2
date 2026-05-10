"""
显示工具模块
处理界面显示和文本生成
"""
import cv2
import numpy as np
from typing import Tuple
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import constants as const


class DisplayHelper:
    """显示帮助类"""
    
    @staticmethod
    def get_level_color(level: str) -> Tuple[int, int, int]:
        """获取等级对应颜色"""
        return const.COLOR_MAP.get(level, (255, 255, 255))
    
    @staticmethod
    def generate_person_text(person_id: int, overall_score: float, attention_level: str) -> str:
        """生成人员信息文本"""
        return f"P{person_id+1}: {attention_level} ({overall_score:.2f})"
    
    @staticmethod
    def generate_detail_text(apathy: float, fatigue: float, 
                           rushing: float, frustration: float) -> str:
        """生成详细指标文本"""
        return f"A:{apathy:.1f} F:{fatigue:.1f} R:{rushing:.1f} Fr:{frustration:.1f}"
    
    @staticmethod
    def put_person_info(frame: np.ndarray, person_id: int, 
                       overall_score: float, attention_level: str,
                       detail_text: str):
        """在帧上绘制人员信息"""
        color = DisplayHelper.get_level_color(attention_level)
        y_base = 60 + person_id * 25
        
        # 主信息
        text = DisplayHelper.generate_person_text(person_id, overall_score, attention_level)
        cv2.putText(frame, text, (20, y_base), 0, 0.5, color, 2)
        
        # 详细指标
        cv2.putText(frame, detail_text, (20, y_base + 20), 0, 0.4, (200, 200, 200), 1)
    
    @staticmethod
    def put_forward_rate(frame: np.ndarray, rate: float):
        """绘制正视率"""
        cv2.putText(frame, f"Forward Rate: {rate:.1f}%", 
                   (20, 30), 0, 0.7, (255, 255, 0), 2)
