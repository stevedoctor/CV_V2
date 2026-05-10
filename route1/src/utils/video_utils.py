"""
视频工具模块
处理视频读写和基本操作
"""
import cv2
import numpy as np
from typing import Optional, Tuple
import os


class VideoHelper:
    """视频帮助类"""
    
    @staticmethod
    def create_writer(output_path: str, fps: int, 
                     frame_size: Tuple[int, int]) -> cv2.VideoWriter:
        """
        创建视频写入器
        
        Args:
            output_path: 输出路径
            fps: 帧率
            frame_size: (宽, 高)
            
        Returns:
            VideoWriter对象
        """
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        return cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    
    @staticmethod
    def get_video_info(video_path: str) -> Optional[dict]:
        """
        获取视频信息
        
        Args:
            video_path: 视频路径
            
        Returns:
            视频信息字典
        """
        if not os.path.exists(video_path):
            return None
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        info = {
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        }
        
        cap.release()
        return info
    
    @staticmethod
    def draw_keypoints(frame: np.ndarray, keypoints: np.ndarray, 
                      color: Tuple[int, int, int] = (0, 255, 0),
                      thickness: int = 2) -> np.ndarray:
        """
        绘制关键点
        
        Args:
            frame: 输入帧
            keypoints: 关键点数组 (17, 3)
            color: 绘制颜色
            thickness: 线粗细
            
        Returns:
            绘制后的帧
        """
        # 定义骨骼连接
        connections = [
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # 上身
            (5, 11), (6, 12), (11, 12),  # 躯干
            (11, 13), (13, 15), (12, 14), (14, 16)  # 下身
        ]
        
        # 绘制连接
        for start, end in connections:
            if keypoints[start][0] > 0 and keypoints[end][0] > 0:
                pt1 = (int(keypoints[start][0]), int(keypoints[start][1]))
                pt2 = (int(keypoints[end][0]), int(keypoints[end][1]))
                cv2.line(frame, pt1, pt2, color, thickness)
        
        # 绘制关键点
        for kp in keypoints:
            if kp[0] > 0 and kp[1] > 0:
                cv2.circle(frame, (int(kp[0]), int(kp[1])), 3, color, -1)
        
        return frame
