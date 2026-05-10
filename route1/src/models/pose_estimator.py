"""
姿态估计模块
使用YOLO进行人体姿态检测
"""
from ultralytics import YOLO
import numpy as np
from typing import List, Tuple, Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Settings


class PoseEstimator:
    """YOLO姿态估计器"""
    
    def __init__(self, model_path: str = None, conf: float = 0.5):
        """
        初始化姿态估计器
        
        Args:
            model_path: 模型路径
            conf: 置信度阈值
        """
        settings = Settings()
        self.model_path = model_path or settings.model_path
        self.conf = conf
        self.model = YOLO(self.model_path)
    
    def detect(self, frame) -> List[np.ndarray]:
        """
        检测单帧中的人员关键点
        
        Args:
            frame: 输入图像
            
        Returns:
            关键点列表，每个元素为(17, 3)的numpy数组
        """
        results = self.model(frame, conf=self.conf, verbose=False)
        
        keypoints_list = []
        if results and len(results) > 0:
            for result in results:
                if result.keypoints is not None:
                    for person in result.keypoints:
                        kps = person.xy[0].cpu().numpy()
                        # 确保有置信度维度
                        if kps.shape[1] == 2:
                            kps = np.hstack([kps, np.ones((kps.shape[0], 1))])
                        keypoints_list.append(kps)
        
        return keypoints_list
    
    def detect_stream(self, video_path: str):
        """
        流式检测视频
        
        Args:
            video_path: 视频路径
            
        Yields:
            (frame, keypoints_list) 元组
        """
        results = self.model(source=video_path, stream=True, conf=self.conf)
        
        for res in results:
            frame = res.orig_img.copy()
            keypoints_list = []
            
            if res.keypoints is not None:
                for person in res.keypoints:
                    kps = person.xy[0].cpu().numpy()
                    if kps.shape[1] == 2:
                        kps = np.hstack([kps, np.ones((kps.shape[0], 1))])
                    keypoints_list.append(kps)
            
            yield frame, keypoints_list
