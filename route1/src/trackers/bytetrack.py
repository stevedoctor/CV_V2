"""
ByteTrack多目标跟踪模块
封装ultralytics内置的ByteTrack跟踪功能，确保跨帧ID一致性

关键技术点：
1. 使用model.track()替代model()启用跟踪
2. persist=True保持跟踪状态跨帧
3. 从result返回的boxes.id获取track_id
"""
from ultralytics import YOLO
import numpy as np
from typing import List, Tuple, Optional, Generator
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ByteTrackManager:
    """
    ByteTrack跟踪管理器
    
    封装YOLO-Pose的跟踪功能，提供：
    - 单帧跟踪接口
    - 视频流跟踪接口
    - track_id一致性保证
    """
    
    def __init__(self, 
                 model_path: str = "yolo11n-pose.pt",
                 conf_threshold: float = 0.5,
                 iou_threshold: float = 0.5,
                 verbose: bool = False):
        """
        初始化ByteTrack管理器
        
        Args:
            model_path: YOLO模型路径
            conf_threshold: 置信度阈值
            iou_threshold: IOU阈值
            verbose: 是否显示详细日志
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.verbose = verbose
        self.frame_count = 0
    
    def track_frame(self, frame: np.ndarray) -> List[Tuple[int, np.ndarray]]:
        """
        单帧跟踪
        
        使用persist=True保持跨帧状态（关键！）
        
        Args:
            frame: 输入帧 (BGR格式)
            
        Returns:
            [(track_id, keypoints), ...]
            - track_id: 跨帧一致的跟踪ID
            - keypoints: (17, 3) 关键点数组 [x, y, confidence]
        """
        results = self.model.track(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=self.verbose
        )
        
        tracked_persons = []
        
        if results and len(results) > 0:
            result = results[0]
            
            if result.keypoints is not None and result.boxes is not None:
                keypoints_data = result.keypoints.data
                
                if result.boxes.id is not None:
                    track_ids = result.boxes.id.cpu().numpy()
                    
                    for idx, track_id in enumerate(track_ids):
                        if idx < len(keypoints_data):
                            kps = keypoints_data[idx].cpu().numpy()
                            
                            if kps.shape[1] == 2:
                                kps = np.hstack([kps, np.ones((kps.shape[0], 1))])
                            
                            tracked_persons.append((int(track_id), kps))
        
        self.frame_count += 1
        return tracked_persons
    
    def track_video(self, 
                    video_path: str) -> Generator[Tuple[np.ndarray, List[Tuple[int, np.ndarray]]], None, None]:
        """
        视频流跟踪
        
        Args:
            video_path: 视频路径
            
        Yields:
            (frame, [(track_id, keypoints), ...])
        """
        results = self.model.track(
            source=video_path,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            persist=True,
            tracker="bytetrack.yaml",
            stream=True,
            verbose=self.verbose
        )
        
        for result in results:
            frame = result.orig_img.copy()
            tracked_persons = []
            
            if result.keypoints is not None and result.boxes is not None:
                keypoints_data = result.keypoints.data
                
                if result.boxes.id is not None:
                    track_ids = result.boxes.id.cpu().numpy()
                    
                    for idx, track_id in enumerate(track_ids):
                        if idx < len(keypoints_data):
                            kps = keypoints_data[idx].cpu().numpy()
                            
                            if kps.shape[1] == 2:
                                kps = np.hstack([kps, np.ones((kps.shape[0], 1))])
                            
                            tracked_persons.append((int(track_id), kps))
            
            yield frame, tracked_persons
    
    def reset(self):
        """重置跟踪状态"""
        self.model.predictor = None
        self.frame_count = 0


def create_tracker(model_path: str = "yolo11n-pose.pt", **kwargs) -> ByteTrackManager:
    """
    创建跟踪器实例
    
    Args:
        model_path: 模型路径
        **kwargs: 其他参数
        
    Returns:
        ByteTrackManager实例
    """
    return ByteTrackManager(model_path=model_path, **kwargs)
