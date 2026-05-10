"""
路线1执行器 - 封装 route1/main_rules.py 的分析逻辑
"""
import sys
import os
import cv2
import json
import asyncio
import websockets
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'route1'))

from src.trackers import ByteTrackManager
from src.processors import PersonTracker, RFACCalculator
from src.rules import RuleEngine
from src.models import AttentionScorer


class Route1Executor:
    """
    本地执行路线1分析
    本地client调用 run() 开始分析，通过 on_progress 回调推送进度
    """
    
    def __init__(self, task_config: dict):
        self.video_path = task_config.get("video_path", "")
        self.vlm_provider = task_config.get("vlm_provider", "none")
        self.vlm_trigger = task_config.get("vlm_trigger", "MODERATE")
        self.vlm_api_key = task_config.get("vlm_api_key", "")
        self.vlm_model = task_config.get("vlm_model", "")
        self.workers = task_config.get("workers", 4)
        self.websocket_url = task_config.get("ws_url", "")
        self.on_progress = None
    
    async def run(self):
        """执行完整分析流程"""
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"视频不存在: {self.video_path}")
        
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        tracker = ByteTrackManager(model_path=os.path.join(os.path.dirname(__file__), '..', '..', 'route1', 'yolo11n-pose.pt'), device="0")
        person_tracker = PersonTracker(history_size=10)
        rfac_calc = RFACCalculator()
        rule_engine = RuleEngine()
        scorer = AttentionScorer()
        
        dt = 1.0 / fps if fps > 0 else 1.0 / 24.0
        
        all_results = []
        frame_count = 0
        total_detections = 0
        level_dist = {0: 0, 1: 0, 2: 0, 3: 0}
        
        cap = cv2.VideoCapture(self.video_path)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            tracks = tracker.track_frame(frame)
            
            for track_id, keypoints in tracks:
                score = scorer.calculate_current_score(keypoints)
                person_tracker.update_person(track_id, score, keypoints, dt)
                person_state = person_tracker.get_or_create_person(track_id)
                rfac = rfac_calc.calculate(person_state, dt)
                rule_result = rule_engine.evaluate_all(rfac)
                
                level_dist[rule_result.overall_level] += 1
                total_detections += 1
            
            all_results.append({
                "frame": frame_count,
                "detections": len(tracks),
                "levels": {k: level_dist[k] for k in level_dist},
            })
            
            frame_count += 1
            
            if frame_count % 100 == 0:
                progress = min(100.0, frame_count / total_frames * 100)
                if self.on_progress:
                    await self.on_progress(progress, f"推理中 {frame_count}/{total_frames} 帧")
        
        cap.release()
        
        total = sum(level_dist.values())
        summary = {
            "route": "route1",
            "total_frames": total_frames,
            "total_detections": total_detections,
            "fps": fps,
            "level_distribution": {
                "NORMAL": level_dist[0],
                "MILD": level_dist[1],
                "MODERATE": level_dist[2],
                "SEVERE": level_dist[3],
            },
            "level_percentages": {
                "NORMAL": round(level_dist[0] / max(total, 1) * 100, 1),
                "MILD": round(level_dist[1] / max(total, 1) * 100, 1),
                "MODERATE": round(level_dist[2] / max(total, 1) * 100, 1),
                "SEVERE": round(level_dist[3] / max(total, 1) * 100, 1),
            },
        }
        
        return summary
    
    async def report_progress(self, progress: float, message: str):
        """通过WebSocket上报进度"""
        if self.websocket_url:
            try:
                async with websockets.connect(self.websocket_url) as ws:
                    await ws.send(json.dumps({
                        "type": "progress",
                        "progress": progress,
                        "message": message,
                    }))
            except Exception:
                pass


def execute_route1(task_config: dict) -> dict:
    """同步入口，供 local_client 直接调用"""
    executor = Route1Executor(task_config)
    return asyncio.run(executor.run())