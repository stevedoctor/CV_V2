"""
系统配置设置
"""
from dataclasses import dataclass, field
from typing import Optional, Dict
import os


@dataclass
class Settings:
    """系统配置类"""
    
    # ==================== 模型配置 ====================
    model_path: str = "yolo11n-pose.pt"
    model_conf: float = 0.5
    
    # ==================== 视频配置 ====================
    video_path: str = "input/meeting_attention_video.mp4"
    output_path: str = "output/result.mp4"
    save_video: bool = True
    fps: int = 20
    
    # ==================== 目录配置 ====================
    input_dir: str = "input"
    output_dir: str = "output"
    audit_dir: str = "audit_videos"
    
    # ==================== 追踪配置 ====================
    history_size: int = 10
    torso_history_size: int = 30
    
    # ==================== RFAC配置 ====================
    forward_threshold: float = 0.4
    normal_threshold: float = 0.3
    
    # ==================== VLM审计配置 ====================
    # 是否启用VLM审计
    vlm_enabled: bool = False
    
    # VLM提供者: mock, ollama, siliconflow
    vlm_provider: str = "ollama"
    
    # 触发VLM审计的等级: MILD, MODERATE, SEVERE
    vlm_trigger_level: str = "MODERATE"
    
    # 最大采样帧数
    vlm_max_frames: int = 4
    
    # ==================== Ollama配置（本地） ====================
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen3-vl:8b"
    
    # ==================== 硅基流动配置（云端） ====================
    siliconflow_model: str = "Qwen/Qwen3-VL-32B-Instruct"
    # API Key从环境变量读取: SILICONFLOW_API_KEY
    
    # ==================== 旧审计配置（兼容） ====================
    audit_enabled: bool = True
    audit_provider: str = "mock"
    audit_threshold: float = 0.4
    
    # ==================== 几何评分配置 ====================
    eye_score_ratio: float = 0.5
    nose_score_ratio: float = 0.6
    
    # ==================== 注意力等级阈值 ====================
    level_thresholds: Dict = field(default_factory=lambda: {
        "NORMAL": 0.3,
        "MILD": 0.5,
        "MODERATE": 0.7
    })
    
    def __post_init__(self):
        """初始化后处理 - 确保目录存在"""
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.audit_dir, exist_ok=True)


# 全局配置实例
settings = Settings()
