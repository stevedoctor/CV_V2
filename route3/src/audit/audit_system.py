"""
审计系统模块
支持多种大模型审计提供者，复用vlms模块
"""
import cv2
import numpy as np
import time
from typing import Dict, List, Optional
from collections import deque
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Settings
from vlms import create_vlm


# VLM Prompt模板
AUDIT_PROMPT = """你是一位专业的会议行为分析师。请分析这组会议视频帧中该人员的注意力状态。

## 背景信息
规则引擎初步判定该人员状态为【{level}】，综合评分：{score:.2f}

## 分析要求
请确认或修正这个判定，并从以下维度分析：

1. **异常类型判断**: FATIGUE（疲劳）/ APATHY（注意力涣散）/ RUSHING（匆忙）/ FRUSTRATION（情绪异常）
2. **置信度**: 0.0-1.0，表示判断的可信程度
3. **详细原因**: 描述观察到的具体行为特征
4. **干预建议**: 针对性的处理建议

## 输出格式
重要：只输出JSON，不要输出任何其他文字，不要输出思考过程。
请返回JSON格式（不要包含```标记）：
{{"anomaly_type": "FATIGUE或APATHY或RUSHING或FRUSTRATION或NONE", "confidence": 0.85, "reasoning": "详细分析原因", "suggestions": "干预建议", "severity": "NORMAL或MILD或MODERATE或SEVERE"}}"""


class AuditResult:
    """审计结果"""
    def __init__(self):
        self.person_id: int = 0
        self.trigger_score: float = 0.0
        self.trigger_level: str = ""
        self.anomaly_type: str = "UNKNOWN"
        self.confidence: float = 0.0
        self.reasoning: str = ""
        self.suggestions: str = ""
        self.severity: str = "UNKNOWN"
        self.provider: str = ""
        self.timestamp: float = 0.0
        self.video_path: str = ""
        self.error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "person_id": self.person_id,
            "trigger_score": self.trigger_score,
            "trigger_level": self.trigger_level,
            "anomaly_type": self.anomaly_type,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "suggestions": self.suggestions,
            "severity": self.severity,
            "provider": self.provider,
            "timestamp": self.timestamp,
            "video_path": self.video_path,
            "error": self.error
        }


class AuditSystem:
    """大模型审计系统"""
    
    def __init__(self, 
                 provider: str = "ollama",
                 trigger_level: str = "MODERATE",
                 max_frames: int = 4,
                 host: str = None,
                 model: str = None,
                 api_key: str = None):
        """
        初始化审计系统
        
        Args:
            provider: VLM提供者 (mock, ollama, siliconflow)
            trigger_level: 触发审计的等级
            max_frames: 最大采样帧数
            host: Ollama服务地址
            model: 模型名称
            api_key: API密钥（用于siliconflow等云端提供者）
        """
        self.settings = Settings()
        self.provider = provider
        self.trigger_level = trigger_level
        self.max_frames = max_frames
        
        # 帧缓冲区（按人员ID分组）
        self.frame_buffer: Dict[int, deque] = {}
        self.buffer_size = 30
        
        # 审计结果
        self.audit_results: Dict[int, List[AuditResult]] = {}
        self.audit_count = 0
        self.audit_dir = self.settings.audit_dir
        
        # 创建VLM实例
        vlm_kwargs = {}
        if host:
            vlm_kwargs["host"] = host
        if model:
            vlm_kwargs["model"] = model
        if api_key:
            vlm_kwargs["api_key"] = api_key
        
        self.vlm = create_vlm(provider, **vlm_kwargs)
        
        # 创建审计目录
        os.makedirs(self.audit_dir, exist_ok=True)
    
    def add_frame(self, frame: np.ndarray, person_id: int = None):
        """添加帧到缓冲区"""
        key = person_id if person_id is not None else -1
        
        if key not in self.frame_buffer:
            self.frame_buffer[key] = deque(maxlen=self.buffer_size)
        
        self.frame_buffer[key].append(frame.copy())
    
    def should_trigger_audit(self, score: float, level: str) -> bool:
        """判断是否应该触发审计"""
        level_priority = {"NORMAL": 0, "MILD": 1, "MODERATE": 2, "SEVERE": 3}
        trigger_priority = level_priority.get(self.trigger_level, 2)
        current_priority = level_priority.get(level, 0)
        
        return current_priority >= trigger_priority
    
    def trigger_audit(self, 
                      person_id: int, 
                      score: float, 
                      level: str,
                      frame: np.ndarray = None) -> AuditResult:
        """触发审计"""
        result = AuditResult()
        result.person_id = person_id
        result.trigger_score = score
        result.trigger_level = level
        result.provider = self.vlm.get_provider_name()
        result.timestamp = time.time()
        
        # 获取待分析帧
        frames = self._get_audit_frames(person_id, frame)
        
        if not frames:
            result.error = "无可用帧进行分析"
            return result
        
        # 保存审计视频片段
        video_path = self._save_audit_video(person_id, frames)
        result.video_path = video_path
        
        # 构建Prompt
        prompt = AUDIT_PROMPT.format(level=level, score=score)
        
        # 调用VLM
        print(f"🔍 [VLM审计] P{person_id} ({level}, score={score:.2f})...")
        print(f"📡 提供者: {self.vlm.get_provider_name()}")
        
        try:
            vlm_response = self.vlm.analyze_frames(frames, prompt)
            
            if "error" in vlm_response and vlm_response["error"]:
                result.error = vlm_response["error"]
                print(f"❌ 错误: {result.error}")
            else:
                result.anomaly_type = vlm_response.get("anomaly_type", "UNKNOWN")
                result.confidence = vlm_response.get("confidence", 0.5)
                result.reasoning = vlm_response.get("reasoning", "")
                result.suggestions = vlm_response.get("suggestions", "")
                result.severity = vlm_response.get("severity", level)
                
                print(f"✅ 异常类型: {result.anomaly_type}")
                print(f"📊 置信度: {result.confidence:.2f}")
                print(f"📝 原因: {result.reasoning[:100]}...")
                if result.suggestions:
                    print(f"💡 建议: {result.suggestions}")
        
        except Exception as e:
            result.error = str(e)
            print(f"❌ 审计异常: {e}")
        
        # 记录结果
        if person_id not in self.audit_results:
            self.audit_results[person_id] = []
        self.audit_results[person_id].append(result)
        self.audit_count += 1
        
        return result
    
    def _get_audit_frames(self, person_id: int, current_frame: np.ndarray = None) -> List[np.ndarray]:
        """获取审计用的帧"""
        frames = []
        
        # 从缓冲区获取历史帧
        if person_id in self.frame_buffer:
            buffer = list(self.frame_buffer[person_id])
            if buffer:
                # 均匀采样
                step = max(1, len(buffer) // self.max_frames)
                frames = [buffer[i] for i in range(0, len(buffer), step)][:self.max_frames]
        
        # 添加当前帧
        if current_frame is not None:
            frames.append(current_frame)
        
        return frames
    
    def _save_audit_video(self, person_id: int, frames: List[np.ndarray]) -> str:
        """保存审计视频"""
        if not frames:
            return ""
        
        timestamp = int(time.time())
        video_path = f"{self.audit_dir}/audit_p{person_id}_{timestamp}.mp4"
        
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 10, (w, h))
        
        for frame in frames:
            out.write(frame)
        out.release()
        
        return video_path
    
    def get_audit_summary(self) -> Dict:
        """获取审计汇总"""
        return {
            "total_audits": self.audit_count,
            "model_provider": self.provider,
            "trigger_level": self.trigger_level,
            "results": {pid: [r.to_dict() for r in results] 
                       for pid, results in self.audit_results.items()}
        }
    
    def print_summary(self):
        """打印审计汇总"""
        print("\n" + "=" * 60)
        print(f"🤖 VLM审计复核汇总 ({self.provider})")
        print(f"触发等级: {self.trigger_level}")
        print(f"总审计次数: {self.audit_count}")
        print("-" * 60)
        
        for pid, results in self.audit_results.items():
            print(f"\n人员 P{pid}:")
            for i, r in enumerate(results, 1):
                print(f"  #{i} 触发原因: {r.trigger_level} (score={r.trigger_score:.2f})")
                print(f"     异常类型: {r.anomaly_type} (置信度={r.confidence:.2f})")
                if r.reasoning:
                    print(f"     分析: {r.reasoning[:50]}...")
        
        print("=" * 60)


def create_audit_system(provider: str = None, **kwargs) -> AuditSystem:
    """创建审计系统"""
    settings = Settings()
    
    if provider is None:
        provider = settings.vlm_provider if settings.vlm_enabled else "mock"
    
    system = AuditSystem(
        provider=provider,
        trigger_level=kwargs.get("trigger_level", settings.vlm_trigger_level),
        max_frames=kwargs.get("max_frames", settings.vlm_max_frames),
        host=kwargs.get("host", settings.ollama_host),
        model=kwargs.get("model", settings.ollama_model)
    )
    
    print(f"[AuditSystem] 初始化完成，提供者: {provider}")
    
    return system


def create_audit_system_v2(provider: str = None, 
                           trigger_level: str = None,
                           ollama_host: str = None,
                           ollama_model: str = None,
                           siliconflow_model: str = None,
                           vlm_api_key: str = None) -> AuditSystem:
    """
    创建审计系统（v2版本，支持所有VLM参数）
    
    Args:
        provider: VLM提供者 (mock, ollama, siliconflow)
        trigger_level: 触发等级
        ollama_host: Ollama服务地址
        ollama_model: Ollama模型
        siliconflow_model: 硅基流动模型
        vlm_api_key: VLM API Key（用于siliconflow）
        
    Returns:
        AuditSystem实例
    """
    import os
    settings = Settings()
    
    if provider is None:
        provider = settings.vlm_provider if settings.vlm_enabled else "mock"
    
    if trigger_level is None:
        trigger_level = settings.vlm_trigger_level
    
    # 根据提供者获取配置
    vlm_kwargs = {
        "trigger_level": trigger_level,
        "max_frames": settings.vlm_max_frames
    }
    
    if provider == "ollama":
        vlm_kwargs["host"] = ollama_host or settings.ollama_host
        vlm_kwargs["model"] = ollama_model or settings.ollama_model
    
    elif provider == "siliconflow":
        vlm_kwargs["model"] = siliconflow_model or settings.siliconflow_model
        vlm_kwargs["api_key"] = vlm_api_key or os.environ.get("SILICONFLOW_API_KEY", "")
    
    
    system = AuditSystem(provider=provider, **vlm_kwargs)
    
    print(f"[AuditSystem] 初始化完成，提供者: {provider}")
    
    return system
