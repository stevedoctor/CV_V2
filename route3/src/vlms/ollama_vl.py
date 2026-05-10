"""
Ollama本地视觉语言模型模块

使用Ollama运行本地VLM模型（如qwen3-vl、llava等）
"""
import numpy as np
from typing import List, Dict, Any, Optional
import base64
import cv2
import json
import re
import requests
from .base_vlm import BaseVLM


class OllamaVL(BaseVLM):
    """
    Ollama本地VLM实现
    
    支持通过Ollama运行本地视觉模型
    """
    
    def __init__(self, 
                 host: str = "http://localhost:11434",
                 model: str = "qwen3-vl:8b",
                 max_retries: int = 1):
        """
        初始化Ollama VLM
        
        Args:
            host: Ollama服务地址
            model: 模型名称 (qwen3-vl:8b, llava, llava:13b等)
            max_retries: 失败重试次数
        """
        self.host = host
        self.model = model
        self.max_retries = max_retries
    
    def analyze_frames(self, 
                       frames: List[np.ndarray],
                       prompt: str) -> Dict[str, Any]:
        """
        使用Ollama分析帧
        """
        images = []
        for frame in frames[:self.get_max_frames()]:
            b64 = self._encode_frame(frame)
            images.append(b64)
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": images,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 2048
            }
        }
        
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    f"{self.host}/api/generate",
                    json=payload,
                    timeout=180
                )
                
                if response.status_code == 200:
                    result = response.json()
                    parsed = self._parse_response(result)
                    if "error" not in parsed or parsed.get("anomaly_type") != "UNKNOWN" or parsed.get("confidence", 0) > 0:
                        return parsed
                    last_error = parsed.get("error", "解析失败")
                    if attempt < self.max_retries:
                        print(f"  [Ollama] 第{attempt+1}次尝试未获得有效结果，重试中...")
                        continue
                    return parsed
                else:
                    last_error = f"Ollama API错误: {response.status_code}"
                    if attempt < self.max_retries:
                        print(f"  [Ollama] 第{attempt+1}次请求失败({response.status_code})，重试中...")
                        continue
                    return {
                        "error": last_error,
                        "anomaly_type": "UNKNOWN",
                        "confidence": 0.0,
                        "reasoning": f"API调用失败: {response.status_code}"
                    }
                    
            except requests.exceptions.ConnectionError:
                return {
                    "error": "无法连接Ollama服务，请确保Ollama正在运行",
                    "anomaly_type": "UNKNOWN",
                    "confidence": 0.0,
                    "reasoning": "Ollama服务未运行"
                }
            except requests.exceptions.Timeout:
                last_error = "请求超时"
                if attempt < self.max_retries:
                    print(f"  [Ollama] 第{attempt+1}次请求超时，重试中...")
                    continue
                return {
                    "error": "请求超时(180s)",
                    "anomaly_type": "UNKNOWN",
                    "confidence": 0.0,
                    "reasoning": "Ollama响应超时，模型可能正在加载或GPU资源不足"
                }
            except Exception as e:
                last_error = str(e)
                if attempt < self.max_retries:
                    print(f"  [Ollama] 第{attempt+1}次异常: {e}，重试中...")
                    continue
                return {
                    "error": str(e),
                    "anomaly_type": "UNKNOWN",
                    "confidence": 0.0,
                    "reasoning": f"发生错误: {e}"
                }
        
        return {
            "error": last_error or "未知错误",
            "anomaly_type": "UNKNOWN",
            "confidence": 0.0,
            "reasoning": f"重试{self.max_retries}次后仍失败"
        }
    
    def _encode_frame(self, frame: np.ndarray) -> str:
        """将帧编码为base64"""
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return base64.b64encode(buffer).decode('utf-8')
    
    def _parse_response(self, response: Dict) -> Dict[str, Any]:
        """解析Ollama响应"""
        try:
            text = response.get("response", "")
            
            if not text or text.strip() == "":
                return {
                    "anomaly_type": "UNKNOWN",
                    "confidence": 0.0,
                    "reasoning": "Ollama返回空响应",
                    "suggestions": "",
                    "severity": "MODERATE"
                }
            
            result = self._extract_json(text)
            
            if result:
                result.setdefault("anomaly_type", "UNKNOWN")
                result.setdefault("confidence", 0.5)
                result.setdefault("reasoning", text[:200] if text else "")
                result.setdefault("suggestions", "")
                result.setdefault("severity", "MODERATE")
                return result
            
            return {
                "anomaly_type": "UNKNOWN",
                "confidence": 0.3,
                "reasoning": text[:500],
                "suggestions": "",
                "severity": "MODERATE",
                "raw_response": text
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "anomaly_type": "UNKNOWN",
                "confidence": 0.0,
                "reasoning": f"解析错误: {e}"
            }
    
    def _extract_json(self, text: str) -> Optional[Dict]:
        """从文本中提取JSON（支持think标签、嵌套JSON）"""
        # 1. 移除 <think>...</think> 标签（qwen3-vl特有）
        text = re.sub(r'<think>[\s\S]*?</think>', '', text)
        text = text.strip()
        
        # 2. 优先尝试代码块中的JSON
        code_block_patterns = [
            r'```json\s*([\s\S]*?)\s*```',
            r'```\s*([\s\S]*?)\s*```',
        ]
        for pattern in code_block_patterns:
            matches = re.findall(pattern, text)
            for match in reversed(matches):
                try:
                    return json.loads(match.strip())
                except:
                    continue
        
        # 3. 用大括号配对提取（处理嵌套JSON）
        brace_count = 0
        start = -1
        for i, ch in enumerate(text):
            if ch == '{':
                if brace_count == 0:
                    start = i
                brace_count += 1
            elif ch == '}':
                brace_count -= 1
                if brace_count == 0 and start >= 0:
                    candidate = text[start:i+1]
                    try:
                        return json.loads(candidate)
                    except:
                        start = -1
                        continue
        
        # 4. 尝试整个文本
        try:
            return json.loads(text)
        except:
            return None
    
    def get_provider_name(self) -> str:
        return f"ollama ({self.model})"
    
    def get_max_frames(self) -> int:
        return 4
    
    def check_connection(self) -> bool:
        """检查Ollama服务是否可用"""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def list_models(self) -> List[str]:
        """列出可用的模型"""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return [m["name"] for m in data.get("models", [])]
        except:
            pass
        return []
