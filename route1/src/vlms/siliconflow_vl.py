"""
硅基流动VLM模块

使用SiliconFlow API调用云端VLM模型
支持模型：Qwen/Qwen3-VL-8B-Instruct, Qwen/Qwen3-VL-32B-Instruct 等
"""
import requests
import base64
import cv2
import numpy as np
import json
import re
import os
from typing import List, Dict, Any, Optional
from .base_vlm import BaseVLM


class SiliconFlowVL(BaseVLM):
    """
    硅基流动VLM实现
    
    使用SiliconFlow API调用云端视觉语言模型
    """
    
    def __init__(self, 
                 api_key: str = None,
                 model: str = "Qwen/Qwen3-VL-32B-Instruct",
                 base_url: str = "https://api.siliconflow.cn/v1"):
        """
        初始化硅基流动VLM
        
        Args:
            api_key: API密钥，None则从环境变量SILICONFLOW_API_KEY读取
            model: 模型名称
            base_url: API基础URL
        """
        self.api_key = api_key or os.environ.get("SILICONFLOW_API_KEY", "")
        self.model = model
        self.base_url = base_url
    
    def analyze_frames(self, 
                       frames: List[np.ndarray],
                       prompt: str) -> Dict[str, Any]:
        """
        使用硅基流动API分析帧
        
        Args:
            frames: 图像帧列表
            prompt: 分析提示词
            
        Returns:
            分析结果字典
        """
        if not self.api_key:
            return {
                "error": "未设置API Key，请设置环境变量 SILICONFLOW_API_KEY 或传入 api_key 参数",
                "anomaly_type": "UNKNOWN",
                "confidence": 0.0,
                "reasoning": "API Key未配置"
            }
        
        # 构建消息内容
        content = []
        
        # 添加图像（base64编码）
        for i, frame in enumerate(frames[:self.get_max_frames()]):
            b64 = self._encode_frame(frame)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}"
                }
            })
        
        # 添加文本提示
        content.append({
            "type": "text",
            "text": prompt
        })
        
        # 构建请求
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ],
            "max_tokens": 1024,
            "temperature": 0.1
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                return self._parse_response(result)
            else:
                error_msg = response.text
                try:
                    error_data = response.json()
                    error_msg = error_data.get("error", {}).get("message", error_msg)
                except:
                    pass
                
                return {
                    "error": f"API错误 ({response.status_code}): {error_msg}",
                    "anomaly_type": "UNKNOWN",
                    "confidence": 0.0,
                    "reasoning": f"API调用失败: {response.status_code}"
                }
                
        except requests.exceptions.Timeout:
            return {
                "error": "请求超时",
                "anomaly_type": "UNKNOWN",
                "confidence": 0.0,
                "reasoning": "API请求超时，请重试"
            }
        except requests.exceptions.ConnectionError:
            return {
                "error": "网络连接失败",
                "anomaly_type": "UNKNOWN",
                "confidence": 0.0,
                "reasoning": "无法连接到硅基流动API"
            }
        except Exception as e:
            return {
                "error": str(e),
                "anomaly_type": "UNKNOWN",
                "confidence": 0.0,
                "reasoning": f"发生错误: {e}"
            }
    
    def _encode_frame(self, frame: np.ndarray) -> str:
        """将帧编码为base64"""
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return base64.b64encode(buffer).decode('utf-8')
    
    def _parse_response(self, response: Dict) -> Dict[str, Any]:
        """解析API响应"""
        try:
            # 提取回复文本
            text = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            if not text:
                return {
                    "error": "响应内容为空",
                    "anomaly_type": "UNKNOWN",
                    "confidence": 0.0,
                    "reasoning": "API返回空响应"
                }
            
            # 尝试提取JSON
            result = self._extract_json(text)
            
            if result:
                # 确保必要字段
                result.setdefault("anomaly_type", "UNKNOWN")
                result.setdefault("confidence", 0.5)
                result.setdefault("reasoning", text[:200])
                result.setdefault("suggestions", "")
                result.setdefault("severity", "MODERATE")
                return result
            
            # 无法解析JSON，返回原始文本
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
                "reasoning": f"响应解析失败: {e}"
            }
    
    def _extract_json(self, text: str) -> Optional[Dict]:
        """从文本中提取JSON（支持think标签、嵌套JSON）"""
        # 1. 移除 <think>...</think> 标签
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
        return f"siliconflow ({self.model.split('/')[-1]})"
    
    def get_max_frames(self) -> int:
        return 4  # 建议不超过4张
    
    def check_api_key(self) -> bool:
        """检查API Key是否有效"""
        if not self.api_key:
            return False
        
        try:
            # 发送简单请求验证
            response = requests.get(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=10
            )
            return response.status_code == 200
        except:
            return False
