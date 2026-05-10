"""
VLM模块
支持多种视觉语言模型提供者
"""
from .base_vlm import BaseVLM
from .mock_vlm import MockVLM
from .ollama_vl import OllamaVL
from .siliconflow_vl import SiliconFlowVL

__all__ = ["BaseVLM", "MockVLM", "OllamaVL", "SiliconFlowVL"]


def create_vlm(provider: str, **kwargs):
    """
    创建VLM实例
    
    Args:
        provider: 提供者名称 (mock, ollama, siliconflow)
        **kwargs: 其他参数
        
    Returns:
        VLM实例
        
    Examples:
        >>> vlm = create_vlm('ollama', model='qwen3-vl:8b')
        >>> vlm = create_vlm('siliconflow', api_key='sk-xxx')
    """
    if provider == "mock":
        return MockVLM()
    
    elif provider == "ollama":
        return OllamaVL(
            host=kwargs.get("host", "http://localhost:11434"),
            model=kwargs.get("model", "qwen3-vl:8b")
        )
    
    elif provider == "siliconflow":
        return SiliconFlowVL(
            api_key=kwargs.get("api_key"),
            model=kwargs.get("model", "Qwen/Qwen3-VL-32B-Instruct")
        )
    
    else:
        raise ValueError(f"未知VLM提供者: {provider}，可选: mock, ollama, siliconflow")
