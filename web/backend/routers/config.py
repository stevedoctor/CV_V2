"""
VLM 配置路由 - 配置增删改查
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import json
import os

router = APIRouter(prefix="/config", tags=["VLM配置"])


CONFIG_FILE = "/home/stve/out/big data/CV_V2/web/backend/data/vlm_config.json"


class VLMConfig(BaseModel):
    provider: str = "none"
    api_key: str = ""
    model: str = ""
    trigger: str = "MODERATE"
    max_workers: int = 4
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen3-vl:8b"


def load_config() -> dict:
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return {
        "provider": "none",
        "api_key": "",
        "model": "Qwen/Qwen3-VL-32B-Instruct",
        "trigger": "MODERATE",
        "max_workers": 4,
        "ollama_host": "http://localhost:11434",
        "ollama_model": "qwen3-vl:8b",
    }


def save_config(data: dict):
    os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
    with open(CONFIG_FILE, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


@router.get("/vlm")
def get_vlm_config():
    """获取当前VLM配置"""
    return load_config()


@router.post("/vlm")
def set_vlm_config(config: VLMConfig):
    """更新VLM配置"""
    data = config.model_dump()
    save_config(data)
    return {"status": "updated", "config": data}


@router.get("/vlm/providers")
def list_providers():
    """列出可用VLM提供者"""
    return {
        "providers": [
            {"id": "none", "name": "无VLM（纯规则引擎）", "models": []},
            {"id": "ollama", "name": "Ollama（本地模型）", "models": ["qwen3-vl:8b", "llava:7b"]},
            {"id": "siliconflow", "name": "硅基流动（云端）", "models": ["Qwen/Qwen3-VL-32B-Instruct"]},
            {"id": "mock", "name": "Mock（测试用）", "models": ["mock"]},
        ]
    }