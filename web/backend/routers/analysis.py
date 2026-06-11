"""
视频分析路由 - 任务创建（文件路径模式）
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
import os
import uuid

from models.database import Task, TaskStatus, get_db_path

router = APIRouter(prefix="/analysis", tags=["视频分析"])


@router.post("/upload")
async def upload_video(data: dict):
    """
    前端传入本地文件路径，创建任务
    返回 task_id，前端用这个ID追踪进度
    """
    video_path = data.get("video_path", "")
    route = data.get("route", "route1")
    vlm_provider = data.get("vlm_provider", "none")
    vlm_trigger = data.get("vlm_trigger", "MODERATE")
    vlm_api_key = data.get("vlm_api_key", "")
    vlm_model = data.get("vlm_model", "")
    ollama_host = data.get("ollama_host", "http://localhost:11434")
    ollama_model = data.get("ollama_model", "qwen3-vl:8b")
    workers = data.get("workers", 4)

    if not video_path:
        raise HTTPException(status_code=400, detail="video_path 不能为空")

    task_id = str(uuid.uuid4())

    engine = create_engine(f"sqlite:///{get_db_path()}")
    with Session(engine) as s:
        task = Task(
            task_id=task_id,
            video_name=os.path.basename(video_path),
            video_path=video_path,
            route=route,
            vlm_provider=vlm_provider,
            vlm_trigger=vlm_trigger,
            vlm_api_key=vlm_api_key,
            vlm_model=vlm_model,
            ollama_host=ollama_host,
            ollama_model=ollama_model,
            workers=workers,
            status=TaskStatus.PENDING,
        )
        s.add(task)
        s.commit()
        s.refresh(task)

    return {
        "task_id": task_id,
        "video_name": os.path.basename(video_path),
        "video_path": video_path,
        "status": "pending",
        "message": "任务已创建，本地client执行分析",
    }


@router.post("/tasks/{task_id}/result")
async def upload_result(task_id: str, result_data: dict):
    """
    本地client完成分析后，上传结果JSON
    """
    import json
    engine = create_engine(f"sqlite:///{get_db_path()}")
    with Session(engine) as s:
        task = s.execute(select(Task).where(Task.task_id == task_id)).scalar_one_or_none()
        if not task:
            raise HTTPException(status_code=404, detail="任务不存在")
        
        task.status = TaskStatus.COMPLETED
        task.progress = 100.0
        task.progress_message = "分析完成"
        task.result_json = json.dumps(result_data, ensure_ascii=False)
        
        from datetime import datetime
        task.completed_at = datetime.now()
        
        s.commit()
        
        return {"status": "updated", "task_id": task_id}


@router.get("/tasks/{task_id}/status")
def get_task_status(task_id: str):
    """获取任务状态（供前端轮询）"""
    engine = create_engine(f"sqlite:///{get_db_path()}")
    with Session(engine) as s:
        task = s.execute(select(Task).where(Task.task_id == task_id)).scalar_one_or_none()
        if not task:
            raise HTTPException(status_code=404, detail="任务不存在")
        return task.to_dict()


@router.get("/tasks/{task_id}/result")
def get_task_result(task_id: str):
    """获取分析结果"""
    engine = create_engine(f"sqlite:///{get_db_path()}")
    with Session(engine) as s:
        task = s.execute(select(Task).where(Task.task_id == task_id)).scalar_one_or_none()
        if not task:
            raise HTTPException(status_code=404, detail="任务不存在")
        
        if task.status != TaskStatus.COMPLETED:
            return {"status": task.status.value if isinstance(task.status, TaskStatus) else task.status,
                    "result": None,
                    "progress": task.progress,
                    "message": task.progress_message or "任务进行中"}
        
        import json
        result = json.loads(task.result_json) if task.result_json else None
        return {"status": "completed", "result": result}