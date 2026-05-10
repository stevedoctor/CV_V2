"""
视频分析路由 - 上传 + 创建任务
"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
import os
import uuid
import shutil

from models.database import Task, TaskStatus, get_db_path

router = APIRouter(prefix="/analysis", tags=["视频分析"])


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(BASE_DIR, "data", "videos")
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    route: str = Form("route1"),
    vlm_provider: str = Form("none"),
    vlm_trigger: str = Form("MODERATE"),
    vlm_api_key: str = Form(""),
    vlm_model: str = Form(""),
    workers: int = Form(4),
    submitted_by: str = Form("anonymous"),
):
    """
    上传视频文件，创建任务
    
    返回 task_id，前端用这个ID追踪进度
    """
    if not file.filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="仅支持 mp4/avi/mov/mkv")
    
    task_id = str(uuid.uuid4())
    
    safe_name = f"{task_id}_{file.filename}"
    video_path = os.path.join(UPLOAD_DIR, safe_name)
    
    with open(video_path, 'wb') as f:
        shutil.copyfileobj(file.file, f)
    
    file_size = os.path.getsize(video_path)
    
    engine = create_engine(f"sqlite:///{get_db_path()}")
    with Session(engine) as s:
        task = Task(
            task_id=task_id,
            video_name=file.filename,
            video_path=video_path,
            route=route,
            vlm_provider=vlm_provider,
            vlm_trigger=vlm_trigger,
            vlm_api_key=vlm_api_key,
            vlm_model=vlm_model,
            workers=workers,
            status=TaskStatus.PENDING,
            submitted_by=submitted_by,
        )
        s.add(task)
        s.commit()
        s.refresh(task)
    
    return {
        "task_id": task_id,
        "video_name": file.filename,
        "video_path": video_path,
        "file_size": file_size,
        "status": "pending",
        "message": "视频上传成功，任务已创建，等待本地client执行",
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