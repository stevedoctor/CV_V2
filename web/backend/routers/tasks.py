"""
任务管理路由 - 增删改查
"""
from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import create_engine, select, update
from sqlalchemy.orm import Session
import uuid
import datetime

from models.database import Task, TaskStatus, init_db, get_db_path

router = APIRouter(prefix="/tasks", tags=["任务管理"])


def get_session():
    engine = create_engine(f"sqlite:///{get_db_path()}")
    with Session(engine) as s:
        yield s


@router.post("", status_code=201)
def create_task(data: dict):
    """创建新任务（前端上传视频后调用）"""
    engine = create_engine(f"sqlite:///{get_db_path()}")
    with Session(engine) as s:
        task_id = data.get("task_id") or str(uuid.uuid4())
        
        existing = s.execute(select(Task).where(Task.task_id == task_id)).scalar_one_or_none()
        if existing:
            raise HTTPException(status_code=409, detail="task_id已存在")
        
        task = Task(
            task_id=task_id,
            video_name=data.get("video_name", "unknown.mp4"),
            video_path=data.get("video_path", ""),
            route=data.get("route", "route1"),
            vlm_provider=data.get("vlm_provider", "none"),
            vlm_trigger=data.get("vlm_trigger", "MODERATE"),
            vlm_api_key=data.get("vlm_api_key", ""),
            vlm_model=data.get("vlm_model", ""),
            workers=data.get("workers", 4),
            status=TaskStatus.PENDING,
            submitted_by=data.get("submitted_by", "anonymous"),
        )
        s.add(task)
        s.commit()
        s.refresh(task)
        return task.to_dict()


@router.get("")
def list_tasks(
    status: str = Query(None),
    route: str = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """获取任务列表（支持筛选）"""
    engine = create_engine(f"sqlite:///{get_db_path()}")
    with Session(engine) as s:
        q = select(Task).order_by(Task.created_at.desc())
        
        if status:
            try:
                s_enum = TaskStatus(status)
                q = q.where(Task.status == s_enum)
            except ValueError:
                pass
        
        if route:
            q = q.where(Task.route == route)
        
        total = len(s.execute(q).scalars().all())
        
        tasks = s.execute(q.limit(limit).offset(offset)).scalars().all()
        
        return {
            "total": total,
            "tasks": [t.to_dict() for t in tasks],
            "limit": limit,
            "offset": offset,
        }


@router.get("/{task_id}")
def get_task(task_id: str):
    """获取单个任务详情"""
    engine = create_engine(f"sqlite:///{get_db_path()}")
    with Session(engine) as s:
        task = s.execute(select(Task).where(Task.task_id == task_id)).scalar_one_or_none()
        if not task:
            raise HTTPException(status_code=404, detail="任务不存在")
        return task.to_dict()


@router.delete("/{task_id}")
def delete_task(task_id: str):
    """删除任务"""
    engine = create_engine(f"sqlite:///{get_db_path()}")
    with Session(engine) as s:
        task = s.execute(select(Task).where(Task.task_id == task_id)).scalar_one_or_none()
        if not task:
            raise HTTPException(status_code=404, detail="任务不存在")
        s.delete(task)
        s.commit()
        return {"deleted": task_id}


@router.post("/{task_id}/cancel")
def cancel_task(task_id: str):
    """取消任务"""
    engine = create_engine(f"sqlite:///{get_db_path()}")
    with Session(engine) as s:
        task = s.execute(select(Task).where(Task.task_id == task_id)).scalar_one_or_none()
        if not task:
            raise HTTPException(status_code=404, detail="任务不存在")
        task.status = TaskStatus.FAILED
        task.error_message = "Cancelled by user"
        task.completed_at = datetime.datetime.now()
        s.commit()
        return {"status": "cancelled"}