"""
WebSocket 路由 - 本地client上报进度
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlalchemy import create_engine, select, update
from sqlalchemy.orm import Session
import json
import asyncio
from datetime import datetime

from models.database import Task, TaskStatus, get_db_path

router = APIRouter()


ACTIVE_WEBSOCKETS = {}


@router.websocket("/ws/{task_id}")
async def websocket_progress(websocket: WebSocket, task_id: str):
    """
    本地client连接到 /ws/{task_id} 推送进度
    前端也连这个端点接收进度
    """
    await websocket.accept()
    ACTIVE_WEBSOCKETS[task_id] = websocket
    
    try:
        engine = create_engine(f"sqlite:///{get_db_path()}")
        with Session(engine) as s:
            task = s.execute(select(Task).where(Task.task_id == task_id)).scalar_one_or_none()
            if task:
                task.status = TaskStatus.RUNNING
                task.started_at = datetime.now()
                s.commit()
        
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            
            msg_type = msg.get("type", "progress")
            
            if msg_type == "progress":
                progress = msg.get("progress", 0)
                message = msg.get("message", "")
                
                engine = create_engine(f"sqlite:///{get_db_path()}")
                with Session(engine) as s:
                    task = s.execute(select(Task).where(Task.task_id == task_id)).scalar_one_or_none()
                    if task:
                        task.progress = progress
                        task.progress_message = message
                        s.commit()
                
                await websocket.send_json({
                    "type": "ack",
                    "progress": progress,
                    "message": message,
                })
            
            elif msg_type == "complete":
                result = msg.get("result", {})
                
                engine = create_engine(f"sqlite:///{get_db_path()}")
                with Session(engine) as s:
                    task = s.execute(select(Task).where(Task.task_id == task_id)).scalar_one_or_none()
                    if task:
                        task.status = TaskStatus.COMPLETED
                        task.progress = 100.0
                        task.progress_message = "分析完成"
                        task.result_json = json.dumps(result, ensure_ascii=False)
                        task.completed_at = datetime.now()
                        s.commit()
                
                await websocket.send_json({"type": "done"})
                break
            
            elif msg_type == "error":
                error_msg = msg.get("error", "Unknown error")
                
                engine = create_engine(f"sqlite:///{get_db_path()}")
                with Session(engine) as s:
                    task = s.execute(select(Task).where(Task.task_id == task_id)).scalar_one_or_none()
                    if task:
                        task.status = TaskStatus.FAILED
                        task.error_message = error_msg
                        task.completed_at = datetime.now()
                        s.commit()
                
                await websocket.send_json({"type": "done"})
                break
    
    except WebSocketDisconnect:
        pass
    finally:
        ACTIVE_WEBSOCKETS.pop(task_id, None)


@router.get("/ws/jobs/next")
def get_next_job():
    """
    本地client轮询此端点获取下一个待执行任务
    返回优先级最高且状态为PENDING的任务
    """
    engine = create_engine(f"sqlite:///{get_db_path()}")
    with Session(engine) as s:
        task = s.execute(
            select(Task)
            .where(Task.status == TaskStatus.PENDING)
            .order_by(Task.created_at.asc())
            .limit(1)
        ).scalar_one_or_none()
        
        if not task:
            return {"job": None, "message": "暂无待执行任务"}
        
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()
        s.commit()
        
        job = task.to_dict()
        job["video_path"] = task.video_path
        return {"job": job}