"""
WebSocket 路由 - 本地client上报进度，前端接收进度
"""
from datetime import datetime
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from models.database import Task, TaskStatus, get_db_path

router = APIRouter()

ACTIVE_WEBSOCKETS: dict[str, set[WebSocket]] = {}


def _get_task(session: Session, task_id: str):
    return session.execute(select(Task).where(Task.task_id == task_id)).scalar_one_or_none()


def _engine():
    return create_engine(f"sqlite:///{get_db_path()}")


async def _broadcast(task_id: str, payload: dict):
    sockets = list(ACTIVE_WEBSOCKETS.get(task_id, set()))
    stale = []

    for socket in sockets:
        try:
            await socket.send_json(payload)
        except Exception:
            stale.append(socket)

    for socket in stale:
        ACTIVE_WEBSOCKETS.get(task_id, set()).discard(socket)


@router.websocket("/{task_id}")
@router.websocket("/ws/{task_id}")
async def websocket_progress(websocket: WebSocket, task_id: str):
    """
    本地client连接到 /ws/{task_id} 推送进度。
    前端也连接同一端点接收广播进度。
    """
    await websocket.accept()
    ACTIVE_WEBSOCKETS.setdefault(task_id, set()).add(websocket)

    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            msg_type = msg.get("type", "progress")

            if msg_type == "progress":
                progress = float(msg.get("progress", 0))
                message = msg.get("message", "")

                with Session(_engine()) as s:
                    task = _get_task(s, task_id)
                    if task:
                        task.status = TaskStatus.RUNNING
                        if not task.started_at:
                            task.started_at = datetime.now()
                        task.progress = progress
                        task.progress_message = message
                        s.commit()

                await _broadcast(task_id, {
                    "type": "progress",
                    "progress": progress,
                    "message": message,
                })

            elif msg_type == "complete":
                result = msg.get("result", {})

                with Session(_engine()) as s:
                    task = _get_task(s, task_id)
                    if task:
                        task.status = TaskStatus.COMPLETED
                        task.progress = 100.0
                        task.progress_message = "分析完成"
                        task.result_json = json.dumps(result, ensure_ascii=False)
                        task.completed_at = datetime.now()
                        s.commit()

                await _broadcast(task_id, {
                    "type": "complete",
                    "progress": 100.0,
                    "message": "分析完成",
                    "result": result,
                })
                break

            elif msg_type == "error":
                error_msg = msg.get("error", "Unknown error")

                with Session(_engine()) as s:
                    task = _get_task(s, task_id)
                    if task:
                        task.status = TaskStatus.FAILED
                        task.error_message = error_msg
                        task.progress_message = error_msg
                        task.completed_at = datetime.now()
                        s.commit()

                await _broadcast(task_id, {
                    "type": "error",
                    "message": error_msg,
                })
                break

            else:
                await websocket.send_json({"type": "error", "message": f"未知消息类型: {msg_type}"})

    except WebSocketDisconnect:
        pass
    finally:
        ACTIVE_WEBSOCKETS.get(task_id, set()).discard(websocket)
        if not ACTIVE_WEBSOCKETS.get(task_id):
            ACTIVE_WEBSOCKETS.pop(task_id, None)


def claim_next_job():
    """
    本地client轮询此端点获取下一个待执行任务。
    返回最早创建且状态为PENDING的任务。
    """
    with Session(_engine()) as s:
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
        task.progress_message = "本地client已领取任务"
        s.commit()
        s.refresh(task)

        job = task.to_dict()
        job["video_path"] = task.video_path
        job["vlm_api_key"] = task.vlm_api_key or ""
        job["vlm_model"] = task.vlm_model or ""
        job["ollama_host"] = task.ollama_host or "http://localhost:11434"
        job["ollama_model"] = task.ollama_model or "qwen3-vl:8b"
        return {"job": job}


@router.get("/jobs/next")
@router.get("/ws/jobs/next")
def get_next_job():
    return claim_next_job()
