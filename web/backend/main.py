"""
FastAPI 主入口 - 会议注意力检测系统
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from models.database import init_db

app = FastAPI(
    title="会议注意力检测系统",
    description="前端在云端，算力在本地。服务器协调任务，本地GPU执行推理。",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

os.makedirs(os.path.join(BASE_DIR, "data", "videos"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "data", "results"), exist_ok=True)

from routers import analysis, tasks, config, websocket

app.include_router(analysis.router, prefix="/api")
app.include_router(tasks.router, prefix="/api")
app.include_router(config.router, prefix="/api")
app.include_router(websocket.router, prefix="/ws")

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "data", "results")), name="static")


@app.on_event("startup")
def startup():
    init_db()
    print("[Server] 数据库初始化完成")


@app.get("/")
def root():
    return {
        "service": "会议注意力检测系统",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "POST /api/analysis/upload": "提交新分析任务",
            "GET /api/tasks": "获取任务列表",
            "GET /api/tasks/{task_id}": "获取任务详情",
            "GET /api/jobs/next": "本地client获取下一个待执行任务",
            "WS /ws/{task_id}": "WebSocket实时进度",
        }
    }


@app.get("/api/jobs/next")
def get_next_job():
    return websocket.claim_next_job()


@app.get("/health")
def health():
    return {"status": "ok"}