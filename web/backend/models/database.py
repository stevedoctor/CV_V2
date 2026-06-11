"""
数据库模型 - SQLite任务记录
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Enum as SQLEnum, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
import enum
import datetime

Base = declarative_base()


class TaskStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskPriority(str, enum.Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


class Task(Base):
    __tablename__ = "tasks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(String(64), unique=True, index=True, nullable=False)
    
    video_name = Column(String(256), nullable=False)
    video_path = Column(String(512), nullable=True)
    
    route = Column(String(32), nullable=False)
    
    status = Column(SQLEnum(TaskStatus), default=TaskStatus.PENDING)
    
    vlm_provider = Column(String(32), default="none")
    vlm_trigger = Column(String(16), default="MODERATE")
    vlm_api_key = Column(String(256), default="")
    vlm_model = Column(String(128), default="")
    ollama_host = Column(String(256), default="http://localhost:11434")
    ollama_model = Column(String(128), default="qwen3-vl:8b")

    workers = Column(Integer, default=4)
    
    progress = Column(Float, default=0.0)
    progress_message = Column(String(256), default="")
    
    result_json = Column(Text, nullable=True)
    
    error_message = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    submitted_by = Column(String(64), default="anonymous")
    
    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "video_name": self.video_name,
            "route": self.route,
            "status": self.status.value if isinstance(self.status, TaskStatus) else self.status,
            "progress": self.progress,
            "progress_message": self.progress_message,
            "vlm_provider": self.vlm_provider,
            "vlm_trigger": self.vlm_trigger,
            "vlm_model": self.vlm_model,
            "ollama_host": self.ollama_host,
            "ollama_model": self.ollama_model,
            "workers": self.workers,
            "result": self.result_json,
            "error": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


def get_db_path() -> str:
    return "/home/stve/out/big data/CV_V2/web/backend/data/tasks.db"


def init_db():
    import os
    os.makedirs(os.path.dirname(get_db_path()), exist_ok=True)

    from sqlalchemy import create_engine
    engine = create_engine(f"sqlite:///{get_db_path()}")
    Base.metadata.create_all(engine)

    with engine.begin() as conn:
        columns = {row[1] for row in conn.execute(text("PRAGMA table_info(tasks)"))}
        if "ollama_host" not in columns:
            conn.execute(text("ALTER TABLE tasks ADD COLUMN ollama_host VARCHAR(256) DEFAULT 'http://localhost:11434'"))
        if "ollama_model" not in columns:
            conn.execute(text("ALTER TABLE tasks ADD COLUMN ollama_model VARCHAR(128) DEFAULT 'qwen3-vl:8b'"))

    return engine