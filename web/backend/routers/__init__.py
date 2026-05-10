from .tasks import router as tasks_router
from .analysis import router as analysis_router
from .config import router as config_router
from .websocket import router as websocket_router

__all__ = ['tasks', 'analysis', 'config', 'websocket']