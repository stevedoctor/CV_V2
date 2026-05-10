# 会议注意力检测系统 - Web 前端

## 架构

```
                    ┌─────────────────────┐
                    │   其他电脑浏览器    │
                    └──────────┬──────────┘
                               │ HTTP
                    ┌──────────▼──────────┐
                    │   轻量云服务器      │
                    │   FastAPI (8000)   │
                    │   React (5173)     │
                    └──────────┬──────────┘
                               │ HTTP/WebSocket
                    ┌──────────▼──────────┐
                    │   本地机器 (你)     │
                    │   Route1/2/3 程序   │
                    │   local_client.py  │
                    └─────────────────────┘
```

## 快速启动（本地测试）

### 1. 启动后端
```bash
cd /home/stve/out/big\ data/CV_V2/web/backend
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 2. 启动前端
```bash
cd /home/stve/out/big\ data/CV_V2/web/frontend
npm run dev
# 浏览器打开 http://localhost:5173
```

### 3. 启动本地 Client
```bash
cd /home/stve/out/big\ data/CV_V2/web/local_client
python client.py --server http://localhost:8000
```

## 上传视频分析流程

1. 浏览器打开 `http://localhost:5173`
2. 上传视频文件 → 选择路线（route1/2/3）
3. 配置 VLM 参数（可选）
4. 点击"开始分析"
5. 前端创建任务 → 上传到服务器
6. 本地 client 轮询 `/api/jobs/next` 取任务
7. 本地执行分析 → WebSocket 上报进度
8. 前端实时显示进度
9. 分析完成 → 结果展示

## 部署到阿里云

### 购买 ECS
1. 阿里云 ECS → GPU 实例 → **gn6i（T4 16GB）**
2. Ubuntu 22.04 / 4核15GB / 50GB SSD
3. 按量付费约 **1.2 元/小时**，或月付约 600 元

### 服务器初始化
```bash
# 安装 Python 3.11
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip

# 安装 Node.js 18
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install nodejs

# 克隆代码
cd /home
git clone <your-repo> .
cd web/backend
pip install -r requirements.txt
```

### 启动服务
```bash
# 后端（后台运行）
nohup uvicorn main:app --host 0.0.0.0 --port 8000 > backend.log 2>&1 &

# 前端（构建+nginx托管）
cd ../frontend
npm run build
# 或用 vite dev server
nohup npm run dev -- --host 0.0.0.0 > frontend.log 2>&1 &
```

### 防火墙
- 开放 **8000** 端口（API）
- 开放 **5173** 端口（前端dev）或 **80** 端口（nginx）

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/analysis/upload` | 上传视频，创建任务 |
| GET | `/api/tasks` | 获取任务列表 |
| GET | `/api/tasks/{task_id}` | 获取任务详情 |
| GET | `/api/jobs/next` | 本地 client 取下一个任务 |
| GET | `/api/config/vlm` | 获取 VLM 配置 |
| POST | `/api/config/vlm` | 更新 VLM 配置 |
| WS | `/ws/{task_id}` | WebSocket 进度 |

## 本地 Client 使用

```bash
# 基本用法
python client.py --server http://YOUR_SERVER_IP:8000

# 调整轮询间隔
python client.py --server http://YOUR_SERVER_IP:8000 --poll-interval 10

# 按 Ctrl+C 退出
```

本地 client 会：
- 每 5 秒轮询 `/api/jobs/next`
- 取到任务后根据 route 执行对应分析程序
- 通过 WebSocket 实时推送进度到服务器
- 完成后上传结果到 `/api/analysis/tasks/{task_id}/result`

## 项目结构

```
web/
├── backend/
│   ├── main.py              # FastAPI 入口
│   ├── requirements.txt     # Python 依赖
│   ├── models/database.py    # SQLite 模型
│   ├── routers/             # API 路由
│   │   ├── tasks.py         # 任务 CRUD
│   │   ├── analysis.py      # 视频上传 + 结果
│   │   ├── config.py       # VLM 配置
│   │   └── websocket.py    # WebSocket
│   └── executors/           # 路线执行器封装
│       ├── route1_executor.py
│       ├── route2_executor.py
│       └── route3_executor.py
├── local_client/
│   └── client.py           # 本地轮询客户端
└── frontend/
    ├── src/
    │   ├── App.jsx          # 主界面
    │   ├── services/api.js  # API 调用
    │   └── index.css        # 样式
    ├── package.json
    └── vite.config.js
```

## 注意事项

- 本地 client 和服务器之间通过 HTTP + WebSocket 通信
- 视频文件存在服务器，结果 JSON 上传到服务器
- 同一时间只能执行一个分析任务（GPU 限制）
- 服务器端的任务队列支持多人同时提交