<div align="center">

# 👁️ RFAC 视频分析平台

### 基于视频的会议注意力检测系统

**YOLO-Pose · ByteTrack · RFAC 四维评分 · VLM 研判 · CNN-LSTM 集成**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)](https://react.dev)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## 📖 项目简介

RFAC 视频分析平台是一套完整的企业级会议注意力检测系统，通过视频分析实时评估参会人员的注意力状态。系统融合了计算机视觉姿态估计、自定义 RFAC 四维评分模型、视觉语言模型（VLM）研判和深度学习时序建模，将每个人的注意力状态分为四个等级：

| 等级 | 含义 | 色标 |
|:----:|:----:|:----:|
| NORMAL | 注意力正常 | 🟢 |
| MILD | 轻度涣散 | 🟡 |
| MODERATE | 中度涣散 | 🟠 |
| SEVERE | 严重涣散 | 🔴 |

系统提供 **三条技术路线**，覆盖从零样本到深度学习的不同场景需求，并配套完整的 Web 平台实现任务管理、实时进度追踪和可视化分析。

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                  Browser (React SPA)                     │
│         任务提交 · 路线选择 · 实时进度 · 结果可视化         │
└──────────────────────┬──────────────────────────────────┘
                       │ HTTP / WebSocket
┌──────────────────────▼──────────────────────────────────┐
│              Cloud Server (FastAPI :8000)                 │
│         任务队列 · 进度推送 · VLM配置 · 结果存储           │
└──────────────────────┬──────────────────────────────────┘
                       │ Poll /api/jobs/next
┌──────────────────────▼──────────────────────────────────┐
│              Local Client (Python + GPU)                  │
│    轮询任务 → 执行分析 → WebSocket进度 → 上传结果          │
│                                                          │
│  ┌──────────┐  ┌──────────┐  ┌───────────────────┐      │
│  │ Route 1  │  │ Route 2  │  │     Route 3       │      │
│  │规则+VLM  │  │VLM端到端 │  │规则+CNN-LSTM+VLM │      │
│  └──────────┘  └──────────┘  └───────────────────┘      │
└─────────────────────────────────────────────────────────┘
```

---

## 🛤️ 三条技术路线

### Route 1 — 规则引擎 + VLM 研判

> 逐帧实时分析，确定性评分，低部署成本

```
YOLO-Pose → ByteTrack → RFAC四维评分 → 规则阈值分级 → VLM复核(可选)
```

- **RFAC 四维评分模型**：自定义注意力量化框架
  - **A**pathy（涣散 55%）：正视率
  - **F**atigue（疲劳 25%）：低头频次 + 身体摇摆
  - **R**ushing（匆忙 15%）：躯干速度 + 手势活跃度
  - **Frustration**（情绪失控 5%）：肢体对抗性 + 前倾角 + 姿态不稳

- **VLM 研审模块**：对 MODERATE 及以上等级触发多线程 VLM 复审，生成干预建议
- 延迟低，无需训练数据，适合快速部署

### Route 2 — VLM 端到端

> 零样本，多线程并发，上下文理解能力强

```
帧采样 → ByteTrack检测 → 个体裁剪 → 多线程VLM分析 → 结果聚合
```

- **FrameSampler**：均匀 / 关键帧采样（默认 8 帧）
- **IndividualAnalyzer**：每人独立 ROI 裁剪 + VLM 并发调用
- **ResultAggregator**：聚合为完整分析报告（等级分布 + RFAC 分数）
- 无需标注数据，VLM 直接输出 RFAC 四维评分 + 注意力等级 + 干预建议

### Route 3 — 规则引擎 + CNN-LSTM 集成

> 规则确定性 + 深度学习泛化，三种集成策略

```
Route 1管线 → CNN-LSTM推理 → EnsemblePredictor → VLM复核(可选)
```

- **CNN-LSTM 架构**：
  ```
  Input: keypoints(T,17,3) + RFAC(T,4)
    → SpatialEncoder(MLP: 55→128→128)
    → BiLSTM(2层, 双向: 128)
    → frame_head(128→4) + clip_head(128→64→4)
  ```
- **FocalLoss** 解决类别不平衡（SEVERE 仅占 0.7%）
- **EnsemblePredictor** 三种融合策略：`max`（悲观）/ `weighted`（加权投票）/ `model`（模型优先）

### 路线对比

| 特性 | Route 1 | Route 2 | Route 3 |
|:----:|:-------:|:-------:|:-------:|
| 核心方法 | 规则引擎 + VLM | VLM 端到端 | 规则 + CNN-LSTM |
| 是否实时 | ✅ 逐帧 | ❌ 离线批处理 | ✅ 逐帧 |
| 是否需要训练 | ❌ | ❌ | ✅ |
| API 成本 | 低 | 高（每帧调用） | 低 |
| 确定性 | 强 | 弱（VLM随机性） | 强 |
| 上下文理解 | 弱 | 强 | 中 |

---

## 🧠 核心模型

### RFAC 四维评分

自定义的注意力量化框架，综合四个维度加权得出总体注意力分数：

```
Overall = 0.55 × Apathy + 0.25 × Fatigue + 0.15 × Rushing + 0.05 × Frustration
```

| 分数范围 | 等级 |
|:--------:|:----:|
| < 0.3 | NORMAL |
| 0.3 – 0.5 | MILD |
| 0.5 – 0.7 | MODERATE |
| ≥ 0.7 | SEVERE |

### YOLO-Pose + ByteTrack

- **YOLO11n-Pose**：轻量级人体姿态估计，输出 17 个 COCO 关键点
- **ByteTrack**：多目标跟踪器，保证跨帧 ID 一致性

### VLM 支持

| Provider | 模型 | 部署方式 |
|:--------:|:----:|:--------:|
| Ollama | Qwen3-VL:8B | 本地 GPU |
| SiliconFlow | Qwen3-VL-32B-Instruct / Qwen2-VL-72B-Instruct | 云端 API |
| Mock | — | 本地测试 |

---

## 💻 Web 平台

基于 **FastAPI + React** 的全栈管理平台，采用云-端分离架构（前端云部署，计算本地 GPU）。

### 功能

- 📤 提交本地视频路径，选择分析路线
- 📊 实时 WebSocket 进度推送
- 📋 任务列表管理（创建/查看/删除）
- 🍩 注意力等级分布饼图（Recharts）
- ⚙️ VLM 配置管理（Provider / 模型 / 线程数 / 触发等级）
- 🔗 连接状态实时监测

### 技术栈

| 层 | 技术 |
|:--:|:----:|
| 前端 | React 18 + Vite 6 + Recharts + Axios |
| 后端 | FastAPI + SQLAlchemy + SQLite + WebSocket |
| 本地客户端 | Python async (httpx + websockets) |

---

## 🚀 快速开始

### 环境要求

- Python 3.10+
- CUDA 兼容 GPU（推理用）
- Node.js 18+（前端开发）
- [Ollama](https://ollama.ai)（可选，本地 VLM）

### 1. 克隆项目

```bash
git clone git@github.com:stevedoctor/CV_V2.git
cd CV_V2
```

### 2. 安装依赖

```bash
# Route 1 & 3
pip install -r route1/requirements.txt

# Route 2
pip install -r route2/requirements.txt

# Route 3 CNN-LSTM 训练（额外）
pip install torch torchvision

# Web 后端
pip install -r web/backend/requirements.txt

# Web 前端
cd web/frontend && npm install
```

### 3. 下载模型权重

下载 [YOLO11n-Pose](https://docs.ultralytics.com/models/yolo11/) 权重文件 `yolo11n-pose.pt`，放置到：

```
route1/yolo11n-pose.pt
route2/yolo11n-pose.pt
route3/yolo11n-pose.pt
```

### 4. 启动服务

```bash
# 终端 1：启动后端
cd web/backend
uvicorn main:app --host 0.0.0.0 --port 8000

# 终端 2：启动前端
cd web/frontend
npm run dev

# 终端 3：启动本地客户端（连接 GPU 执行分析）
cd web/local_client
python client.py --server http://localhost:8000
```

访问 http://localhost:5173 即可使用。

### 5. 环境变量

| 变量 | 用途 |
|:----:|:----:|
| `SILICONFLOW_API_KEY` | SiliconFlow 云端 VLM API Key |
| `CV_WEB_SERVER_URL` | 本地客户端连接的服务器地址（默认 `http://localhost:8000`） |

---

## 📁 项目结构

```
CV_V2/
├── route1/                    # Route 1: 规则引擎 + VLM 研判
│   ├── main_rules.py          # 入口：MeetingAttentionDetector
│   ├── src/
│   │   ├── trackers/          # ByteTrack 多目标跟踪
│   │   ├── models/            # 姿态估计 + 注意力评分
│   │   ├── processors/        # RFAC计算 + 人员状态追踪
│   │   ├── rules/             # 规则阈值分级引擎
│   │   ├── audit/             # VLM 多线程研判
│   │   ├── core/              # 几何计算模块
│   │   └── config/            # 配置 + 常量
│   └── requirements.txt
│
├── route2/                    # Route 2: VLM 端到端
│   ├── main_vlm.py            # 入口：analyze_video()
│   ├── src/
│   │   ├── samplers/          # 帧采样器
│   │   ├── trackers/          # ByteTrack 跟踪
│   │   ├── analyzers/         # 个体裁剪 + VLM 分析
│   │   ├── vlms/              # VLM 抽象层（Ollama/SiliconFlow/Mock）
│   │   ├── prompts/           # RFAC Prompt 模板
│   │   ├── postprocessors/    # 结果聚合
│   │   └── visualizers/       # 帧标注可视化
│   └── requirements.txt
│
├── route3/                    # Route 3: 规则 + CNN-LSTM 集成
│   ├── main_rules.py          # 入口：Extend MeetingAttentionDetector
│   ├── src/
│   │   ├── ...                # (同 Route 1 结构)
│   │   └── models/
│   │       ├── cnn_lstm.py    # AttentionLSTM 网络
│   │       ├── train.py       # 训练脚本（FocalLoss）
│   │       ├── inference.py   # EnsemblePredictor
│   │       └── dataset_loader.py
│   └── requirements.txt
│
├── web/
│   ├── backend/               # FastAPI 后端
│   │   ├── main.py
│   │   ├── models/            # SQLAlchemy 数据模型
│   │   ├── routers/           # API 路由（分析/任务/配置/WebSocket）
│   │   └── executors/         # 路线执行器
│   ├── frontend/              # React 前端
│   │   └── src/
│   │       ├── App.jsx        # 主应用组件
│   │       ├── index.css      # 样式
│   │       └── services/      # API 调用
│   └── local_client/          # Python 本地 GPU 客户端
│       └── client.py
│
└── .gitignore
```

---

## 📈 开发进度

| 步骤 | 内容 | 状态 |
|:----:|:----:|:----:|
| Step 1 | 基础姿态估计（YOLO-Pose + ByteTrack + 滑动窗口） | ✅ 完成 |
| Step 2 | Route 1 规则引擎（RFAC 四维 + 阈值分级 + VLM 研判 + 多线程） | ✅ 完成 |
| Step 3 | Route 2 VLM 端到端（帧采样 + 个体裁剪 + 多线程 VLM + 结果聚合） | ✅ 完成 |
| Step 4 | 构建训练数据集（1592 clips, FocalLoss, 数据划分） | 🔧 进行中 |
| Step 5 | Route 3 CNN-LSTM（依赖 Step 4 数据集） | ⏳ 待开始 |
| Step 6 | 系统集成与优化（路线对比 / TensorRT / API文档） | ⚠️ 部分完成 |

---

## 📄 License

This project is licensed under the MIT License.
