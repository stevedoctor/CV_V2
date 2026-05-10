# 会议注意力检测系统 - 开发日志 [版本2.4]

## 项目目的
基于视频的员工注意力检测系统，用于监测会议/班前会中员工的注意力状态。

---

## 双路线策略

### 路线1: YOLO-Pose + ByteTrack + 规则引擎 + VLM审计（本路线）
- **当前状态**: ✅ 支持多VLM提供者
- **新增**: 硅基流动云端API支持

### 路线2: 多模态大模型直接分析
- **位置**: `../route2/`

---

## 历史进展

### v2.3及之前
详见 `README_v2.3.md` 及更早版本

---

## 当前进展

### 版本2.4: 添加硅基流动API支持

**需求背景**：
用户有：
1. 本地部署的 `qwen3-vl:8b`（Ollama）
2. 硅基流动API Key，希望使用云端 `Qwen2-VL-72B-Instruct`

**新增模块**：

| 文件 | 说明 |
|------|------|
| `src/vlms/siliconflow_vl.py` | 🆕 硅基流动API实现 |
| `src/config/settings.py` | 添加 `siliconflow_model` 配置 |
| `main_rules.py` | 添加 `--vlm-provider siliconflow` 参数 |

---

## VLM提供者对比

| 提供者 | 默认模型 | 运行位置 | 成本 | 特点 |
|--------|----------|----------|------|------|
| **mock** | - | 本地模拟 | 免费 | 测试用 |
| **ollama** | `qwen3-vl:8b` | 本地 | 免费 | 隐私保护 |
| **siliconflow** | `Qwen/Qwen2-VL-72B-Instruct` | 云端 | 按token付费 | 效果最强 |
| **qwen** | `qwen-vl-max` | 云端 | 按token付费 | 阿里云官方 |

---

## 使用方式

### 方式一：本地Ollama（免费）

```bash
# 1. 确保Ollama运行
ollama serve

# 2. 确认模型已安装
ollama list
# 应看到 qwen3-vl:8b

# 3. 运行
python main_rules.py --video meeting.mp4 --vlm

# 或明确指定
python main_rules.py --video meeting.mp4 --vlm \
    --vlm-provider ollama \
    --ollama-model qwen3-vl:8b
```

### 方式二：硅基流动云端（推荐）

```bash
# 1. 设置API Key
export SILICONFLOW_API_KEY="sk-xxxxxxxx"

# 2. 运行（使用72B模型）
python main_rules.py --video meeting.mp4 --vlm \
    --vlm-provider siliconflow

# 3. 可选：指定其他模型
python main_rules.py --video meeting.mp4 --vlm \
    --vlm-provider siliconflow \
    --siliconflow-model Qwen/Qwen2-VL-7B-Instruct
```

### 方式三：通义千问（阿里云）

```bash
# 1. 设置API Key
export DASHSCOPE_API_KEY="sk-xxxxxxxx"

# 2. 运行
python main_rules.py --video meeting.mp4 --vlm --vlm-provider qwen
```

---

## 命令参数完整列表

```bash
python main_rules.py --help
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--video` | data/videos/meeting_attention_video.mp4 | 视频路径 |
| `--model` | yolo11n-pose.pt | YOLO模型 |
| `--no-display` | False | 不显示窗口 |
| `--save` | False | 保存输出视频 |
| `--frames` | None | 最大帧数 |
| **`--vlm`** | False | 启用VLM审计 |
| **`--vlm-provider`** | ollama | VLM提供者 |
| **`--vlm-trigger`** | MODERATE | 触发等级 |
| **`--vlm-api-key`** | - | VLM API Key |
| **`--ollama-host`** | http://localhost:11434 | Ollama地址 |
| **`--ollama-model`** | qwen3-vl:8b | Ollama模型 |
| **`--siliconflow-model`** | Qwen/Qwen2-VL-72B-Instruct | 硅基流动模型 |

---

## 环境变量

| 变量名 | 用途 |
|--------|------|
| `SILICONFLOW_API_KEY` | 硅基流动API Key |
| `DASHSCOPE_API_KEY` | 通义千问API Key |

---

## 硅基流动模型选择建议

| 模型 | 参数量 | 价格 | 推荐场景 |
|------|--------|------|----------|
| `Qwen/Qwen2-VL-7B-Instruct` | 7B | 便宜 | 快速验证 |
| `Qwen/Qwen2-VL-32B-Instruct` | 32B | 中等 | 平衡选择 |
| **`Qwen/Qwen2-VL-72B-Instruct`** | **72B** | **较高** | **最佳效果** |

**当前默认使用72B模型**

---

## 文件结构

```
route1/src/vlms/
├── __init__.py           # VLM工厂函数
├── base_vlm.py           # 基类
├── mock_vlm.py           # 模拟测试
├── ollama_vl.py          # 本地Ollama
├── qwen_vl.py            # 通义千问
└── siliconflow_vl.py     # 🆕 硅基流动
```

---

## 更新时间
2026-05-03

## 版本历史
- v2.4: 添加硅基流动API支持，支持本地/云端切换
- v2.3: 集成VLM审计功能
- v2.2: 阈值参数优化
- v2.1: 俯仰角计算Bug修复
- v2.0: 路线1 MVP完成
