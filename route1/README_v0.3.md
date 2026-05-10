# 会议注意力检测系统 - 开发日志 [版本0.3]

## 项目目的
基于视频的员工注意力检测系统，用于监测会议/班前会中员工的注意力状态，及时发现异常行为（疲劳、分心、情绪失控等）。

---

## 双路线策略

### 路线1: YOLO-Pose + ByteTrack + 规则/模型（本路线）
- **技术栈**: YOLOv8-Pose + ByteTrack多目标跟踪 + 规则阈值引擎 + CNN-LSTM分类
- **适用场景**: 实时预警、低延迟检测

### 路线2: 多模态大模型直接分析
- **技术栈**: 视频帧采样 + 通义千问VL/GPT-4V/Ollama
- **适用场景**: 离线复盘、零样本快速原型、数据标注辅助

---

## 历史进展

### 步骤0.1: 创建项目结构（已完成）
✅ 建立清晰的项目目录结构

### 步骤0.2: 复制基础代码（已完成）
✅ 从CV/目录复制核心模块

---

## 当前进展

### 步骤0.3: 准备跟踪依赖（已完成）
**目标**: 确认ByteTrack跟踪的技术方案

**实现方案**:
**重要发现**: Ultralytics YOLOv8已内置ByteTrack跟踪功能！

使用`model.track()`替代`model()`即可启用跟踪：
```python
from ultralytics import YOLO
model = YOLO('yolo11n-pose.pt')

# 带跟踪的检测
results = model.track(source=video_path, stream=True, 
                      tracker="bytetrack.yaml", persist=True)
```

优势：
- 无需额外安装boxmot（避免依赖冲突）
- 与YOLO-Pose无缝集成
- 自动维护track_id跨帧一致性

**依赖清单** (requirements.txt):
```
ultralytics>=8.0.0
numpy
opencv-python
```

**关键参数说明**:
| 参数 | 说明 |
|------|------|
| `tracker="bytetrack.yaml"` | 使用ByteTrack算法 |
| `persist=True` | 保持跟踪状态跨帧 |
| `stream=True` | 流式处理视频 |

**下一步计划**:
- 步骤1.0: 创建ByteTrack跟踪模块（封装ultralytics.track）

---

## 更新时间
2026-05-03 00:45

## 版本历史
- v0.3: 确认使用ultralytics内置ByteTrack
- v0.2: 复制基础代码模块
- v0.1: 创建项目结构和初始README
