# 会议注意力检测系统 - 开发日志 [版本1.0]

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

### 步骤0.3: 准备跟踪依赖（已完成）
✅ 确认使用ultralytics内置ByteTrack

---

## 当前进展（里程碑版本）

### 步骤1.0: 创建ByteTrack跟踪模块（已完成）
**目标**: 实现跨帧ID一致的跟踪核心模块

**实现方案**:
封装ultralytics YOLO的`model.track()`方法，提供简洁的跟踪接口。

**核心代码** (`src/trackers/bytetrack.py`):

**关键接口**:
```python
class ByteTrackManager:
    def track_frame(self, frame) -> List[Tuple[int, np.ndarray]]:
        """单帧跟踪，返回[(track_id, keypoints), ...]"""
        
    def track_video(self, video_path) -> Generator:
        """视频流跟踪，yield(frame, tracked_persons)"""
```

**技术关键点**:
1. `persist=True` - 保持跟踪状态跨帧（最重要的参数！）
2. `result.boxes.id` - 获取跨帧一致的track_id
3. `result.keypoints.data` - 获取姿态关键点

**与旧PersonTracker的区别**:
| 特性 | 旧方案 | 新方案(ByteTrack) |
|------|--------|------------------|
| ID来源 | 帧内索引 `i` | track_id |
| 跨帧一致性 | ❌ 无 | ✅ 有 |
| 遮挡处理 | ❌ 无 | ✅ 自动处理 |
| 人员重入 | 新ID | 恢复原ID |

**使用示例**:
```python
from src.trackers import ByteTrackManager

tracker = ByteTrackManager(model_path="yolo11n-pose.pt")

# 单帧跟踪
tracked = tracker.track_frame(frame)
for track_id, keypoints in tracked:
    print(f"Track {track_id}: {keypoints.shape}")

# 视频流跟踪
for frame, tracked in tracker.track_video("video.mp4"):
    for track_id, kps in tracked:
        process(track_id, kps)
```

**文件结构更新**:
```
route1/src/trackers/
├── __init__.py
└── bytetrack.py      # 新增
```

**下一步计划**:
- 步骤1.1: 创建跟踪测试脚本，验证ID一致性

---

## 更新时间
2026-05-03 00:50

## 版本历史
- v1.0: 创建ByteTrack跟踪模块（里程碑版本）
- v0.3: 确认使用ultralytics内置ByteTrack
- v0.2: 复制基础代码模块
- v0.1: 创建项目结构和初始README
