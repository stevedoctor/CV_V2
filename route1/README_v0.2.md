# 会议注意力检测系统 - 开发日志 [版本0.2]

## 项目目的
基于视频的员工注意力检测系统，用于监测会议/班前会中员工的注意力状态，及时发现异常行为（疲劳、分心、情绪失控等）。

---

## 双路线策略

### 路线1: YOLO-Pose + ByteTrack + 规则/模型（本路线）
- **技术栈**: YOLOv8-Pose + ByteTrack多目标跟踪 + 规则阈值引擎 + CNN-LSTM分类
- **合并策略**: 将原"思路1（CNN-LSTM）"和"思路2（规则引擎）"合并为统一路线
- **适用场景**: 实时预警、低延迟检测

### 路线2: 多模态大模型直接分析
- **技术栈**: 视频帧采样 + 通义千问VL/GPT-4V/Ollama
- **适用场景**: 离线复盘、零样本快速原型、数据标注辅助

---

## 历史进展

### 步骤0.1: 创建项目结构（已完成）
✅ 建立清晰的项目目录结构

---

## 当前进展

### 步骤0.2: 复制基础代码（已完成）
**目标**: 从CV/目录复制可复用的基础代码模块

**实现方案**:
从原有CV/项目复制核心模块，作为路线1的基础框架。

**复制的模块清单**:
```
route1/src/
├── core/                  # 几何计算核心
│   ├── __init__.py
│   └── geometry.py        # 计算俯仰角、肩宽、身体晃动等
├── models/                # 模型模块
│   ├── __init__.py
│   ├── pose_estimator.py  # YOLO-Pose封装
│   └── attention_scorer.py# 注意力评分器
├── processors/            # 处理器模块
│   ├── __init__.py
│   ├── person_tracker.py  # 人员状态追踪
│   └── rfac_calculator.py # RFAC四维度计算器
├── utils/                 # 工具模块
│   ├── __init__.py
│   ├── display_utils.py   # 显示辅助
│   └── video_utils.py     # 视频处理
├── config/                # 配置模块
│   ├── __init__.py
│   ├── settings.py        # 系统配置
│   └── constants.py       # 关键点索引、阈值
└── audit/                 # 审计模块
    ├── __init__.py
    └── audit_system.py    # 大模型复核框架
```

**复制的测试视频**:
```
route1/data/videos/
├── meeting_attention_video.mp4    (86MB)
└── meeting_attention_video1.mp4   (866MB)
```

**关键发现**:
- ✅ 几何计算模块已有完整的俯仰角计算功能（`geometry.py:219`）
- ✅ RFAC四维度计算已实现，但依赖的是索引ID，后续需改造使用track_id
- ⚠️ 当前PersonTracker使用帧内索引作为ID，跨帧不一致

**下一步计划**:
- 步骤0.3: 安装BoxMot依赖（ByteTrack库）
- 步骤1.0: 创建ByteTrack跟踪模块

---

## 更新时间
2026-05-03 00:42

## 版本历史
- v0.2: 复制基础代码模块
- v0.1: 创建项目结构和初始README
