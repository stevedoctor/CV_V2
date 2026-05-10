# 会议注意力检测系统 - 路线1

> **YOLO-Pose + ByteTrack + 规则引擎** 实时检测方案

## 快速开始

```bash
# 进入目录
cd route1

# 快速测试（100帧）
python main_rules.py --frames 100 --no-display

# 完整运行（带显示）
python main_rules.py --video data/videos/meeting_attention_video.mp4

# 保存输出
python main_rules.py --video data/videos/meeting_attention_video.mp4 --save
```

---

## 双路线策略

### ✅ 路线1: YOLO-Pose + ByteTrack + 规则/模型（本路线）
**状态**: MVP完成

| 组件 | 文件 | 说明 |
|------|------|------|
| ByteTrack跟踪 | `src/trackers/bytetrack.py` | 跨帧ID一致 |
| RFAC计算 | `src/processors/rfac_calculator.py` | 四维度指标 |
| 规则引擎 | `src/rules/rule_engine.py` | 0-3级异常判定 |
| 主入口 | `main_rules.py` | 完整处理流程 |

### 🔲 路线2: 多模态大模型直接分析
**状态**: 待开发  
**位置**: `../route2/`

---

## 系统架构

```
视频 → ByteTrack跟踪 → YOLO-Pose → RFAC计算 → 规则引擎 → 异常预警
            ↓
       track_id (跨帧一致)
```

**RFAC四维度模型**:
| 维度 | 权重 | 指标 |
|------|------|------|
| Apathy | 50% | 正视率 |
| Fatigue | 20% | 低头占比 |
| Rushing | 15% | 躯干速度 |
| Frustration | 15% | 手势活跃度 |

---

## 文件结构

```
route1/
├── main_rules.py        # 主入口
├── requirements.txt     # 依赖
├── README.md           # 本文档
├── README_v2.0.md      # 最新开发日志
├── src/
│   ├── trackers/       # ByteTrack
│   ├── rules/          # 规则引擎
│   ├── processors/     # RFAC计算
│   ├── models/         # 姿态估计
│   ├── core/           # 几何计算
│   └── config/         # 配置
├── tests/
│   └── test_tracking.py
└── data/videos/        # 测试视频
```

---

## 开发日志

| 版本 | 说明 | 文件 |
|------|------|------|
| v2.0 | MVP完成 | `README_v2.0.md` |
| v1.3 | 主入口 | `README_v1.3.md` |
| v1.2 | 规则引擎 | `README_v1.2.md` |
| v1.0 | ByteTrack | `README_v1.0.md` |

完整开发记录请查看 `README_v*.md` 系列文件。

---

## 后续计划

1. **近期**: 完整测试验证
2. **中期**: 开发路线2（VLM）
3. **后期**: CNN-LSTM模型 + 双轨集成

---

*更新时间: 2026-05-03*
