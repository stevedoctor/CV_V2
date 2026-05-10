# 会议注意力检测系统 - 开发日志 [版本2.1]

## 项目目的
基于视频的员工注意力检测系统，用于监测会议/班前会中员工的注意力状态，及时发现异常行为（疲劳、分心、情绪失控等）。

---

## 双路线策略

### 路线1: YOLO-Pose + ByteTrack + 规则/模型（本路线）✅ MVP完成
- **当前状态**: 规则引擎MVP已完成，已修复关键Bug

### 路线2: 多模态大模型直接分析（待开发）
- **开发位置**: `../route2/`

---

## 历史进展

### v2.0及之前
详见 `README_v2.0.md`

---

## 当前进展

### 版本2.1: 俯仰角计算Bug修复

**问题发现**：
在v2.0测试中发现"正常"占比为0%，深度诊断发现：

```
俯仰角均值: -97.7° （异常！预期应在 -30° ~ 30°）
低头帧占比: 100% （不可能！）
```

**根因分析**：

原版俯仰角计算使用肩膀中心：
```python
dy = nose[1] - shoulder_center[1]
dx = nose[0] - shoulder_center[0]
return np.degrees(np.arctan2(dy, dx))
```

问题：图像坐标系中鼻子在肩膀上方，导致角度约 -90°

**修复方案**：

改用眼睛中心作为参考点：
```python
eye_center_y = (left_eye[1] + right_eye[1]) / 2
eye_distance = abs(right_eye[0] - left_eye[0])
nose_offset_y = nose[1] - eye_center_y
normalized_offset = nose_offset_y / eye_distance
pitch = (normalized_offset - 0.4) * 90  # 估算角度
```

**修改文件**：

| 文件 | 修改内容 |
|------|----------|
| `src/core/geometry.py` | 重写`calculate_head_pitch`方法 |
| `src/config/constants.py` | `HEAD_DOWN_THRESHOLD`: -15 → 15 |
| `src/processors/person_tracker.py` | 判断逻辑: `<` → `>` |

**修复效果对比**：

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| 俯仰角均值 | -97.7° | ~12° |
| 正常占比 | 0% | **10.4%** |
| 轻微占比 | 77.7% | 85.9% |
| 中度占比 | 20.3% | 2.5% |
| 重度占比 | 2.0% | 1.2% |

**测试验证**：

```bash
# 规则引擎测试
python main_rules.py --frames 100 --no-display

# 跟踪ID一致性测试
python tests/test_tracking.py --frames 100 --no-display

# 结果：6个ID中5个100%稳定跟踪
```

---

## 技术要点记录

### 头部俯仰角计算原理

```
眼睛中心 ────┐
             │ ← 眼距 (归一化基准)
             │
鼻子 ────────┘ ← 垂直偏移判断低头程度

低头时：鼻子在眼睛下方更多 → offset大 → pitch正
抬头时：鼻子在眼睛上方 → offset负 → pitch负
正视：  offset ≈ 0.4 × 眼距 → pitch ≈ 0
```

### ByteTrack跟踪效果

- 6个人员ID被稳定跟踪
- 5个ID在100帧内100%出现，无间隙
- 1个ID有7帧间隙（遮挡后恢复原ID）

---

## 下一步计划

1. 继续优化阈值参数（根据实际业务需求微调）
2. 开发路线2（VLM多模态分析）
3. 使用VLM生成标注数据

---

## 更新时间
2026-05-03 02:00

## 版本历史
- v2.1: 俯仰角计算Bug修复，正常占比从0%提升到10.4%
- v2.0: 路线1 MVP完成
- v1.0-v1.3: 核心模块开发
- v0.1-v0.3: 项目初始化
