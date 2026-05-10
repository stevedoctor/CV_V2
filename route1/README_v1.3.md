# 会议注意力检测系统 - 开发日志 [版本1.3]

## 项目目的
基于视频的员工注意力检测系统，用于监测会议/班前会中员工的注意力状态，及时发现异常行为（疲劳、分心、情绪失控等）。

---

## 双路线策略

### 路线1: YOLO-Pose + ByteTrack + 规则/模型（本路线）
- **技术栈**: YOLOv8-Pose + ByteTrack多目标跟踪 + 规则阈值引擎 + CNN-LSTM分类
- **适用场景**: 实时预警、低延迟检测
- **当前阶段**: 规则引擎MVP已完成

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

### 步骤1.0: 创建ByteTrack跟踪模块（已完成）
✅ 实现跨帧ID一致的跟踪核心模块

### 步骤1.1: 创建跟踪测试脚本（已完成）
✅ 验证ByteTrack跟踪ID的一致性和稳定性

### 步骤1.2: 创建规则阈值分级引擎（已完成）
✅ 实现将RFAC指标转换为0-3级异常等级

---

## 当前进展

### 步骤1.3: 创建主入口程序（已完成）
**目标**: 整合ByteTrack + RFAC + 规则引擎，提供完整的处理流程

**实现方案**:
创建`MeetingAttentionDetector`类，封装完整检测流程。

**核心代码** (`main_rules.py`):

**完整流程**:
```
视频帧 → ByteTrack跟踪 → 姿态估计 → RFAC计算 → 规则引擎 → 可视化
           ↓
      track_id (跨帧一致)
```

**关键类**:
```python
class MeetingAttentionDetector:
    def __init__(self, model_path, history_size=10, fps=20):
        """初始化检测器"""
        self.tracker = ByteTrackManager(model_path)
        self.person_tracker = PersonTracker(history_size)
        self.rfac_calculator = RFACCalculator()
        self.rule_engine = RuleEngine()
        
    def process_frame(self, frame) -> List[tuple]:
        """处理单帧，返回[(track_id, rule_result, rfac_score, keypoints), ...]"""
        
    def render_frame(self, frame, results) -> np.ndarray:
        """渲染可视化结果"""
```

**使用方式**:
```bash
cd route1

# 基本运行（显示窗口）
python main_rules.py --video data/videos/meeting_attention_video.mp4

# 后台运行
python main_rules.py --video data/videos/meeting_attention_video.mp4 --no-display

# 保存输出视频
python main_rules.py --video data/videos/meeting_attention_video.mp4 --save --output output_result.mp4

# 限制帧数（快速测试）
python main_rules.py --video data/videos/meeting_attention_video.mp4 --frames 100 --no-display
```

**可视化说明**:
- 绿色: 正常（0级）
- 黄色: 轻微异常（1级）
- 橙色: 中度异常（2级）
- 红色: 重度异常（3级）

**输出格式**:
```
ID:1 L0(正常)  # track_id + 综合等级
R0 F1 A0       # 各维度等级: R=Rushing, F=Fatigue, A=Apathy
```

**模块导入验证**:
```
✅ 所有模块导入成功
```

**文件结构更新**:
```
route1/
├── main_rules.py      # 新增 - 主入口
└── src/
    ├── trackers/      # ByteTrack模块
    ├── rules/         # 规则引擎
    ├── processors/    # RFAC计算
    ├── models/        # 姿态估计
    └── ...
```

**下一步计划**:
- 步骤2.0: 运行集成测试验证，输出完整测试报告

---

## 更新时间
2026-05-03 01:05

## 版本历史
- v1.3: 创建主入口程序，整合完整检测流程
- v1.2: 创建规则阈值分级引擎
- v1.1: 创建跟踪测试脚本
- v1.0: 创建ByteTrack跟踪模块（里程碑版本）
- v0.3: 确认使用ultralytics内置ByteTrack
- v0.2: 复制基础代码模块
- v0.1: 创建项目结构和初始README
