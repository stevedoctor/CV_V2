# 会议注意力检测系统 - 开发日志 [版本1.2]

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

### 步骤1.0: 创建ByteTrack跟踪模块（已完成）
✅ 实现跨帧ID一致的跟踪核心模块

### 步骤1.1: 创建跟踪测试脚本（已完成）
✅ 验证ByteTrack跟踪ID的一致性和稳定性

---

## 当前进展

### 步骤1.2: 创建规则阈值分级引擎（已完成）
**目标**: 实现将RFAC指标转换为0-3级异常等级的规则引擎

**实现方案**:
创建可配置的规则判定引擎，支持：
1. 四维度独立分级
2. 综合异常评估
3. 干预建议生成

**核心代码** (`src/rules/rule_engine.py`):

**关键类**:
```python
@dataclass
class RuleThresholds:
    """规则阈值配置（所有阈值可调整）"""
    rushing_velocity_l0: float = 0.1   # 躯干速度阈值
    fatigue_head_down_l0: float = 0.2 # 低头时间占比阈值
    apathy_forward_l0: float = 0.7    # 正视率阈值
    frustration_activity_l0: float = 0.15  # 手势活跃度阈值

class RuleEngine:
    def evaluate_rushing(self, torso_velocity: float) -> int:
        """评估匆忙等级 (0-3)"""
        
    def evaluate_fatigue(self, head_down_ratio: float) -> int:
        """评估疲劳等级 (0-3)"""
        
    def evaluate_apathy(self, forward_rate: float) -> int:
        """评估注意力涣散等级 (0-3)"""
        
    def evaluate_frustration(self, gesture_activity: float) -> int:
        """评估情绪失控等级 (0-3)"""
        
    def evaluate_all(self, rfac_score) -> RuleResult:
        """综合评估所有维度"""
        
    def get_intervention_suggestion(self, result: RuleResult) -> str:
        """获取干预建议"""
```

**分级规则详解**:
| 维度 | 指标 | 0级(正常) | 1级(轻微) | 2级(中度) | 3级(重度) |
|------|------|----------|----------|----------|----------|
| Rushing | 躯干速度 | <0.1 | <0.3 | <0.5 | ≥0.5 |
| Fatigue | 低头占比 | <0.2 | <0.4 | <0.6 | ≥0.6 |
| Apathy | 正视率 | ≥0.7 | ≥0.5 | ≥0.3 | <0.3 |
| Frustration | 手势活跃度 | <0.15 | <0.35 | <0.55 | ≥0.55 |

**使用示例**:
```python
from src.rules import RuleEngine, RuleThresholds

# 使用默认阈值
engine = RuleEngine()

# 自定义阈值
thresholds = RuleThresholds(
    fatigue_head_down_l1=0.3,  # 调整疲劳判定阈值
    apathy_forward_l0=0.8     # 调整注意力判定阈值
)
engine = RuleEngine(thresholds=thresholds)

# 评估单维度
rushing_level = engine.evaluate_rushing(torso_velocity=0.4)

# 综合评估
result = engine.evaluate_all(rfac_score)
print(result)  # RuleResult(Rushing=中度, Fatigue=轻微, ...)

# 获取干预建议
suggestion = engine.get_intervention_suggestion(result)
```

**RuleResult输出示例**:
```
RuleResult(
    rushing_level=1,     # 轻微匆忙
    fatigue_level=2,     # 中度疲劳
    apathy_level=0,      # 注意力正常
    frustration_level=0, # 情绪正常
    overall_level=1      # 综合轻微异常
)
```

**文件结构更新**:
```
route1/src/rules/
├── __init__.py
└── rule_engine.py      # 新增
```

**下一步计划**:
- 步骤1.3: 创建主入口程序（整合ByteTrack + RFAC + 规则引擎）

---

## 更新时间
2026-05-03 01:00

## 版本历史
- v1.2: 创建规则阈值分级引擎
- v1.1: 创建跟踪测试脚本
- v1.0: 创建ByteTrack跟踪模块（里程碑版本）
- v0.3: 确认使用ultralytics内置ByteTrack
- v0.2: 复制基础代码模块
- v0.1: 创建项目结构和初始README
