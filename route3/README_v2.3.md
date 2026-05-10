# 会议注意力检测系统 - 开发日志 [版本2.3]

## 项目目的
基于视频的员工注意力检测系统，用于监测会议/班前会中员工的注意力状态。

---

## 双路线策略

### 路线1: YOLO-Pose + ByteTrack + 规则引擎 + VLM审计（本路线）
- **当前状态**: ✅ VLM审计集成完成
- **新增功能**: 可选VLM审计，仅存疑状态触发

### 路线2: 多模态大模型直接分析
- **位置**: `../route2/`

---

## 历史进展

### v2.2及之前
详见 `README_v2.2.md` 及更早版本

---

## 当前进展

### 版本2.3: 集成VLM审计功能

**需求说明**：
1. 可选择使用本地模型（Ollama）或云端大模型（通义千问VL）
2. 仅存疑状态（MODERATE/SEVERE）触发VLM复核
3. 复用路线2的vlms模块

**新增模块**：

| 模块 | 文件 | 说明 |
|------|------|------|
| VLM接口 | `src/vlms/` | 复用route2模块 |
| Ollama实现 | `src/vlms/ollama_vl.py` | 本地模型支持 |

**配置项**（`src/config/settings.py`）：

```python
# VLM审计配置
vlm_enabled: bool = False
vlm_provider: str = "ollama"      # mock, ollama, qwen
vlm_trigger_level: str = "MODERATE"  # 触发等级

# Ollama配置
ollama_host: str = "http://localhost:11434"
ollama_model: str = "qwen2-vl"

# 通义千问配置
qwen_model: str = "qwen-vl-max"
```

**使用方式**：

```bash
cd route1

# 仅规则引擎（不启用VLM）
python main_rules.py --video meeting.mp4

# 启用VLM审计（本地Ollama）
python main_rules.py --video meeting.mp4 --vlm

# 启用VLM审计（云端通义千问）
python main_rules.py --video meeting.mp4 --vlm --vlm-provider qwen

# 自定义Ollama模型
python main_rules.py --video meeting.mp4 --vlm --ollama-model llava:13b

# 调整触发等级
python main_rules.py --video meeting.mp4 --vlm --vlm-trigger SEVERE
```

**测试结果**（100帧，mock模式）：

```
============================================================
📊 规则引擎检测报告
============================================================
总检测人次：594

异常等级分布:
  正常: 285 (48.0%)
  轻微: 291 (49.0%)
  中度: 11 (1.9%)
  重度: 7 (1.2%)

🤖 VLM审计次数: 18
============================================================

🤖 VLM审计复核汇总 (mock)
触发等级: MODERATE
总审计次数: 18

人员 P5:
  #1 触发原因: MODERATE (score=0.55)

人员 P6:
  #1-17 多次触发（MODERATE/SEVERE）
============================================================
```

**文件更新清单**：

| 文件 | 操作 | 说明 |
|------|------|------|
| `src/vlms/` | 🆕 复制 | 复用route2的VLM模块 |
| `src/vlms/ollama_vl.py` | 🆕 新建 | Ollama本地模型实现 |
| `src/config/settings.py` | ✏️ 修改 | 添加VLM配置项 |
| `src/audit/audit_system.py` | ✏️ 重写 | 集成真实VLM调用 |
| `main_rules.py` | ✏️ 修改 | 添加VLM参数 |
| `requirements.txt` | ✏️ 修改 | 添加requests依赖 |

---

## 命令参数完整列表

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--video` | data/videos/meeting_attention_video.mp4 | 视频路径 |
| `--model` | yolo11n-pose.pt | YOLO模型路径 |
| `--no-display` | False | 不显示窗口 |
| `--save` | False | 保存输出视频 |
| `--frames` | None | 最大处理帧数 |
| **`--vlm`** | False | **启用VLM审计** |
| **`--vlm-provider`** | ollama | **VLM提供者** (mock/ollama/qwen) |
| **`--vlm-trigger`** | MODERATE | **触发等级** (MILD/MODERATE/SEVERE) |
| **`--ollama-host`** | http://localhost:11434 | **Ollama服务地址** |
| **`--ollama-model`** | qwen2-vl | **Ollama模型** |

---

## VLM输出格式

```json
{
    "anomaly_type": "FATIGUE|APATHY|RUSHING|FRUSTRATION|NONE",
    "confidence": 0.85,
    "reasoning": "详细分析原因",
    "suggestions": "干预建议",
    "severity": "NORMAL|MILD|MODERATE|SEVERE"
}
```

---

## 更新时间
2026-05-03

## 版本历史
- v2.3: 集成VLM审计功能（本地Ollama + 云端通义千问）
- v2.2: 阈值参数优化
- v2.1: 俯仰角计算Bug修复
- v2.0: 路线1 MVP完成
