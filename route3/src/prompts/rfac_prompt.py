"""
RFAC四维度Prompt模板

路线2使用，支持：
1. 全局分析Prompt（整帧所有人）
2. 个体分析Prompt（裁剪的单人图片）
"""


class RFACPromptBuilder:
    """
    RFAC Prompt构建器
    
    生成用于分析会议注意力的标准化Prompt
    """
    
    @staticmethod
    def build_individual_prompt(person_id: int = None) -> str:
        """
        构建个体分析Prompt（核心）
        
        用于裁剪后的单人图片，精确分析单个人员的注意力状态
        
        Args:
            person_id: 人员ID（可选）
            
        Returns:
            Prompt字符串
        """
        id_info = f"（人员编号: P{person_id}）" if person_id is not None else ""
        
        return f"""你是一位专业的会议/班前会行为分析师。请分析这些视频帧中该人员{ id_info}的注意力状态。

## 分析维度（RFAC模型）

请从以下四个维度进行评估：

### 1. Apathy（注意力涣散）- 权重55%
- 该人员是否正视前方/发言人
- 眼神是否聚焦，还是游离、发呆
- 是否存在频繁转头、东张西望
- 评分范围：0-1，0=完全专注，1=严重涣散

### 2. Fatigue（疲劳状态）- 权重25%
- 头部是否下垂或频繁低头
- 身体是否松垮、前倾或后仰
- 是否有打哈欠、闭眼等嗜睡表现
- 评分范围：0-1，0=精神饱满，1=极度疲劳

### 3. Rushing（匆忙状态）- 权重15%
- 肢体动作是否过快或频率异常
- 是否有焦躁不安、频繁挪动身体的表现
- 评分范围：0-1，0=从容淡定，1=极度匆忙焦躁

### 4. Frustration（情绪失控）- 权重5%
- 面部表情是否紧张、愤怒或沮丧
- 手势是否异常（如握拳、烦躁摸脸）
- 评分范围：0-1，0=情绪平稳，1=情绪失控

## 综合评分计算
overall_score = 0.55 × apathy + 0.25 × fatigue + 0.15 × rushing + 0.05 × frustration

## 等级划分
- NORMAL (overall < 0.3): 状态良好，专注
- MILD (0.3 ≤ overall < 0.5): 轻微异常，值得关注
- MODERATE (0.5 ≤ overall < 0.7): 中度异常，需要关注
- SEVERE (overall ≥ 0.7): 重度异常，建议立即干预

## 输出要求
重要：只输出JSON，不要输出任何其他文字，不要输出思考过程。
请返回JSON格式（不要包含```标记）：
{{"apathy_score": 0.0, "fatigue_score": 0.0, "rushing_score": 0.0, "frustration_score": 0.0, "overall_score": 0.0, "attention_level": "NORMAL或MILD或MODERATE或SEVERE", "reasoning": "详细分析原因", "suggestions": "干预建议"}}"""
    
    @staticmethod
    def build_global_prompt() -> str:
        """
        构建全局分析Prompt（整帧）
        
        用于整帧图片，分析所有可见人员
        
        Returns:
            Prompt字符串
        """
        return """你是一位专业的会议/班前会行为分析师。请分析这组视频帧中参会人员的整体注意力状态。

## 分析维度（RFAC模型）

1. Apathy（注意力涣散）- 权重55%: 观察整体正视率、眼神聚焦度
2. Fatigue（疲劳状态）- 权重25%: 观察整体头部姿态、身体松垮情况
3. Rushing（匆忙状态）- 权重15%: 观察整体肢体动作频率
4. Frustration（情绪失控）- 权重5%: 观察整体面部表情和手势

## 等级划分
- NORMAL (overall < 0.3): 整体状态良好
- MILD (0.3 ≤ overall < 0.5): 部分人员轻微异常
- MODERATE (0.5 ≤ overall < 0.7): 多人中度异常
- SEVERE (overall ≥ 0.7): 多人重度异常

## 输出要求
重要：只输出JSON，不要输出任何其他文字，不要输出思考过程。
请返回JSON格式（不要包含```标记）：
{"apathy_score": 0.0, "fatigue_score": 0.0, "rushing_score": 0.0, "frustration_score": 0.0, "overall_score": 0.0, "attention_level": "NORMAL或MILD或MODERATE或SEVERE", "num_persons": 人数, "reasoning": "详细分析", "suggestions": "干预建议"}"""
    
    @staticmethod
    def build_prompt() -> str:
        """兼容旧接口，默认返回个体分析Prompt"""
        return RFACPromptBuilder.build_individual_prompt()
