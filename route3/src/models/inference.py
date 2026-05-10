"""
CNN-LSTM 推理引擎

用于实时推理: 输入关键点+RFAC → 输出注意力等级
支持: 帧级/片段级预测, 滑动窗口推理, 与route1规则引擎集成
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
import os

from src.models.cnn_lstm import AttentionLSTM


class AttentionPredictor:
    """CNN-LSTM 推理器"""
    
    def __init__(self,
                 model_path: str = "checkpoints/best_model.pth",
                 device: str = "cpu",
                 spatial_dim: int = 128,
                 lstm_hidden: int = 64,
                 lstm_layers: int = 2):
        self.device = torch.device(device)
        
        self.model = AttentionLSTM(
            kp_dim=51,
            rfac_dim=4,
            spatial_dim=spatial_dim,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            num_classes=4,
            bidirectional=True,
            dropout=0.0,
        )
        
        if os.path.exists(model_path):
            ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt)
            self.model.load_state_dict(state_dict)
            print(f"[AttentionPredictor] 加载模型: {model_path}")
            if "val_metrics" in ckpt:
                m = ckpt["val_metrics"]
                print(f"  验证 clip_acc: {m.get('clip_acc', 'N/A')}")
        else:
            print(f"[AttentionPredictor] 模型文件不存在: {model_path}, 使用随机初始化")
        
        self.model.to(self.device)
        self.model.eval()
        
        self.level_names = {0: "NORMAL", 1: "MILD", 2: "MODERATE", 3: "SEVERE"}
    
    def predict(self,
                keypoints: np.ndarray,
                rfac: np.ndarray,
                valid_mask: Optional[np.ndarray] = None
                ) -> Dict[str, any]:
        """
        预测注意力等级
        
        Args:
            keypoints: (T, 17, 3) 关键点序列
            rfac: (T, 4) RFAC特征
            valid_mask: (T,) bool, 可选
        
        Returns:
            {"frame_levels": list[int], "frame_probs": np.ndarray,
             "clip_level": int, "clip_probs": np.ndarray}
        """
        T = len(keypoints)
        
        kp = torch.tensor(keypoints, dtype=torch.float32).unsqueeze(0).to(self.device)
        rf = torch.tensor(rfac, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        if valid_mask is not None:
            mask = torch.tensor(valid_mask, dtype=torch.bool).unsqueeze(0).to(self.device)
        else:
            mask = None
        
        with torch.no_grad():
            frame_logits, clip_logits = self.model(kp, rf, mask)
        
        frame_probs = torch.softmax(frame_logits[0], dim=-1).cpu().numpy()
        clip_probs = torch.softmax(clip_logits[0], dim=-1).cpu().numpy()
        
        frame_levels = frame_probs.argmax(axis=-1).tolist()
        clip_level = int(clip_probs.argmax())
        
        return {
            "frame_levels": frame_levels,
            "frame_probs": frame_probs,
            "clip_level": clip_level,
            "clip_probs": clip_probs,
            "clip_label": self.level_names[clip_level],
        }
    
    def predict_sliding_window(self,
                               keypoints: np.ndarray,
                               rfac: np.ndarray,
                               window_size: int = 120,
                               stride: int = 48,
                               valid_mask: Optional[np.ndarray] = None
                               ) -> Dict[str, any]:
        """
        滑动窗口推理 (长序列)
        
        对超过模型最大长度的序列，分段推理后拼接
        """
        T = len(keypoints)
        all_frame_probs = np.zeros((T, 4), dtype=np.float32)
        count = np.zeros(T, dtype=np.float32)
        
        for start in range(0, T, stride):
            end = min(start + window_size, T)
            
            kp_seg = keypoints[start:end]
            rf_seg = rfac[start:end]
            vm_seg = valid_mask[start:end] if valid_mask is not None else None
            
            result = self.predict(kp_seg, rf_seg, vm_seg)
            seg_len = end - start
            all_frame_probs[start:end] += result["frame_probs"][:seg_len]
            count[start:end] += 1
        
        count = np.maximum(count, 1)
        avg_probs = all_frame_probs / count[:, None]
        frame_levels = avg_probs.argmax(axis=-1).tolist()
        
        level_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for lv in frame_levels:
            level_counts[lv] = level_counts.get(lv, 0) + 1
        clip_level = max(level_counts, key=level_counts.get)
        
        return {
            "frame_levels": frame_levels,
            "frame_probs": avg_probs,
            "clip_level": clip_level,
            "clip_label": self.level_names[clip_level],
        }


class EnsemblePredictor:
    """规则引擎 + CNN-LSTM 集成推理"""
    
    def __init__(self,
                 model_path: str = "checkpoints/best_model.pth",
                 device: str = "cpu",
                 rule_weight: float = 0.5,
                 model_weight: float = 0.5,
                 conflict_strategy: str = "max"):
        """
        Args:
            rule_weight: 规则引擎权重
            model_weight: 模型权重
            conflict_strategy: "max"取较高等级 | "weighted"加权投票 | "model"模型优先
        """
        self.predictor = AttentionPredictor(model_path=model_path, device=device)
        self.rule_weight = rule_weight
        self.model_weight = model_weight
        self.conflict_strategy = conflict_strategy
        self.level_names = {0: "NORMAL", 1: "MILD", 2: "MODERATE", 3: "SEVERE"}
    
    def predict(self,
                rule_levels: List[int],
                keypoints: np.ndarray,
                rfac: np.ndarray,
                valid_mask: Optional[np.ndarray] = None
                ) -> Dict[str, any]:
        """
        集成预测
        
        Args:
            rule_levels: 规则引擎每帧等级 (T,)
            keypoints: (T, 17, 3)
            rfac: (T, 4)
            valid_mask: (T,) 可选
        
        Returns:
            {"frame_levels": list, "clip_level": int, "clip_label": str}
        """
        model_result = self.predictor.predict(keypoints, rfac, valid_mask)
        model_frame_levels = np.array(model_result["frame_probs"].argmax(axis=-1))
        model_frame_probs = model_result["frame_probs"]
        
        rule_arr = np.array(rule_levels)
        
        T = len(rule_arr)
        ensemble_levels = np.zeros(T, dtype=np.int64)
        
        for t in range(T):
            r_lv = rule_arr[t]
            m_lv = int(model_frame_levels[t])
            
            if r_lv == m_lv:
                ensemble_levels[t] = r_lv
            elif self.conflict_strategy == "max":
                ensemble_levels[t] = max(r_lv, m_lv)
            elif self.conflict_strategy == "weighted":
                r_score = self.rule_weight * _level_to_score(r_lv)
                m_score = self.model_weight * _level_to_score(m_lv)
                ensemble_levels[t] = _score_to_level(r_score + m_score)
            elif self.conflict_strategy == "model":
                ensemble_levels[t] = m_lv
            else:
                ensemble_levels[t] = max(r_lv, m_lv)
        
        level_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for lv in ensemble_levels:
            level_counts[int(lv)] = level_counts.get(int(lv), 0) + 1
        clip_level = max(level_counts, key=level_counts.get)
        
        return {
            "frame_levels": ensemble_levels.tolist(),
            "clip_level": int(clip_level),
            "clip_label": self.level_names[clip_level],
            "rule_clip_level": _majority_vote(rule_arr),
            "model_clip_level": model_result["clip_level"],
        }


def _level_to_score(level: int) -> float:
    return [0.0, 0.33, 0.66, 1.0][level]


def _score_to_level(score: float) -> int:
    if score < 0.165:
        return 0
    elif score < 0.495:
        return 1
    elif score < 0.83:
        return 2
    else:
        return 3


def _majority_vote(levels: np.ndarray) -> int:
    level_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for lv in levels:
        level_counts[int(lv)] = level_counts.get(int(lv), 0) + 1
    return max(level_counts, key=level_counts.get)
