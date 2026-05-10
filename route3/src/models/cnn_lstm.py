"""
CNN-LSTM 注意力检测模型

输入: 关键点序列 (T, 17, 3) + RFAC特征 (T, 4)
输出: 帧级注意力等级 (T, 4) + 片段级注意力等级 (4)

架构:
  Spatial MLP: (17*3+4=55) → 128 → 128
  BiLSTM: 128 → 64*2=128
  Frame Head: 128 → 4 (帧级分类)
  Clip Head: 128 → 4 (片段级分类, 从LSTM最后隐状态)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SpatialEncoder(nn.Module):
    """逐帧空间编码器: keypoints + rfac → feature"""
    
    def __init__(self, kp_dim: int = 51, rfac_dim: int = 4, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(kp_dim + rfac_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, keypoints: torch.Tensor, rfac: torch.Tensor) -> torch.Tensor:
        """
        Args:
            keypoints: (B, T, 17, 3) or (B, T, 51)
            rfac: (B, T, 4)
        Returns:
            (B, T, hidden_dim)
        """
        if keypoints.dim() == 4:
            B, T = keypoints.shape[:2]
            kp = keypoints.reshape(B, T, -1)
        else:
            B, T = keypoints.shape[:2]
            kp = keypoints
        
        x = torch.cat([kp, rfac], dim=-1)  # (B, T, 55)
        B, T, D = x.shape
        x = x.reshape(B * T, D)
        x = self.net(x)
        x = x.reshape(B, T, -1)
        return x


class AttentionLSTM(nn.Module):
    """CNN-LSTM 注意力检测模型"""
    
    def __init__(self,
                 kp_dim: int = 51,
                 rfac_dim: int = 4,
                 spatial_dim: int = 128,
                 lstm_hidden: int = 64,
                 lstm_layers: int = 2,
                 num_classes: int = 4,
                 bidirectional: bool = True,
                 dropout: float = 0.3):
        super().__init__()
        
        self.spatial = SpatialEncoder(kp_dim, rfac_dim, spatial_dim)
        
        self.lstm = nn.LSTM(
            input_size=spatial_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if lstm_layers > 1 else 0.0
        )
        
        lstm_out_dim = lstm_hidden * (2 if bidirectional else 1)
        
        self.frame_head = nn.Sequential(
            nn.Linear(lstm_out_dim, lstm_out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim, num_classes)
        )
        
        self.clip_head = nn.Sequential(
            nn.Linear(lstm_out_dim, lstm_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)
                hidden_size = param.data.shape[0] // 4
                param.data[hidden_size:2*hidden_size].fill_(1.0)
    
    def forward(self,
                keypoints: torch.Tensor,
                rfac: torch.Tensor,
                mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            keypoints: (B, T, 17, 3) or (B, T, 51)
            rfac: (B, T, 4)
            mask: (B, T) bool, True=valid
        
        Returns:
            frame_logits: (B, T, 4)
            clip_logits: (B, 4)
        """
        spatial_feat = self.spatial(keypoints, rfac)  # (B, T, spatial_dim)
        
        if mask is not None:
            lengths = mask.sum(dim=1).long()
            packed = nn.utils.rnn.pack_padded_sequence(
                spatial_feat, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            lstm_out, (h_n, c_n) = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, (h_n, c_n) = self.lstm(spatial_feat)
        
        frame_logits = self.frame_head(lstm_out)  # (B, T, 4)
        
        if self.lstm.bidirectional:
            h_fwd = h_n[-2]
            h_bwd = h_n[-1]
            clip_feat = torch.cat([h_fwd, h_bwd], dim=-1)
        else:
            clip_feat = h_n[-1]
        
        clip_logits = self.clip_head(clip_feat)  # (B, 4)
        
        return frame_logits, clip_logits
    
    def predict_frame(self,
                      keypoints: torch.Tensor,
                      rfac: torch.Tensor,
                      mask: Optional[torch.Tensor] = None
                      ) -> torch.Tensor:
        """预测帧级标签 (inference用)"""
        self.eval()
        with torch.no_grad():
            frame_logits, _ = self.forward(keypoints, rfac, mask)
            return frame_logits.argmax(dim=-1)
    
    def predict_clip(self,
                     keypoints: torch.Tensor,
                     rfac: torch.Tensor,
                     mask: Optional[torch.Tensor] = None
                     ) -> torch.Tensor:
        """预测片段级标签 (inference用)"""
        self.eval()
        with torch.no_grad():
            _, clip_logits = self.forward(keypoints, rfac, mask)
            return clip_logits.argmax(dim=-1)
