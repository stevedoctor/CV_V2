"""
CNN-LSTM 训练脚本

支持: 帧级+片段级联合训练, 类别加权, 早停, 模型保存
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict
import numpy as np
import os
import sys
import json
import time
import argparse
from collections import Counter

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from src.models.cnn_lstm import AttentionLSTM
from src.models.dataset_loader import create_dataloaders, AttentionDataset


def compute_class_weights(dataset: AttentionDataset, num_classes: int = 4) -> torch.Tensor:
    """计算类别权重 (inverse frequency)"""
    counter = Counter()
    for i in range(len(dataset)):
        item = dataset[i]
        counter[int(item["clip_label"])] += 1
    
    total = sum(counter.values())
    weights = []
    for c in range(num_classes):
        count = counter.get(c, 1)
        weights.append(total / (num_classes * count))
    
    return torch.tensor(weights, dtype=torch.float32)


class FocalLoss(nn.Module):
    """Focal Loss for class imbalance"""
    
    def __init__(self, alpha: torch.Tensor = None, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal = (1 - pt) ** self.gamma * ce_loss
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            at = alpha[targets]
            focal = at * focal
        if self.reduction == 'mean':
            return focal.mean()
        elif self.reduction == 'sum':
            return focal.sum()
        return focal


def train_one_epoch(model: AttentionLSTM,
                    loader: DataLoader,
                    optimizer: optim.Optimizer,
                    frame_criterion: nn.Module,
                    clip_criterion: nn.Module,
                    device: torch.device,
                    frame_weight: float = 0.6,
                    clip_weight: float = 0.4) -> Dict[str, float]:
    model.train()
    total_loss = 0
    frame_correct = 0
    frame_total = 0
    clip_correct = 0
    clip_total = 0
    
    for batch in loader:
        keypoints = batch["keypoints_3d"].to(device)
        rfac = batch["rfac"].to(device)
        mask = batch["mask"].to(device)
        frame_labels = batch["frame_labels"].to(device)
        clip_labels = batch["clip_labels"].to(device)
        
        optimizer.zero_grad()
        
        frame_logits, clip_logits = model(keypoints, rfac, mask)
        
        T_out = frame_logits.shape[1]
        valid_mask = mask[:, :T_out].bool()
        if valid_mask.any():
            frame_pred = frame_logits[valid_mask]
            frame_tgt = frame_labels[:, :T_out][valid_mask]
            f_loss = frame_criterion(frame_pred, frame_tgt)
            frame_correct += (frame_pred.argmax(dim=-1) == frame_tgt).sum().item()
            frame_total += frame_tgt.numel()
        else:
            f_loss = torch.tensor(0.0, device=device)
        
        c_loss = clip_criterion(clip_logits, clip_labels)
        clip_correct += (clip_logits.argmax(dim=-1) == clip_labels).sum().item()
        clip_total += clip_labels.numel()
        
        loss = frame_weight * f_loss + clip_weight * c_loss
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    n = len(loader)
    return {
        "loss": total_loss / max(n, 1),
        "frame_acc": frame_correct / max(frame_total, 1),
        "clip_acc": clip_correct / max(clip_total, 1),
    }


@torch.no_grad()
def evaluate(model: AttentionLSTM,
             loader: DataLoader,
             frame_criterion: nn.Module,
             clip_criterion: nn.Module,
             device: torch.device,
             frame_weight: float = 0.6,
             clip_weight: float = 0.4) -> Dict[str, float]:
    model.eval()
    total_loss = 0
    frame_correct = 0
    frame_total = 0
    clip_correct = 0
    clip_total = 0
    
    level_correct = {0: 0, 1: 0, 2: 0, 3: 0}
    level_total = {0: 0, 1: 0, 2: 0, 3: 0}
    
    for batch in loader:
        keypoints = batch["keypoints_3d"].to(device)
        rfac = batch["rfac"].to(device)
        mask = batch["mask"].to(device)
        frame_labels = batch["frame_labels"].to(device)
        clip_labels = batch["clip_labels"].to(device)
        
        frame_logits, clip_logits = model(keypoints, rfac, mask)
        
        T_out = frame_logits.shape[1]
        valid_mask = mask[:, :T_out].bool()
        frame_labels_trimmed = frame_labels[:, :T_out]
        if valid_mask.any():
            frame_pred = frame_logits[valid_mask]
            frame_tgt = frame_labels_trimmed[valid_mask]
            f_loss = frame_criterion(frame_pred, frame_tgt)
            preds = frame_pred.argmax(dim=-1)
            frame_correct += (preds == frame_tgt).sum().item()
            frame_total += frame_tgt.numel()
            for lv in range(4):
                level_total[lv] += (frame_tgt == lv).sum().item()
                level_correct[lv] += ((preds == lv) & (frame_tgt == lv)).sum().item()
        else:
            f_loss = torch.tensor(0.0, device=device)
        
        c_loss = clip_criterion(clip_logits, clip_labels)
        clip_preds = clip_logits.argmax(dim=-1)
        clip_correct += (clip_preds == clip_labels).sum().item()
        clip_total += clip_labels.numel()
        
        loss = frame_weight * f_loss + clip_weight * c_loss
        total_loss += loss.item()
    
    n = len(loader)
    result = {
        "loss": total_loss / max(n, 1),
        "frame_acc": frame_correct / max(frame_total, 1),
        "clip_acc": clip_correct / max(clip_total, 1),
    }
    
    for lv in range(4):
        if level_total[lv] > 0:
            result[f"frame_acc_L{lv}"] = level_correct[lv] / level_total[lv]
    
    return result


def train(config: dict):
    device_str = config.get("device", "cuda")
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA不可用，回退到CPU")
        device_str = "cpu"
    device = torch.device(device_str)
    print(f"设备: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset_dir=config["dataset_dir"],
        batch_size=config["batch_size"],
        num_workers=config.get("num_workers", 0),
        augment_train=config.get("augment", True),
        max_seq_len=config.get("max_seq_len", 200),
    )
    
    print(f"训练集: {len(train_loader.dataset)} 样本")
    print(f"验证集: {len(val_loader.dataset)} 样本")
    print(f"测试集: {len(test_loader.dataset)} 样本")
    
    model = AttentionLSTM(
        kp_dim=51,
        rfac_dim=4,
        spatial_dim=config.get("spatial_dim", 128),
        lstm_hidden=config.get("lstm_hidden", 64),
        lstm_layers=config.get("lstm_layers", 2),
        num_classes=4,
        bidirectional=config.get("bidirectional", True),
        dropout=config.get("dropout", 0.3),
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数: 总计 {total_params:,}, 可训练 {trainable:,}")
    
    class_weights = compute_class_weights(train_loader.dataset).to(device)
    print(f"类别权重: {class_weights.tolist()}")
    
    frame_criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    clip_criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.get("lr", 1e-3),
        weight_decay=config.get("weight_decay", 1e-4),
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.get("epochs", 50), eta_min=1e-6
    )
    
    save_dir = config.get("save_dir", "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    
    best_val_acc = 0.0
    patience = config.get("patience", 10)
    patience_counter = 0
    
    frame_weight = config.get("frame_weight", 0.6)
    clip_weight = config.get("clip_weight", 0.4)
    epochs = config.get("epochs", 50)
    
    print(f"\n{'='*60}")
    print(f"开始训练: {epochs} epochs")
    print(f"{'='*60}")
    
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        
        train_metrics = train_one_epoch(
            model, train_loader, optimizer,
            frame_criterion, clip_criterion, device,
            frame_weight, clip_weight
        )
        
        val_metrics = evaluate(
            model, val_loader, frame_criterion, clip_criterion, device,
            frame_weight, clip_weight
        )
        
        scheduler.step()
        
        elapsed = time.time() - t0
        
        print(f"Epoch {epoch:3d}/{epochs} ({elapsed:.1f}s) | "
              f"Train L={train_metrics['loss']:.4f} FA={train_metrics['frame_acc']:.3f} CA={train_metrics['clip_acc']:.3f} | "
              f"Val L={val_metrics['loss']:.4f} FA={val_metrics['frame_acc']:.3f} CA={val_metrics['clip_acc']:.3f}")
        
        if val_metrics["clip_acc"] > best_val_acc:
            best_val_acc = val_metrics["clip_acc"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": best_val_acc,
                "val_metrics": val_metrics,
                "config": config,
            }, os.path.join(save_dir, "best_model.pth"))
            print(f"  → 保存最佳模型 (clip_acc={best_val_acc:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"\n早停: {patience} epochs 无改善")
            break
    
    print(f"\n{'='*60}")
    print(f"训练完成, 最佳验证 clip_acc: {best_val_acc:.4f}")
    print(f"{'='*60}")
    
    ckpt = torch.load(os.path.join(save_dir, "best_model.pth"), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    
    test_metrics = evaluate(
        model, test_loader, frame_criterion, clip_criterion, device,
        frame_weight, clip_weight
    )
    
    print(f"\n测试集结果:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  Frame Acc: {test_metrics['frame_acc']:.4f}")
    print(f"  Clip Acc: {test_metrics['clip_acc']:.4f}")
    for lv in range(4):
        key = f"frame_acc_L{lv}"
        if key in test_metrics:
            names = {0: "NORMAL", 1: "MILD", 2: "MODERATE", 3: "SEVERE"}
            print(f"  {names[lv]} Acc: {test_metrics[key]:.4f}")
    
    with open(os.path.join(save_dir, "test_results.json"), 'w') as f:
        json.dump(test_metrics, f, indent=2, default=str)
    
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CNN-LSTM 训练")
    parser.add_argument("--dataset-dir", type=str, default="dataset")
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--spatial-dim", type=int, default=128)
    parser.add_argument("--lstm-hidden", type=int, default=64)
    parser.add_argument("--lstm-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--max-seq-len", type=int, default=200)
    parser.add_argument("--no-augment", action="store_true")
    
    args = parser.parse_args()
    
    config = vars(args)
    config["augment"] = not config.pop("no_augment", False)
    config["frame_weight"] = 0.6
    config["clip_weight"] = 0.4
    config["weight_decay"] = 1e-4
    config["num_workers"] = 0
    
    train(config)
