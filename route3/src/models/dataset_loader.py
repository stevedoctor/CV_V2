"""
数据集加载器

从 dataset/clips/ 和 dataset/annotations/ 加载训练数据
支持: 帧级标注 + 片段级标注, 数据增强, 变长序列padding
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
import os
import json

from dataset.scripts.data_augment import DataAugmentor


class AttentionDataset(Dataset):
    """注意力检测数据集"""
    
    def __init__(self,
                 clips_dir: str,
                 anno_dir: str,
                 split_file: Optional[str] = None,
                 clip_ids: Optional[List[str]] = None,
                 max_seq_len: int = 200,
                 augment: bool = False,
                 augment_ratio: float = 0.5,
                 target: str = "both"):
        """
        Args:
            clips_dir: clips目录路径
            anno_dir: annotations目录路径
            split_file: 划分文件路径 (train.txt/val.txt/test.txt)
            clip_ids: 直接指定clip_id列表 (优先于split_file)
            max_seq_len: 最大序列长度
            augment: 是否启用数据增强
            augment_ratio: 增强概率
            target: "frame" | "clip" | "both"
        """
        self.clips_dir = clips_dir
        self.anno_dir = anno_dir
        self.max_seq_len = max_seq_len
        self.augment = augment
        self.augment_ratio = augment_ratio
        self.target = target
        
        if clip_ids is not None:
            self.clip_ids = clip_ids
        elif split_file is not None:
            with open(split_file, 'r') as f:
                self.clip_ids = [line.strip() for line in f if line.strip()]
        else:
            self.clip_ids = self._scan_all_clips()
        
        self.samples = self._build_sample_index()
    
    def _scan_all_clips(self) -> List[str]:
        clip_ids = []
        if os.path.exists(self.clips_dir):
            for d in sorted(os.listdir(self.clips_dir)):
                if os.path.isdir(os.path.join(self.clips_dir, d)):
                    if os.path.exists(os.path.join(self.anno_dir, f"{d}.json")):
                        clip_ids.append(d)
        return clip_ids
    
    def _build_sample_index(self) -> List[Tuple[str, int]]:
        """构建 (clip_id, person_id) 索引"""
        samples = []
        for clip_id in self.clip_ids:
            anno_path = os.path.join(self.anno_dir, f"{clip_id}.json")
            if not os.path.exists(anno_path):
                continue
            with open(anno_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for p_key in data.get("persons", {}):
                pid = int(p_key.replace("P", ""))
                npy_base = os.path.join(self.clips_dir, clip_id, f"P{pid}")
                if os.path.exists(f"{npy_base}_frames.npy"):
                    samples.append((clip_id, pid))
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        clip_id, pid = self.samples[idx]
        npy_base = os.path.join(self.clips_dir, clip_id, f"P{pid}")
        
        keypoints = np.load(f"{npy_base}_frames.npy").astype(np.float32)
        rfac = np.load(f"{npy_base}_rfac.npy").astype(np.float32)
        valid = np.load(f"{npy_base}_valid.npy").astype(np.float32)
        frame_labels = np.load(f"{npy_base}_labels_frame.npy").astype(np.int64)
        clip_label = np.load(f"{npy_base}_label_clip.npy").astype(np.int64)
        
        if self.augment and np.random.random() < self.augment_ratio:
            kp_aug, _, rfac_aug = DataAugmentor.augment_pipeline(keypoints, frame_labels, rfac)
            keypoints = kp_aug
            rfac = rfac_aug
        
        T = min(len(keypoints), self.max_seq_len)
        keypoints = keypoints[:T]
        rfac = rfac[:T]
        valid = valid[:T]
        frame_labels = frame_labels[:T]
        
        B, K, C = keypoints.shape
        kp_flat = keypoints.reshape(T, K * C)
        
        return {
            "keypoints": torch.tensor(kp_flat, dtype=torch.float32),
            "keypoints_3d": torch.tensor(keypoints, dtype=torch.float32),
            "rfac": torch.tensor(rfac, dtype=torch.float32),
            "mask": torch.tensor(valid, dtype=torch.bool),
            "frame_labels": torch.tensor(frame_labels, dtype=torch.long),
            "clip_label": torch.tensor(clip_label[0], dtype=torch.long),
            "clip_id": clip_id,
            "person_id": pid,
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """变长序列padding + batch组装"""
    max_len = max(item["keypoints"].shape[0] for item in batch)
    B = len(batch)
    kp_dim = batch[0]["keypoints"].shape[-1]
    rfac_dim = batch[0]["rfac"].shape[-1]
    
    keypoints = torch.zeros(B, max_len, kp_dim)
    keypoints_3d = torch.zeros(B, max_len, 17, 3)
    rfac = torch.zeros(B, max_len, rfac_dim)
    mask = torch.zeros(B, max_len, dtype=torch.bool)
    frame_labels = torch.zeros(B, max_len, dtype=torch.long)
    clip_labels = torch.zeros(B, dtype=torch.long)
    clip_ids = []
    person_ids = []
    
    for i, item in enumerate(batch):
        T = item["keypoints"].shape[0]
        keypoints[i, :T] = item["keypoints"]
        keypoints_3d[i, :T] = item["keypoints_3d"]
        rfac[i, :T] = item["rfac"]
        mask[i, :T] = item["mask"]
        frame_labels[i, :T] = item["frame_labels"]
        clip_labels[i] = item["clip_label"]
        clip_ids.append(item["clip_id"])
        person_ids.append(item["person_id"])
    
    return {
        "keypoints": keypoints,
        "keypoints_3d": keypoints_3d,
        "rfac": rfac,
        "mask": mask,
        "frame_labels": frame_labels,
        "clip_labels": clip_labels,
        "clip_ids": clip_ids,
        "person_ids": person_ids,
    }


def create_dataloaders(dataset_dir: str,
                       batch_size: int = 32,
                       num_workers: int = 0,
                       augment_train: bool = True,
                       max_seq_len: int = 200
                       ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建train/val/test DataLoader
    
    Args:
        dataset_dir: 数据集根目录 (包含 clips/, annotations/, splits/)
        batch_size: batch大小
        num_workers: 数据加载线程数
        augment_train: 训练集是否增强
        max_seq_len: 最大序列长度
    """
    clips_dir = os.path.join(dataset_dir, "clips")
    anno_dir = os.path.join(dataset_dir, "annotations")
    splits_dir = os.path.join(dataset_dir, "splits")
    
    train_ds = AttentionDataset(
        clips_dir, anno_dir,
        split_file=os.path.join(splits_dir, "train.txt"),
        max_seq_len=max_seq_len,
        augment=augment_train,
        augment_ratio=0.5,
    )
    val_ds = AttentionDataset(
        clips_dir, anno_dir,
        split_file=os.path.join(splits_dir, "val.txt"),
        max_seq_len=max_seq_len,
        augment=False,
    )
    test_ds = AttentionDataset(
        clips_dir, anno_dir,
        split_file=os.path.join(splits_dir, "test.txt"),
        max_seq_len=max_seq_len,
        augment=False,
    )
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn,
        pin_memory=True, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
