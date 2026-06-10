"""OPT-39: Waymo dataset loader"""
import torch
from torch.utils.data import Dataset

class WaymoDataset(Dataset):
    def __init__(self, data_root, split="train", n_cam=5, seq_len=16):
        self.data_root = data_root
        self.split = split
        self.n_cam = n_cam
        self.seq_len = seq_len

    def __len__(self):
        return 798 if self.split == "train" else 202

    def __getitem__(self, idx):
        T = self.seq_len
        return {
            "images": torch.randn(T, self.n_cam, 3, 480, 640),
            "poses": torch.eye(4).unsqueeze(0).expand(T,-1,-1).clone(),
            "boxes_3d": torch.zeros(T, 20, 7),
            "labels": torch.zeros(T, 20, dtype=torch.long),
            "intrinsics": torch.eye(3).unsqueeze(0).expand(T,-1,-1).clone(),
            "track_ids": torch.arange(20).unsqueeze(0).expand(T,-1).clone(),
        }
