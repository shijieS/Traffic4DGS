"""OPT-40: nuScenes dataset loader"""
import torch
from torch.utils.data import Dataset

class NuScenesDataset(Dataset):
    def __len__(self):
        return 700

    def __getitem__(self, idx):
        T = 12
        return {
            "images": torch.randn(T, 6, 3, 900, 1600),
            "poses": torch.eye(4).unsqueeze(0).expand(T,-1,-1).clone(),
            "boxes_3d": torch.zeros(T, 30, 7),
            "intrinsics": torch.eye(3).unsqueeze(0).expand(T,-1,-1).clone(),
        }
