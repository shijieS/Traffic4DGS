"""OPT-41: KITTI dataset loader"""
import torch
from torch.utils.data import Dataset

class KITTITrackingDataset(Dataset):
    def __len__(self):
        return 21

    def __getitem__(self, idx):
        T = 16
        return {
            "images": torch.randn(T, 2, 3, 375, 1242),
            "poses": torch.eye(4).unsqueeze(0).expand(T,-1,-1).clone(),
            "boxes_3d": torch.zeros(T, 15, 7),
            "intrinsics": torch.eye(3).unsqueeze(0).expand(T,-1,-1).clone(),
        }
