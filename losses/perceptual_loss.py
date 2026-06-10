"""OPT-31: Perceptual loss"""
import torch
import torch.nn as nn

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3,64,3,1,1), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(64,128,3,1,1), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(128,256,3,1,1), nn.ReLU())
        self.pool = nn.AvgPool2d(2,2)
        self.s1 = nn.Parameter(torch.ones(1)*0.1)
        self.s2 = nn.Parameter(torch.ones(1)*0.1)
        self.s3 = nn.Parameter(torch.ones(1)*0.1)

    def forward(self, pred, target):
        f1_p, f1_t = self.conv1(pred), self.conv1(target)
        f2_p, f2_t = self.conv2(self.pool(f1_p)), self.conv2(self.pool(f1_t))
        f3_p, f3_t = self.conv3(self.pool(f2_p)), self.conv3(self.pool(f2_t))
        return self.s1*((f1_p-f1_t)**2).mean() + self.s2*((f2_p-f2_t)**2).mean() + self.s3*((f3_p-f3_t)**2).mean()
