"""OPT-37: Cross-attention fusion of semantic and geometric features"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionFusion(nn.Module):
    def __init__(self, sem_dim=64, geo_dim=32, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.hd = (sem_dim + geo_dim) // n_heads
        self.q_proj = nn.Linear(sem_dim, n_heads * self.hd)
        self.k_proj = nn.Linear(geo_dim, n_heads * self.hd)
        self.v_proj = nn.Linear(geo_dim, n_heads * self.hd)
        self.out = nn.Linear(n_heads * self.hd, sem_dim)
        self.gate = nn.Sequential(nn.Linear(sem_dim*2, sem_dim), nn.Sigmoid())

    def forward(self, sem_feat, geo_feat):
        B, N, _ = sem_feat.shape
        Q = self.q_proj(sem_feat).view(B, N, self.n_heads, self.hd).transpose(1,2)
        K = self.k_proj(geo_feat).view(B, geo_feat.shape[1], self.n_heads, self.hd).transpose(1,2)
        V = self.v_proj(geo_feat).view(B, geo_feat.shape[1], self.n_heads, self.hd).transpose(1,2)
        attn = F.softmax((Q @ K.transpose(-2,-1)) / (self.hd**0.5), dim=-1)
        out = (attn @ V).transpose(1,2).contiguous().view(B, N, -1)
        out = self.out(out)
        g = self.gate(torch.cat([sem_feat, out], dim=-1))
        return sem_feat + g * out
