"""OPT-36: Skeleton non-rigid prior for pedestrians"""
import torch
import torch.nn as nn

class SkeletonNonRigidPrior(nn.Module):
    def __init__(self, n_bones=17):
        super().__init__()
        self.n_bones = n_bones
        self.joint_rotations = nn.Parameter(torch.zeros(n_bones, 3))
        self.bone_lengths = nn.Parameter(torch.ones(n_bones))

    def forward(self, positions, bone_ids):
        deformed = torch.zeros_like(positions)
        for b in range(self.n_bones):
            m = bone_ids == b
            if not m.any():
                continue
            axis = self.joint_rotations[b]
            angle = axis.norm().clamp(min=1e-8)
            ax = axis / angle
            K = torch.tensor([[0,-ax[2],ax[1]],[ax[2],0,-ax[0]],[-ax[1],ax[0],0]], device=ax.device)
            R = torch.eye(3, device=ax.device) + torch.sin(angle)*K + (1-torch.cos(angle))*K@K
            deformed[m] = (R @ positions[m].T).T
        return deformed
