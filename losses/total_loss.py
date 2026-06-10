"""OPT-49: Adaptive total loss with uncertainty weighting"""
import torch
import torch.nn as nn

class AdaptiveTotalLoss(nn.Module):
    def __init__(self, n_losses=8):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(n_losses))
        self.names = ["rgb","depth","semantic","silhouette","temporal","rigidity","tracking","regularization"]

    def forward(self, losses):
        total = torch.tensor(0.0, device=self.log_vars.device)
        weights = {}
        for i, name in enumerate(self.names):
            if name in losses:
                prec = torch.exp(-self.log_vars[i])
                total += prec/2 * losses[name] + self.log_vars[i]/2
                weights[name] = (prec/2).item()
        return total, weights
