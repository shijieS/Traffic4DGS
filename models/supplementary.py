"""OPT-55: Supplementary derivations"""
import torch, math

class SE3Derivations:
    @staticmethod
    def verify_exp_map():
        xi = torch.randn(6)
        o, v = xi[:3], xi[3:]
        th = o.norm()
        oh = o / th.clamp(min=1e-8)
        K = torch.tensor([[0,-oh[2],oh[1]],[oh[2],0,-oh[0]],[-oh[1],oh[0],0]])
        R = torch.eye(3) + math.sin(th)*K + (1-math.cos(th))*K@K
        return (R.T@R - torch.eye(3)).abs().max() < 1e-5 and (R.det()-1).abs() < 1e-5

    @staticmethod
    def verify_rigidity():
        pts = torch.randn(10, 3)
        R, _ = torch.linalg.qr(torch.randn(3,3))
        t = torch.randn(3)
        d0 = torch.cdist(pts, pts)
        d1 = torch.cdist((R@pts.T).T + t, (R@pts.T).T + t)
        return (d0-d1).abs().max() < 1e-4
