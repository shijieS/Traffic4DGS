"""OPT-53: Multi-view reprojection"""
import torch

def reproject(pts_3d, K, pose):
    w2c = torch.inverse(pose)
    pts_cam = (w2c[:3,:3] @ pts_3d.T).T + w2c[:3,3]
    depths = pts_cam[:,2]
    pts_2d = (K @ pts_cam.T).T
    pts_2d = pts_2d[:,:2] / pts_2d[:,2:3].clamp(min=1e-6)
    return pts_2d, depths

def reprojection_loss(pts_3d, K, poses, target_2d, visibility):
    err = 0.0
    n = 0
    for v in range(K.shape[0]):
        p2d, _ = reproject(pts_3d, K[v], poses[v])
        e = ((p2d - target_2d[v])**2).sum(-1)
        vis = visibility[v].bool()
        if vis.any():
            err += e[vis].sum()
            n += vis.sum().item()
    return err / max(n, 1)
