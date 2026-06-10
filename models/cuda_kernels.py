"""OPT-54: CUDA kernel interface"""
import torch

class CUDARasterizerInterface:
    @staticmethod
    def rasterize(positions, scales, rotations, opacities, colors, cam_mat, img_size, tile_size=16):
        try:
            from cuda_ext import rasterize
            return rasterize(positions, scales, rotations, opacities, colors, cam_mat, img_size, tile_size)
        except ImportError:
            return CUDARasterizerInterface._fallback(positions, colors, cam_mat, img_size)

    @staticmethod
    def _fallback(positions, colors, cam_mat, img_size):
        H, W = img_size
        proj = cam_mat[:3,:3] @ positions.T + cam_mat[:3,3:4]
        image = torch.zeros(H, W, colors.shape[1], device=colors.device)
        return image
