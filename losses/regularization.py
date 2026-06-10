"""
Regularization Loss Functions.

This module implements regularization losses for Gaussian properties
including scale, opacity, rotation, and motion smoothness.

Mathematical Foundation:
    Scale Regularization:
        L_scale = ||Σ - Σ_target||²
        
    Opacity Regularization:
        L_opacity = -Σ αᵢ log(αᵢ) + ||α - α_target||²
        
    Rotation Regularization:
        L_rotation = ||RᵀR - I||² (orthogonality)
        
    Motion Smoothness:
        L_motion = ||vₜ - vₜ₋₁||² (velocity continuity)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List


class GaussianRegularizationLoss(nn.Module):
    """Regularization for Gaussian primitive properties.
    
    Includes:
    - Scale regularization
    - Opacity regularization
    - Rotation orthonormalization
    """
    
    def __init__(
        self,
        scale_weight: float = 0.01,
        opacity_weight: float = 0.001,
        rotation_weight: float = 0.01,
        target_scale: float = 0.01,
        target_opacity: float = 0.5,
        scale_range: Tuple[float, float] = (0.001, 1.0),
    ) -> None:
        """Initialize Gaussian regularization loss.
        
        Args:
            scale_weight: Weight for scale regularization
            opacity_weight: Weight for opacity regularization
            rotation_weight: Weight for rotation regularization
            target_scale: Target Gaussian scale
            target_opacity: Target opacity
            scale_range: Valid scale range
        """
        super().__init__()
        self.scale_weight = scale_weight
        self.opacity_weight = opacity_weight
        self.rotation_weight = rotation_weight
        self.target_scale = target_scale
        self.target_opacity = target_opacity
        self.scale_range = scale_range
    
    def forward(
        self,
        scales: torch.Tensor,
        opacities: torch.Tensor,
        rotations: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute Gaussian regularization loss.
        
        Args:
            scales: Log scales [N, 3]
            opacities: Log opacities [N, 1]
            rotations: Quaternions [N, 4]
            mask: Optional mask for active Gaussians
            
        Returns:
            loss: Regularization loss
            info: Loss breakdown
        """
        info = {}
        losses = []
        
        # Scale regularization
        if self.scale_weight > 0:
            scale_values = torch.exp(scales)
            
            # Clamp to valid range
            scale_values = torch.clamp(
                scale_values,
                min=self.scale_range[0],
                max=self.scale_range[1]
            )
            
            # L2 to target
            loss_scale = ((scale_values - self.target_scale) ** 2).mean()
            losses.append(self.scale_weight * loss_scale)
            info['scale_reg'] = loss_scale.item()
        
        # Opacity regularization
        if self.opacity_weight > 0:
            opacity_values = torch.sigmoid(opacities)
            
            # Entropy regularization (prevent all visible/invisible)
            entropy = -(
                opacity_values * torch.log(opacity_values + 1e-8) +
                (1 - opacity_values) * torch.log(1 - opacity_values + 1e-8)
            )
            loss_entropy = entropy.mean()
            
            # L2 to target
            loss_opacity = ((opacity_values - self.target_opacity) ** 2).mean()
            
            loss_op = loss_entropy + 0.1 * loss_opacity
            losses.append(self.opacity_weight * loss_op)
            info['opacity_reg'] = loss_op.item()
        
        # Rotation regularization (orthonormality)
        if self.rotation_weight > 0:
            rot_mats = self._quaternion_to_matrix(rotations)  # [N, 3, 3]
            
            # R @ R^T should be identity
            rt_r = torch.matmul(rot_mats, rot_mats.transpose(-2, -1))
            I = torch.eye(3, device=rot_mats.device).unsqueeze(0)
            loss_ortho = ((rt_r - I) ** 2).mean()
            
            # Determinant should be +1
            det_r = torch.det(rot_mats)
            loss_det = ((det_r - 1) ** 2).mean()
            
            loss_rot = loss_ortho + 0.1 * loss_det
            losses.append(self.rotation_weight * loss_rot)
            info['rotation_reg'] = loss_rot.item()
        
        total_loss = sum(losses) if losses else torch.tensor(0.0)
        return total_loss, info
    
    def _quaternion_to_matrix(self, q: torch.Tensor) -> torch.Tensor:
        """Convert quaternion to rotation matrix."""
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        
        R = torch.zeros(q.shape[0], 3, 3, device=q.device)
        R[:, 0, 0] = 1 - 2*(y*y + z*z)
        R[:, 0, 1] = 2*(x*y - z*w)
        R[:, 0, 2] = 2*(x*z + y*w)
        R[:, 1, 0] = 2*(x*y + z*w)
        R[:, 1, 1] = 1 - 2*(x*x + z*z)
        R[:, 1, 2] = 2*(y*z - x*w)
        R[:, 2, 0] = 2*(x*z - y*w)
        R[:, 2, 1] = 2*(y*z + x*w)
        R[:, 2, 2] = 1 - 2*(x*x + y*y)
        
        return R


class MotionSmoothnessLoss(nn.Module):
    """Motion smoothness regularization for dynamic objects.
    
    Encourages continuous and smooth motion trajectories.
    """
    
    def __init__(
        self,
        velocity_weight: float = 0.01,
        acceleration_weight: float = 0.05,
        angular_weight: float = 0.01,
    ) -> None:
        """Initialize motion smoothness loss.
        
        Args:
            velocity_weight: Weight for velocity regularization
            acceleration_weight: Weight for acceleration regularization
            angular_weight: Weight for angular velocity regularization
        """
        super().__init__()
        self.velocity_weight = velocity_weight
        self.acceleration_weight = acceleration_weight
        self.angular_weight = angular_weight
    
    def forward(
        self,
        positions: torch.Tensor,
        rotations: Optional[torch.Tensor] = None,
        dt: float = 1.0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute motion smoothness loss.
        
        Args:
            positions: Position trajectory [T, N, 3] or [T, 3]
            rotations: Optional rotation quaternions [T, N, 4] or [T, 4]
            dt: Time step
            
        Returns:
            loss: Motion smoothness loss
            info: Loss breakdown
        """
        info = {}
        losses = []
        
        if positions.dim() == 2:
            positions = positions.unsqueeze(1)
        if rotations is not None and rotations.dim() == 2:
            rotations = rotations.unsqueeze(1)
        
        T, N, _ = positions.shape
        
        # Velocity
        velocity = torch.diff(positions, dim=0) / dt  # [T-1, N, 3]
        
        # Velocity regularization (small velocity)
        if self.velocity_weight > 0:
            loss_velocity = (velocity ** 2).mean()
            losses.append(self.velocity_weight * loss_velocity)
            info['velocity'] = loss_velocity.item()
        
        # Acceleration regularization (smooth velocity changes)
        if self.acceleration_weight > 0 and T > 2:
            acceleration = torch.diff(velocity, dim=0) / dt  # [T-2, N, 3]
            loss_accel = (acceleration ** 2).mean()
            losses.append(self.acceleration_weight * loss_accel)
            info['acceleration'] = loss_accel.item()
        
        # Angular velocity regularization
        if self.angular_weight > 0 and rotations is not None and T > 1:
            rot_mats = self._quaternion_to_matrix(rotations)  # [T, N, 3, 3]
            
            # Relative rotations
            relative_rot = torch.matmul(rot_mats[1:], rot_mats[:-1].transpose(-2, -1))
            
            # Angular velocity (from rotation matrix)
            trace = relative_rot[:, :, 0, 0] + relative_rot[:, :, 1, 1] + relative_rot[:, :, 2, 2]
            angle = torch.acos(torch.clamp((trace - 1) / 2, -1, 1))
            
            loss_angular = (angle ** 2).mean()
            losses.append(self.angular_weight * loss_angular)
            info['angular'] = loss_angular.item()
        
        total_loss = sum(losses) if losses else torch.tensor(0.0, device=positions.device)
        return total_loss, info
    
    def _quaternion_to_matrix(self, q: torch.Tensor) -> torch.Tensor:
        """Convert quaternion to rotation matrix."""
        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        
        R = torch.zeros(*q.shape[:-1], 3, 3, device=q.device)
        R[..., 0, 0] = 1 - 2*(y*y + z*z)
        R[..., 0, 1] = 2*(x*y - z*w)
        R[..., 0, 2] = 2*(x*z + y*w)
        R[..., 1, 0] = 2*(x*y + z*w)
        R[..., 1, 1] = 1 - 2*(x*x + z*z)
        R[..., 1, 2] = 2*(y*z - x*w)
        R[..., 2, 0] = 2*(x*z - y*w)
        R[..., 2, 1] = 2*(y*z + x*w)
        R[..., 2, 2] = 1 - 2*(x*x + y*y)
        
        return R


class DiffeomorphismRegularizationLoss(nn.Module):
    """Regularization for diffeomorphic deformations.
    
    Ensures deformations are smooth and invertible.
    """
    
    def __init__(
        self,
        jacobian_weight: float = 0.1,
        curl_weight: float = 0.05,
        divergence_weight: float = 0.05,
    ) -> None:
        """Initialize diffeomorphism regularization.
        
        Args:
            jacobian_weight: Weight for Jacobian determinant constraint
            curl_weight: Weight for curl (solenoidal field)
            divergence_weight: Weight for divergence-free constraint
        """
        super().__init__()
        self.jacobian_weight = jacobian_weight
        self.curl_weight = curl_weight
        self.divergence_weight = divergence_weight
    
    def forward(
        self,
        displacement: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute diffeomorphism regularization loss.
        
        Args:
            displacement: Deformation vector field [B, 3, H, W, D] or [B, 3, H, W]
            positions: Optional position grid
            
        Returns:
            loss: Diffeomorphism loss
            info: Loss breakdown
        """
        info = {}
        losses = []
        
        # Displacement smoothness
        loss_smooth = self._compute_smoothness(displacement)
        if loss_smooth is not None:
            losses.append(0.01 * loss_smooth)
            info['smooth'] = loss_smooth.item()
        
        # Jacobian determinant close to 1 (for invertibility)
        if self.jacobian_weight > 0 and positions is not None:
            loss_jac = self._compute_jacobian_det_loss(displacement, positions)
            losses.append(self.jacobian_weight * loss_jac)
            info['jacobian'] = loss_jac.item()
        
        # Divergence-free (incompressible flow)
        if self.divergence_weight > 0:
            loss_div = self._compute_divergence(displacement)
            losses.append(self.divergence_weight * loss_div)
            info['divergence'] = loss_div.item()
        
        # Curl-free (potential flow)
        if self.curl_weight > 0:
            loss_curl = self._compute_curl(displacement)
            losses.append(self.curl_weight * loss_curl)
            info['curl'] = loss_curl.item()
        
        total_loss = sum(losses) if losses else torch.tensor(0.0, device=displacement.device)
        return total_loss, info
    
    def _compute_smoothness(self, x: torch.Tensor) -> torch.Tensor:
        """Compute smoothness (TV) loss."""
        gy = torch.diff(x, dim=-2 if x.dim() == 4 else -3)
        gx = torch.diff(x, dim=-1 if x.dim() == 4 else -2)
        return torch.mean(torch.abs(gy)) + torch.mean(torch.abs(gx))
    
    def _compute_jacobian_det_loss(
        self,
        displacement: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Jacobian determinant loss."""
        # Approximate Jacobian
        grad = torch.autograd.grad(
            outputs=displacement,
            inputs=positions,
            grad_outputs=torch.ones_like(displacement),
            create_graph=True,
        )[0]
        
        # Identity + Jacobian for deformation mapping
        I_plus_J = torch.eye(3, device=displacement.device).unsqueeze(0) + grad
        
        # Determinant should be 1
        det = torch.det(I_plus_J)
        loss = ((det - 1) ** 2).mean()
        
        return loss
    
    def _compute_divergence(self, x: torch.Tensor) -> torch.Tensor:
        """Compute divergence of vector field."""
        if x.dim() == 4:  # [B, 3, H, W]
            dx = x[:, 0]
            dy = x[:, 1]
            dz = x[:, 2] if x.shape[1] > 2 else torch.zeros_like(dx)
        else:
            return torch.tensor(0.0)
        
        div_x = torch.diff(dx, dim=-2)
        div_y = torch.diff(dy, dim=-1)
        
        if dz is not None and dz.dim() > 2:
            div_z = torch.diff(dz, dim=-3)
            div = div_x + div_y + div_z
        else:
            div = div_x + div_y
        
        return (div ** 2).mean()
    
    def _compute_curl(self, x: torch.Tensor) -> torch.Tensor:
        """Compute curl of vector field."""
        if x.shape[1] != 3:
            return torch.tensor(0.0, device=x.device)
        
        u, v, w = x[:, 0], x[:, 1], x[:, 2]
        
        # Curl components
        dw_dy = torch.diff(w, dim=-2)
        dv_dz = torch.zeros_like(dw_dy)
        du_dz = torch.zeros_like(dw_dy)
        dw_dx = torch.diff(w, dim=-1)
        dv_dx = torch.diff(v, dim=-1)
        du_dy = torch.diff(u, dim=-2)
        
        curl_x = dw_dy - dv_dz
        curl_y = du_dz - dw_dx
        curl_z = dv_dx - du_dy
        
        curl = torch.sqrt(curl_x ** 2 + curl_y ** 2 + curl_z ** 2)
        
        return curl.mean()
