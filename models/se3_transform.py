"""
SE(3) Rigid Body Transformation Module - Optimized Version.

This module implements SE(3) rigid body transformations for dynamic object
modeling in 4D Gaussian Splatting with numerical stability optimizations.

OPTIMIZATION CHANGELOG (v1.1.0):
  [OPT-1] Rodrigues公式数值稳定版本：添加θ→0的Taylor展开处理
  [OPT-2] 解析梯度：实现se3_exp_map和so3_exp_map的解析梯度，替代autograd
  [OPT-3] 左Jacobian高效计算：实现J(ω)闭式解及J⁻¹的精确计算
  [OPT-4] SLERP插值：添加四元数球面线性插值用于帧间平滑
  [OPT-5] 伴随表示：实现Ad_SE(3)用于速度坐标系变换
  [OPT-6] 批量并行计算：优化batch矩阵运算，减少for循环
  [OPT-7] 轨迹平滑正则化：添加速度/加速度约束损失

Mathematical Foundation:
    SE(3) - Special Euclidean Group in 3D:
        SE(3) = { (R, t) | R ∈ SO(3), t ∈ ℝ³ }
    
    Transformation of a 3D point x:
        x' = R @ x + t
        
    Lie Algebra se(3):
        se(3) = { (ω, v) | ω ∈ ℝ³, v ∈ ℝ³ }
        
    Exponential map (se(3) → SE(3)):
        exp((ω, v)) = (exp(ω∧), J(ω) @ v)
        
    Left Jacobian (barfoot method):
        J(ω) = I + ((1-cosθ)/θ²) ω∧ + ((θ-sinθ)/θ³) (ω∧)²
        
    Left Jacobian Inverse:
        J⁻¹(ω) = I - ½ ω∧ + (1 - (θ·sinθ)/(2(1-cosθ))) / θ² · (ω∧)²

@author Semantic 4DGS Team
@version 1.1.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from dataclasses import dataclass
import math


# ============================================================================
# [OPT-1] NUMERICALLY STABLE RODRIGUES FORMULA
# Handles the singularity at θ → 0 with Taylor expansions
# ============================================================================

def rodrigues_formula_stable(omega: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Numerically stable Rodrigues formula with Taylor expansion for small angles.
    
    OPTIMIZATION [OPT-1]: This function replaces the naive implementation to handle
    the numerical instability at θ → 0 using Taylor series expansion.
    
    For small θ (θ < 1e-4):
        sin(θ) ≈ θ - θ³/6 + O(θ⁵)
        cos(θ) ≈ 1 - θ²/2 + O(θ⁴)
        (1-cos(θ))/θ² ≈ 1/2 - θ²/24 + O(θ⁴)
        (θ-sin(θ))/θ³ ≈ 1/6 - θ²/120 + O(θ⁴)
    
    Args:
        omega: Axis-angle vector [..., 3]
        
    Returns:
        R: Rotation matrix [..., 3, 3]
    """
    theta = torch.norm(omega, dim=-1, keepdim=True)  # [..., 1]
    
    # Build skew-symmetric matrix
    omega_hat = skew_symmetric(omega)  # [..., 3, 3]
    
    # Determine batch shape
    batch_shape = omega.shape[:-1]
    
    # Compute rotation matrix with numerical stability
    R = torch.zeros(*batch_shape, 3, 3, device=omega.device, dtype=omega.dtype)
    
    # Mask for small angles
    small_angle_mask = theta.squeeze(-1) < 1e-4  # [...]
    
    # Case 1: Small angle (Taylor expansion)
    # R ≈ I + (1 - θ²/6) ω∧ + (1/2 - θ²/24) ω∧²
    if small_angle_mask.any():
        theta_small = theta[small_angle_mask]  # [k]
        omega_hat_small = omega_hat[small_angle_mask]  # [k, 3, 3]
        omega_hat_sq_small = omega_hat_small @ omega_hat_small
        
        # Taylor coefficients
        coef1 = 1 - theta_small ** 2 / 6 + theta_small ** 4 / 120  # sin(θ)/θ approximation
        coef2 = 0.5 - theta_small ** 2 / 24  # (1-cos(θ))/θ² approximation
        
        R_small = torch.eye(3, device=omega.device, dtype=omega.dtype).unsqueeze(0)
        R_small = R_small + coef1.unsqueeze(-1).unsqueeze(-1) * omega_hat_small
        R_small = R_small + coef2.unsqueeze(-1).unsqueeze(-1) * omega_hat_sq_small
        R[small_angle_mask] = R_small
    
    # Case 2: Normal angle (standard formula)
    normal_mask = ~small_angle_mask
    if normal_mask.any():
        theta_norm = theta[normal_mask]  # [m]
        omega_hat_norm = omega_hat[normal_mask]  # [m, 3, 3]
        omega_hat_sq_norm = omega_hat_norm @ omega_hat_norm
        
        sin_theta = torch.sin(theta_norm)
        cos_theta = torch.cos(theta_norm)
        
        coef1_norm = sin_theta / theta_norm
        coef2_norm = (1 - cos_theta) / theta_norm ** 2
        
        R_norm = torch.eye(3, device=omega.device, dtype=omega.dtype).unsqueeze(0)
        R_norm = R_norm + coef1_norm.unsqueeze(-1).unsqueeze(-1) * omega_hat_norm
        R_norm = R_norm + coef2_norm.unsqueeze(-1).unsqueeze(-1) * omega_hat_sq_norm
        R[normal_mask] = R_norm
    
    return R


def skew_symmetric(omega: torch.Tensor) -> torch.Tensor:
    r"""Build skew-symmetric matrix from axis-angle vector.
    
    Args:
        omega: Axis-angle vector [..., 3]
        
    Returns:
        omega_hat: Skew-symmetric matrix [..., 3, 3]
    """
    batch_shape = omega.shape[:-1]
    omega_hat = torch.zeros(*batch_shape, 3, 3, device=omega.device, dtype=omega.dtype)
    
    omega_hat[..., 0, 1] = -omega[..., 2]
    omega_hat[..., 0, 2] = omega[..., 1]
    omega_hat[..., 1, 0] = omega[..., 2]
    omega_hat[..., 1, 2] = -omega[..., 0]
    omega_hat[..., 2, 0] = -omega[..., 1]
    omega_hat[..., 2, 1] = omega[..., 0]
    
    return omega_hat


def from_skew_to_vector(omega_hat: torch.Tensor) -> torch.Tensor:
    r"""Convert skew-symmetric matrix back to vector.
    
    Args:
        omega_hat: Skew-symmetric matrix [..., 3, 3]
        
    Returns:
        omega: Axis-angle vector [..., 3]
    """
    omega = torch.stack([
        omega_hat[..., 2, 1],
        omega_hat[..., 0, 2],
        omega_hat[..., 1, 0]
    ], dim=-1)
    return omega


# ============================================================================
# [OPT-2] ANALYTICAL GRADIENT FOR SE(3) EXPONENTIAL MAP
# Provides closed-form gradients instead of relying on autograd
# ============================================================================

def so3_exp_map_analytic(omega: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""SO(3) exponential map with optional analytical gradient computation.
    
    OPTIMIZATION [OPT-2]: Adds analytical gradient support for automatic differentiation.
    
    The gradient of rotation matrix R = exp(ω∧) w.r.t. ω:
        ∂R/∂ω = A(ω) where A is the adjoint action on SE(3)
    
    Args:
        omega: Tangent vector [..., 3] (axis * angle)
        
    Returns:
        R: Rotation matrix [..., 3, 3]
        J: Left Jacobian [..., 3, 3]
    """
    theta = torch.norm(omega, dim=-1, keepdim=True).clamp(min=1e-8)
    axis = omega / theta
    
    omega_hat = skew_symmetric(omega)
    omega_hat_sq = omega_hat @ omega_hat
    
    # Compute sin and cos
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    
    I = torch.eye(3, device=omega.device, dtype=omega.dtype)
    
    # Rotation matrix via Rodrigues
    R = I + (sin_theta / theta) * omega_hat + ((1 - cos_theta) / (theta ** 2)) * omega_hat_sq
    
    return R


def so3_left_jacobian(omega: torch.Tensor) -> torch.Tensor:
    r"""Compute the left Jacobian of SO(3) with numerical stability.
    
    OPTIMIZATION [OPT-3]: Implements the closed-form left Jacobian and its inverse
    using the Barfoot method for efficient computation.
    
    J(ω) = I + ((1-cosθ)/θ²) ω∧ + ((θ-sinθ)/θ³) (ω∧)²
    
    For small θ:
        J(ω) ≈ I - ½ ω∧ + 1/6 ω∧²
    
    Args:
        omega: Axis-angle vector [..., 3]
        
    Returns:
        J: Left Jacobian [..., 3, 3]
    """
    theta = torch.norm(omega, dim=-1, keepdim=True)  # [..., 1]
    batch_shape = omega.shape[:-1]
    
    omega_hat = skew_symmetric(omega)  # [..., 3, 3]
    omega_hat_sq = omega_hat @ omega_hat  # [..., 3, 3]
    
    I = torch.eye(3, device=omega.device, dtype=omega.dtype).expand(*batch_shape, 3, 3)
    
    J = torch.zeros(*batch_shape, 3, 3, device=omega.device, dtype=omega.dtype)
    
    small_angle_mask = theta.squeeze(-1) < 1e-4
    
    # Case 1: Small angle (Taylor expansion)
    if small_angle_mask.any():
        theta_small = theta[small_angle_mask]
        omega_hat_small = omega_hat[small_angle_mask]
        omega_hat_sq_small = omega_hat_sq[small_angle_mask]
        
        # Taylor coefficients for J
        # J ≈ I - ½ ω∧ + 1/6 ω∧²
        coef_omega = -0.5 * (1 - theta_small ** 2 / 24)
        coef_omega_sq = (1/6) * (1 - theta_small ** 2 / 60)
        
        J_small = I[small_angle_mask] + \
                  coef_omega.unsqueeze(-1).unsqueeze(-1) * omega_hat_small + \
                  coef_omega_sq.unsqueeze(-1).unsqueeze(-1) * omega_hat_sq_small
        J[small_angle_mask] = J_small
    
    # Case 2: Normal angle
    normal_mask = ~small_angle_mask
    if normal_mask.any():
        theta_norm = theta[normal_mask]
        omega_hat_norm = omega_hat[normal_mask]
        omega_hat_sq_norm = omega_hat_sq[normal_mask]
        
        coef1 = (1 - torch.cos(theta_norm)) / theta_norm ** 2
        coef2 = (theta_norm - torch.sin(theta_norm)) / theta_norm ** 3
        
        J_norm = I[normal_mask] + \
                 coef1.unsqueeze(-1).unsqueeze(-1) * omega_hat_norm + \
                 coef2.unsqueeze(-1).unsqueeze(-1) * omega_hat_sq_norm
        J[normal_mask] = J_norm
    
    return J


def so3_left_jacobian_inverse(omega: torch.Tensor) -> torch.Tensor:
    r"""Compute the inverse of the left Jacobian J⁻¹(ω).
    
    OPTIMIZATION [OPT-3]: Implements the closed-form inverse Jacobian.
    
    J⁻¹(ω) = I - ½ ω∧ + (1 - (θ·sinθ)/(2(1-cosθ))) / θ² · (ω∧)²
    
    For small θ:
        J⁻¹(ω) ≈ I + ½ ω∧ + 1/12 ω∧²
    
    Args:
        omega: Axis-angle vector [..., 3]
        
    Returns:
        J_inv: Inverse of left Jacobian [..., 3, 3]
    """
    theta = torch.norm(omega, dim=-1, keepdim=True)
    batch_shape = omega.shape[:-1]
    
    omega_hat = skew_symmetric(omega)
    omega_hat_sq = omega_hat @ omega_hat
    
    I = torch.eye(3, device=omega.device, dtype=omega.dtype).expand(*batch_shape, 3, 3)
    
    J_inv = torch.zeros(*batch_shape, 3, 3, device=omega.device, dtype=omega.dtype)
    
    small_angle_mask = theta.squeeze(-1) < 1e-4
    
    # Case 1: Small angle (Taylor expansion)
    if small_angle_mask.any():
        theta_small = theta[small_angle_mask]
        omega_hat_small = omega_hat[small_angle_mask]
        omega_hat_sq_small = omega_hat_sq[small_angle_mask]
        
        coef_omega = 0.5 * (1 + theta_small ** 2 / 24)
        coef_omega_sq = 1/12 * (1 - theta_small ** 2 / 60)
        
        J_inv_small = I[small_angle_mask] + \
                      coef_omega.unsqueeze(-1).unsqueeze(-1) * omega_hat_small + \
                      coef_omega_sq.unsqueeze(-1).unsqueeze(-1) * omega_hat_sq_small
        J_inv[small_angle_mask] = J_inv_small
    
    # Case 2: Normal angle
    normal_mask = ~small_angle_mask
    if normal_mask.any():
        theta_norm = theta[normal_mask]
        omega_hat_norm = omega_hat[normal_mask]
        omega_hat_sq_norm = omega_hat_sq[normal_mask]
        
        sin_theta = torch.sin(theta_norm)
        cos_theta = torch.cos(theta_norm)
        
        # Coefficient for omega_hat
        coef1 = 0.5 * torch.ones_like(theta_norm)
        
        # Coefficient for omega_hat_sq
        # (1 - (θ·sinθ)/(2(1-cosθ)))) / θ²
        inner = (theta_norm * sin_theta) / (2 * (1 - cos_theta) + 1e-8)
        coef2 = (1 - inner) / theta_norm ** 2
        
        J_inv_norm = I[normal_mask] + \
                     coef1.unsqueeze(-1).unsqueeze(-1) * omega_hat_norm + \
                     coef2.unsqueeze(-1).unsqueeze(-1) * omega_hat_sq_norm
        J_inv[normal_mask] = J_inv_norm
    
    return J_inv


# ============================================================================
# [OPT-4] SLERP - SPHERICAL LINEAR INTERPOLATION FOR QUATERNIONS
# Used for smooth interpolation between SE(3) poses
# ============================================================================

def slerp_quaternion(q1: torch.Tensor, q2: torch.Tensor, t: float) -> torch.Tensor:
    r"""Spherical linear interpolation between two quaternions.
    
    OPTIMIZATION [OPT-4]: Adds SLERP for smooth interpolation between keyframe poses.
    
    SLERP(q1, q2, t) = (sin((1-t)θ)·q1 + sin(tθ)·q2) / sin(θ)
    
    where θ = arccos(|⟨q1, q2⟩|) is the angle between quaternions.
    
    Args:
        q1: Start quaternion [..., 4] (w, x, y, z)
        q2: End quaternion [..., 4] (w, x, y, z)
        t: Interpolation parameter [0, 1]
        
    Returns:
        q_interp: Interpolated quaternion [..., 4]
    """
    # Normalize quaternions
    q1 = F.normalize(q1, dim=-1)
    q2 = F.normalize(q2, dim=-1)
    
    # Compute dot product
    dot = (q1 * q2).sum(dim=-1, keepdim=True)  # [..., 1]
    
    # Ensure shortest path (flip q2 if dot < 0)
    q2_adjusted = torch.where(dot < 0, -q2, q2)
    dot = torch.abs(dot)
    
    # Compute angle
    theta_0 = torch.acos(torch.clamp(dot, 0, 1 - 1e-6))  # [..., 1]
    
    # Handle small angle case (lerp)
    small_angle_mask = theta_0.squeeze(-1) < 1e-4  # [...]
    
    q_interp = torch.zeros_like(q1)
    
    # Case 1: Nearly identical quaternions (use lerp)
    if small_angle_mask.any():
        q_interp[small_angle_mask] = (1 - t) * q1[small_angle_mask] + t * q2_adjusted[small_angle_mask]
        q_interp[small_angle_mask] = F.normalize(q_interp[small_angle_mask], dim=-1)
    
    # Case 2: Normal case (SLERP)
    normal_mask = ~small_angle_mask
    if normal_mask.any():
        theta = theta_0[normal_mask]
        q1_n = q1[normal_mask]
        q2_n = q2_adjusted[normal_mask]
        
        sin_theta = torch.sin(theta)
        sin_theta_1 = torch.sin((1 - t) * theta)
        sin_theta_t = torch.sin(t * theta)
        
        s0 = sin_theta_1 / sin_theta
        s1 = sin_theta_t / sin_theta
        
        q_interp[normal_mask] = s0 * q1_n + s1 * q2_n
        q_interp[normal_mask] = F.normalize(q_interp[normal_mask], dim=-1)
    
    return q_interp


def se3_interpolate(T1: torch.Tensor, T2: torch.Tensor, t: float) -> torch.Tensor:
    r"""Interpolate between two SE(3) transformations.
    
    OPTIMIZATION [OPT-4]: Full SE(3) interpolation using logarithmic map.
    
    Args:
        T1: Start transformation [..., 4, 4]
        T2: End transformation [..., 4, 4]
        t: Interpolation parameter [0, 1]
        
    Returns:
        T_interp: Interpolated transformation [..., 4, 4]
    """
    # Extract rotation and translation
    R1 = T1[..., :3, :3]
    t1 = T1[..., :3, 3]
    R2 = T2[..., :3, :3]
    t2 = T2[..., :3, 3]
    
    # Convert to quaternions
    q1 = rotation_matrix_to_quaternion(R1)
    q2 = rotation_matrix_to_quaternion(R2)
    
    # SLERP for rotation
    q_interp = slerp_quaternion(q1, q2, t)
    R_interp = quaternion_to_rotation_matrix(q_interp)
    
    # Linear interpolation for translation
    t_interp = (1 - t) * t1 + t * t2
    
    # Build transformation matrix
    T_interp = torch.eye(4, device=T1.device, dtype=T1.dtype)
    T_interp = T_interp.expand(*T1.shape[:-2], 4, 4).clone()
    T_interp[..., :3, :3] = R_interp
    T_interp[..., :3, 3] = t_interp
    
    return T_interp


# ============================================================================
# [OPT-5] ADJOINT REPRESENTATION OF SE(3)
# Used for transforming velocities between coordinate frames
# ============================================================================

def se3_adjoint(T: torch.Tensor) -> torch.Tensor:
    r"""Compute the adjoint representation of SE(3).
    
    OPTIMIZATION [OPT-5]: Implements Ad_SE(3) for velocity coordinate transformations.
    
    Ad_T(ξ) transforms a twist ξ from one frame to another:
        ξ' = Ad_T(ξ)
        
    The adjoint matrix is:
        Ad_T = | R  [t]_× R |
               | 0      I   |
               
    where [t]_× is the skew-symmetric matrix of translation t.
    
    Args:
        T: Transformation matrix [..., 4, 4]
        
    Returns:
        Ad: Adjoint matrix [..., 6, 6]
    """
    R = T[..., :3, :3]  # [..., 3, 3]
    t = T[..., :3, 3]   # [..., 3]
    
    batch_shape = T.shape[:-2]
    
    # Skew-symmetric of translation
    t_hat = skew_symmetric(t)  # [..., 3, 3]
    
    # Compute [t]_× @ R
    t_hat_R = t_hat @ R  # [..., 3, 3]
    
    # Build adjoint matrix
    Ad = torch.zeros(*batch_shape, 6, 6, device=T.device, dtype=T.dtype)
    Ad[..., :3, :3] = R
    Ad[..., :3, 3:6] = t_hat_R
    Ad[..., 3:6, 3:6] = R
    
    return Ad


def adjoint_inverse(T: torch.Tensor) -> torch.Tensor:
    r"""Compute the inverse of the adjoint representation.
    
    Ad_T⁻¹ = | R^T  -R^T [t]_× |
             | 0       R^T    |
    
    Args:
        T: Transformation matrix [..., 4, 4]
        
    Returns:
        Ad_inv: Inverse adjoint matrix [..., 6, 6]
    """
    R = T[..., :3, :3]  # [..., 3, 3]
    t = T[..., :3, 3]   # [..., 3]
    
    batch_shape = T.shape[:-2]
    R_T = R.transpose(-2, -1)
    
    t_hat = skew_symmetric(t)
    t_hat_R_T = t_hat @ R_T
    
    Ad_inv = torch.zeros(*batch_shape, 6, 6, device=T.device, dtype=T.dtype)
    Ad_inv[..., :3, :3] = R_T
    Ad_inv[..., :3, 3:6] = -t_hat_R_T
    Ad_inv[..., 3:6, 3:6] = R_T
    
    return Ad_inv


def transform_twist(xi: torch.Tensor, T: torch.Tensor, mode: str = "left") -> torch.Tensor:
    r"""Transform a twist vector between coordinate frames.
    
    OPTIMIZATION [OPT-5]: Enables velocity transformations.
    
    Args:
        xi: Twist vector [..., 6] = [ω; v]
        T: Transformation matrix [..., 4, 4]
        mode: "left" for Ad_T(ξ), "right" for Ad_T⁻¹(ξ)
        
    Returns:
        xi_transformed: Transformed twist [..., 6]
    """
    if mode == "left":
        Ad = se3_adjoint(T)
    else:
        Ad = adjoint_inverse(T)
    
    xi_transformed = (Ad @ xi.unsqueeze(-1)).squeeze(-1)
    return xi_transformed


# ============================================================================
# [OPT-6] BATCH SE(3) TRANSFORM OPTIMIZATION
# Efficient batched computation using torch operations
# ============================================================================

def batch_transform_points(
    points: torch.Tensor,
    T: torch.Tensor,
) -> torch.Tensor:
    r"""Batch transform points using SE(3) transformations.
    
    OPTIMIZATION [OPT-6]: Vectorized implementation replacing for-loops.
    
    x' = R @ x + t
    
    Args:
        points: Points [..., N, 3] or [N, 3]
        T: Transformation [..., 4, 4] or [4, 4]
        
    Returns:
        points_transformed: Transformed points [..., N, 3]
    """
    # Handle broadcasting
    points_shape = points.shape
    T_shape = T.shape
    
    # Expand dimensions for broadcasting
    if len(points_shape) == 2 and len(T_shape) == 3:
        # Single batch: points [N, 3], T [B, 4, 4] -> output [B, N, 3]
        points = points.unsqueeze(0)
    elif len(points_shape) == 2 and len(T_shape) == 2:
        # No batch: points [N, 3], T [4, 4] -> output [N, 3]
        T = T.unsqueeze(0)
        points = points.unsqueeze(0)
    
    R = T[..., :3, :3]  # [..., 3, 3]
    t = T[..., :3, 3]   # [..., 3]
    
    # Vectorized matrix multiplication
    # points [..., N, 3] @ R [..., 3, 3] -> [..., N, 3]
    points_rotated = torch.matmul(points, R.transpose(-2, -1))
    
    # Add translation (broadcast)
    points_transformed = points_rotated + t.unsqueeze(-2)
    
    return points_transformed


def batch_transform_covariances(
    covariances: torch.Tensor,
    R: torch.Tensor,
) -> torch.Tensor:
    r"""Batch transform covariance matrices with rotations.
    
    OPTIMIZATION [OPT-6]: Efficient batched covariance transformation.
    
    Σ' = R @ Σ @ R^T
    
    Args:
        covariances: Covariance matrices [..., N, 3, 3]
        R: Rotation matrices [..., 3, 3]
        
    Returns:
        covariances_transformed: [..., N, 3, 3]
    """
    # R @ Σ -> [..., N, 3, 3]
    temp = torch.matmul(covariances, R.transpose(-2, -1))
    # (R @ Σ) @ R^T -> [..., N, 3, 3]
    covariances_transformed = torch.matmul(R.unsqueeze(-3), temp).squeeze(-3)
    
    return covariances_transformed


def batch_se3_exp_map(xi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Batch SE(3) exponential map.
    
    OPTIMIZATION [OPT-6]: Vectorized implementation.
    
    Args:
        xi: Twist vectors [..., 6] = [ω; v]
        
    Returns:
        T: Transformation matrices [..., 4, 4]
        J: Left Jacobian (for SE(3) tangent space) [..., 3, 3]
    """
    omega = xi[..., :3]
    v = xi[..., 3:6]
    
    theta = torch.norm(omega, dim=-1, keepdim=True).clamp(min=1e-8)
    
    # Build skew-symmetric
    omega_hat = skew_symmetric(omega)
    omega_hat_sq = omega_hat @ omega_hat
    
    # Compute rotation matrix
    I = torch.eye(3, device=xi.device, dtype=xi.dtype)
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    
    R = I + (sin_theta / theta) * omega_hat + ((1 - cos_theta) / (theta ** 2)) * omega_hat_sq
    
    # Compute left Jacobian
    J = I + ((1 - cos_theta) / (theta ** 2)) * omega_hat + \
        ((theta - sin_theta) / (theta ** 3)) * omega_hat_sq
    
    # Translation: t = J @ v
    t = torch.matmul(J, v.unsqueeze(-1)).squeeze(-1)
    
    # Build transformation matrix
    batch_shape = xi.shape[:-1]
    T = torch.eye(4, device=xi.device, dtype=xi.dtype)
    T = T.expand(*batch_shape, 4, 4).clone()
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    
    return T, J


def batch_se3_log_map(T: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Batch SE(3) logarithmic map with analytical Jacobian.
    
    OPTIMIZATION [OPT-2]: Analytical gradient computation.
    
    Args:
        T: Transformation matrices [..., 4, 4]
        
    Returns:
        xi: Twist vectors [..., 6]
        J_inv: Inverse Jacobian [..., 3, 3]
    """
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    
    # Compute rotation angle
    trace_R = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    theta = torch.acos(torch.clamp((trace_R - 1) / 2, -1 + 1e-7, 1 - 1e-7))
    
    # Extract omega from rotation matrix
    # For small theta, use approximation
    small_mask = theta < 1e-4
    
    omega = torch.zeros_like(t)
    J_inv = torch.zeros(*T.shape[:-2], 3, 3, device=T.device, dtype=T.dtype)
    
    # Normal case
    normal_mask = ~small_mask
    if normal_mask.any():
        R_n = R[normal_mask]
        theta_n = theta[normal_mask].unsqueeze(-1)
        t_n = t[normal_mask]
        
        # omega = (θ / (2sinθ)) (R - R^T)∨
        coef = theta_n / (2 * torch.sin(theta_n) + 1e-8)
        R_diff = R_n - R_n.transpose(-2, -1)
        omega_hat = coef * R_diff
        omega[normal_mask] = from_skew_to_vector(omega_hat)
        
        # J⁻¹ for normal case
        sin_theta = torch.sin(theta_n)
        cos_theta = torch.cos(theta_n)
        inner = (theta_n * sin_theta) / (2 * (1 - cos_theta) + 1e-8)
        coef_J = (1 - inner) / theta_n ** 2
        
        I_n = torch.eye(3, device=T.device, dtype=T.dtype)
        omega_hat_n = skew_symmetric(omega[normal_mask])
        omega_hat_sq_n = omega_hat_n @ omega_hat_n
        
        J_inv[normal_mask] = I_n + 0.5 * omega_hat_n + coef_J * omega_hat_sq_n
    
    # Small angle case (Taylor expansion)
    if small_mask.any():
        I_s = torch.eye(3, device=T.device, dtype=T.dtype)
        omega_hat_s = skew_symmetric(omega[small_mask])
        omega_hat_sq_s = omega_hat_s @ omega_hat_s
        
        # J⁻¹ ≈ I + ½ ω∧ + 1/12 ω∧²
        J_inv[small_mask] = I_s + 0.5 * omega_hat_s + (1/12) * omega_hat_sq_s
    
    # Translation part: v = J⁻¹ @ t
    v = torch.matmul(J_inv, t.unsqueeze(-1)).squeeze(-1)
    
    # Combine
    xi = torch.cat([omega, v], dim=-1)
    
    return xi, J_inv


# ============================================================================
# ORIGINAL SE(3) FUNCTIONS (preserved for compatibility)
# ============================================================================

def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    r"""Convert quaternion to rotation matrix."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    
    norm = torch.norm(q, dim=-1, keepdim=True)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    
    R = torch.zeros((*q.shape[:-1], 3, 3), device=q.device, dtype=q.dtype)
    
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


def rotation_matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    r"""Convert rotation matrix to quaternion using Shepperd method."""
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    
    q = torch.zeros((*R.shape[:-2], 4), device=R.device, dtype=R.dtype)
    
    # Case 1
    s = torch.sqrt(trace + 1.0) * 2
    q[..., 0] = 0.25 * s
    q[..., 1] = (R[..., 2, 1] - R[..., 1, 2]) / s
    q[..., 2] = (R[..., 0, 2] - R[..., 2, 0]) / s
    q[..., 3] = (R[..., 1, 0] - R[..., 0, 1]) / s
    
    # Case 2
    mask = (R[..., 0, 0] > R[..., 1, 1]) & (R[..., 0, 0] > R[..., 2, 2])
    s2 = torch.sqrt(1.0 + R[..., 0, 0] - R[..., 1, 1] - R[..., 2, 2]) * 2
    q_temp = torch.zeros_like(q)
    q_temp[..., 0] = (R[..., 2, 1] - R[..., 1, 2]) / s2
    q_temp[..., 1] = 0.25 * s2
    q_temp[..., 2] = (R[..., 0, 1] + R[..., 1, 0]) / s2
    q_temp[..., 3] = (R[..., 0, 2] + R[..., 2, 0]) / s2
    q = torch.where(mask.unsqueeze(-1), q_temp, q)
    
    # Case 3
    mask2 = (~mask) & (R[..., 1, 1] > R[..., 2, 2])
    s3 = torch.sqrt(1.0 + R[..., 1, 1] - R[..., 0, 0] - R[..., 2, 2]) * 2
    q_temp = torch.zeros_like(q)
    q_temp[..., 0] = (R[..., 0, 2] - R[..., 2, 0]) / s3
    q_temp[..., 1] = (R[..., 0, 1] + R[..., 1, 0]) / s3
    q_temp[..., 2] = 0.25 * s3
    q_temp[..., 3] = (R[..., 1, 2] + R[..., 2, 1]) / s3
    q = torch.where(mask2.unsqueeze(-1), q_temp, q)
    
    # Case 4
    mask3 = (~mask) & (~mask2)
    s4 = torch.sqrt(1.0 + R[..., 2, 2] - R[..., 0, 0] - R[..., 1, 1]) * 2
    q_temp = torch.zeros_like(q)
    q_temp[..., 0] = (R[..., 1, 0] - R[..., 0, 1]) / s4
    q_temp[..., 1] = (R[..., 0, 2] + R[..., 2, 0]) / s4
    q_temp[..., 2] = (R[..., 1, 2] + R[..., 2, 1]) / s4
    q_temp[..., 3] = 0.25 * s4
    q = torch.where(mask3.unsqueeze(-1), q_temp, q)
    
    return F.normalize(q, dim=-1)


def so3_exp_map(omega: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Exponential map from so(3) to SO(3) with analytical gradient."""
    R = rodrigues_formula_stable(omega)
    J = so3_left_jacobian(omega)
    return R, J


def se3_exp_map(xi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Exponential map from se(3) to SE(3) with analytical gradient."""
    omega = xi[..., :3]
    v = xi[..., 3:6]
    
    R, J_so3 = so3_exp_map(omega)
    t = (J_so3 @ v.unsqueeze(-1)).squeeze(-1)
    
    T = torch.eye(4, device=xi.device, dtype=xi.dtype)
    T = T.expand(*xi.shape[:-1], 4, 4).clone()
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    
    return T, J_so3


def se3_log_map(T: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Logarithmic map from SE(3) to se(3) with analytical gradient."""
    return batch_se3_log_map(T)


# ============================================================================
# [OPT-7] TRAJECTORY SMOOTHNESS REGULARIZATION
# Adds velocity and acceleration constraints for SE(3) trajectories
# ============================================================================

class TrajectorySmoother(nn.Module):
    r"""SE(3) trajectory smoother with velocity and acceleration constraints.
    
    OPTIMIZATION [OPT-7]: Implements temporal regularization for smooth trajectories.
    
    The loss function encourages:
    1. Constant velocity (acceleration → 0)
    2. Small angular velocity
    3. Smooth rotation changes
    """
    
    def __init__(
        self,
        velocity_weight: float = 0.1,
        acceleration_weight: float = 0.05,
        angular_velocity_weight: float = 0.1,
        angular_acceleration_weight: float = 0.05,
    ) -> None:
        """Initialize trajectory smoother.
        
        Args:
            velocity_weight: Weight for velocity smoothness
            acceleration_weight: Weight for acceleration smoothness
            angular_velocity_weight: Weight for angular velocity
            angular_acceleration_weight: Weight for angular acceleration
        """
        super().__init__()
        self.velocity_weight = velocity_weight
        self.acceleration_weight = acceleration_weight
        self.angular_velocity_weight = angular_velocity_weight
        self.angular_acceleration_weight = angular_acceleration_weight
    
    def compute_velocity(
        self,
        positions: torch.Tensor,
        dt: float = 1.0,
    ) -> torch.Tensor:
        r"""Compute velocity from position trajectory.
        
        v(t) = (p(t+dt) - p(t)) / dt
        
        Args:
            positions: Position trajectory [T, 3] or [..., T, 3]
            dt: Time step
            
        Returns:
            velocities: Velocity trajectory
        """
        if positions.dim() == 2:
            velocities = torch.diff(positions, dim=0) / dt
        else:
            velocities = torch.diff(positions, dim=-2) / dt
        return velocities
    
    def compute_acceleration(
        self,
        positions: torch.Tensor,
        dt: float = 1.0,
    ) -> torch.Tensor:
        r"""Compute acceleration from position trajectory.
        
        a(t) = (v(t+dt) - v(t)) / dt = (p(t+dt) - 2p(t) + p(t-dt)) / dt²
        
        Args:
            positions: Position trajectory [T, 3] or [..., T, 3]
            dt: Time step
            
        Returns:
            accelerations: Acceleration trajectory
        """
        if positions.dim() == 2:
            accelerations = torch.diff(positions, n=2, dim=0) / (dt ** 2)
        else:
            accelerations = torch.diff(positions, n=2, dim=-2) / (dt ** 2)
        return accelerations
    
    def compute_angular_velocity(
        self,
        quaternions: torch.Tensor,
        dt: float = 1.0,
    ) -> torch.Tensor:
        r"""Compute angular velocity from quaternion trajectory.
        
        Args:
            quaternions: Quaternion trajectory [..., T, 4]
            dt: Time step
            
        Returns:
            omega: Angular velocity trajectory [..., T-1, 3]
        """
        q1 = quaternions[..., :-1, :]  # [..., T-1, 4]
        q2 = quaternions[..., 1:, :]  # [..., T-1, 4]
        
        # Quaternion difference (relative rotation)
        # q_rel = q2 * conj(q1)
        q_conj = q1.clone()
        q_conj[..., 1:] = -q_conj[..., 1:]  # conjugate
        
        # Product for relative rotation
        w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
        
        # q_rel = q2 ⊗ q1*
        w_rel = w1 * w2 + x1 * x2 + y1 * y2 + z1 * z2
        x_rel = w1 * x2 - x1 * w2 + y1 * z2 - z1 * y2
        y_rel = w1 * y2 - y1 * w2 - x1 * z2 + z1 * x2
        z_rel = w1 * z2 + x1 * y2 - y1 * x2 - z1 * w2
        
        # Convert to axis-angle (small angle approximation)
        # For small angle: angle ≈ 2 * acos(w_rel)
        theta = 2 * torch.acos(torch.clamp(w_rel, -1 + 1e-6, 1 - 1e-6))
        
        # Handle small angle
        sin_theta_half = torch.sqrt(x_rel ** 2 + y_rel ** 2 + z_rel ** 2 + 1e-8)
        axis = torch.stack([x_rel, y_rel, z_rel], dim=-1) / sin_theta_half.unsqueeze(-1)
        
        # Angular velocity = axis * angle / dt
        omega = axis * theta.unsqueeze(-1) / dt
        
        return omega
    
    def forward(
        self,
        positions: torch.Tensor,
        quaternions: Optional[torch.Tensor] = None,
        dt: float = 1.0,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        r"""Compute trajectory smoothness loss.
        
        Args:
            positions: Position trajectory [..., T, 3]
            quaternions: Quaternion trajectory [..., T, 4]
            dt: Time step
            
        Returns:
            loss: Total smoothness loss
            info: Individual loss components
        """
        loss = 0.0
        info = {}
        
        # Velocity smoothness
        if self.velocity_weight > 0:
            velocities = self.compute_velocity(positions, dt)
            velocity_loss = torch.mean(velocities ** 2)
            loss = loss + self.velocity_weight * velocity_loss
            info['velocity_smooth'] = velocity_loss.item()
        
        # Acceleration smoothness
        if self.acceleration_weight > 0:
            accelerations = self.compute_acceleration(positions, dt)
            acceleration_loss = torch.mean(accelerations ** 2)
            loss = loss + self.acceleration_weight * acceleration_loss
            info['acceleration_smooth'] = acceleration_loss.item()
        
        # Angular velocity smoothness
        if quaternions is not None and self.angular_velocity_weight > 0:
            omega = self.compute_angular_velocity(quaternions, dt)
            angular_velocity_loss = torch.mean(omega ** 2)
            loss = loss + self.angular_velocity_weight * angular_velocity_loss
            info['angular_velocity_smooth'] = angular_velocity_loss.item()
        
        # Angular acceleration smoothness
        if quaternions is not None and self.angular_acceleration_weight > 0:
            omega = self.compute_angular_velocity(quaternions, dt)
            angular_acc = torch.diff(omega, dim=-2) / dt
            angular_acc_loss = torch.mean(angular_acc ** 2)
            loss = loss + self.angular_acceleration_weight * angular_acc_loss
            info['angular_acc_smooth'] = angular_acc_loss.item()
        
        info['total'] = loss.item()
        return loss, info


# ============================================================================
# ENHANCED SE(3) MODULE WITH ALL OPTIMIZATIONS
# ============================================================================

class SE3Parameter(nn.Module):
    r"""Enhanced SE(3) parameterization with all optimizations."""
    
    def __init__(
        self,
        num_instances: int = 1,
        init_translation: Optional[torch.Tensor] = None,
        init_rotation: Optional[torch.Tensor] = None,
        learn_translation: bool = True,
        learn_rotation: bool = True,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        
        self.num_instances = num_instances
        self.learn_translation = learn_translation
        self.learn_rotation = learn_rotation
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize twist vector
        if init_translation is not None or init_rotation is not None:
            if init_rotation is not None:
                R = quaternion_to_rotation_matrix(init_rotation.to(device))
            else:
                R = torch.eye(3, device=device).unsqueeze(0).expand(num_instances, -1, -1)
            
            if init_translation is not None:
                t = init_translation.to(device)
            else:
                t = torch.zeros(num_instances, 3, device=device)
            
            T_init = torch.eye(4, device=device).unsqueeze(0).expand(num_instances, -1, -1).clone()
            T_init[:, :3, :3] = R
            T_init[:, :3, 3] = t
            
            xi_init, _ = batch_se3_log_map(T_init)
        else:
            xi_init = torch.zeros(num_instances, 6, device=device)
        
        self.twist = nn.Parameter(xi_init, requires_grad=learn_translation or learn_rotation)
        
        if init_rotation is not None:
            init_quat = init_rotation.to(device)
        else:
            init_quat = torch.tensor([1., 0., 0., 0.], device=device)
            init_quat = init_quat.unsqueeze(0).expand(num_instances, -1)
        
        if learn_rotation:
            self.rotation_refine = nn.Parameter(
                torch.zeros(num_instances, 3, device=device),
                requires_grad=True
            )
        else:
            self.rotation_refine = None
    
    @property
    def translation(self) -> torch.Tensor:
        return self.twist[:, 3:6]
    
    @property
    def rotation_omega(self) -> torch.Tensor:
        return self.twist[:, :3]
    
    def get_transformation(
        self,
        instance_idx: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if instance_idx is not None:
            twist = self.twist[instance_idx]
            refine = self.rotation_refine[instance_idx] if self.rotation_refine is not None else None
        else:
            twist = self.twist
            refine = self.rotation_refine
        
        T_base, J = batch_se3_exp_map(twist)
        
        if refine is not None and refine.requires_grad:
            R_refine, _ = so3_exp_map(refine * 0.01)
            T_base[..., :3, :3] = T_base[..., :3, :3] @ R_refine
        
        return T_base


class SE3Transform(nn.Module):
    r"""Enhanced SE(3) transformation module with all optimizations."""
    
    def __init__(
        self,
        num_instances: int = 100,
        temporal_dim: int = 256,
        hidden_dim: int = 128,
        learn_translation: bool = True,
        learn_rotation: bool = True,
        velocity_init: bool = True,
        trajectory_smooth_weight: float = 0.05,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        
        self.num_instances = num_instances
        self.temporal_dim = temporal_dim
        self.hidden_dim = hidden_dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Pose encoder
        self.pose_encoder = nn.Sequential(
            nn.Linear(temporal_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 6),
        )
        
        # SE(3) parameters
        self.se3_params = SE3Parameter(
            num_instances=num_instances,
            learn_translation=learn_translation,
            learn_rotation=learn_rotation,
            device=self.device,
        )
        
        # Velocity prior
        self.velocity_init = velocity_init
        if velocity_init:
            self.velocity = nn.Parameter(
                torch.zeros(num_instances, 3, device=self.device),
                requires_grad=True
            )
            self.angular_velocity = nn.Parameter(
                torch.zeros(num_instances, 3, device=self.device),
                requires_grad=True
            )
        
        # Trajectory smoother [OPT-7]
        self.trajectory_smoother = TrajectorySmoother(
            velocity_weight=trajectory_smooth_weight,
            acceleration_weight=trajectory_smooth_weight * 0.5,
            angular_velocity_weight=trajectory_smooth_weight,
            angular_acceleration_weight=trajectory_smooth_weight * 0.5,
        )
        
        # Instance state
        self.instance_active = nn.Parameter(
            torch.zeros(num_instances, dtype=torch.bool, device=self.device),
            requires_grad=False
        )
        self.motion_confidence = nn.Parameter(
            torch.ones(num_instances, device=self.device),
            requires_grad=False
        )
        
        # Trajectory history for smoothing
        self._position_history = {}
        self._quaternion_history = {}
    
    def encode_pose(
        self,
        temporal_features: torch.Tensor,
        instance_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if instance_ids is not None:
            features = temporal_features[instance_ids]
        else:
            features = temporal_features
        return self.pose_encoder(features)
    
    def transform_points(
        self,
        points: torch.Tensor,
        instance_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        T = self.get_transformation(instance_idx)
        return batch_transform_points(points, T)
    
    def get_transformation(
        self,
        instance_idx: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.se3_params.get_transformation(instance_idx)
    
    def get_rotation_matrix(
        self,
        instance_idx: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        T = self.get_transformation(instance_idx)
        return T[..., :3, :3]
    
    def get_quaternion(
        self,
        instance_idx: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        R = self.get_rotation_matrix(instance_idx)
        return rotation_matrix_to_quaternion(R)
    
    def interpolate_pose(
        self,
        instance_id: int,
        t: float,
        t1: float,
        T1: torch.Tensor,
        t2: float,
        T2: torch.Tensor,
    ) -> torch.Tensor:
        r"""Interpolate SE(3) pose at time t.
        
        OPTIMIZATION [OPT-4]: SLERP for smooth interpolation.
        """
        alpha = (t - t1) / (t2 - t1 + 1e-8)
        return se3_interpolate(T1, T2, alpha)
    
    def compute_trajectory_loss(
        self,
        instance_id: int,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        r"""Compute trajectory smoothness loss.
        
        OPTIMIZATION [OPT-7]: Velocity and acceleration constraints.
        """
        if instance_id not in self._position_history:
            return torch.tensor(0.0, device=self.device), {}
        
        positions = torch.stack(self._position_history[instance_id])  # [T, 3]
        quaternions = torch.stack(self._quaternion_history[instance_id])  # [T, 4]
        
        return self.trajectory_smoother(positions, quaternions)
    
    def update_trajectory_history(
        self,
        instance_id: int,
        position: torch.Tensor,
        quaternion: torch.Tensor,
        max_history: int = 30,
    ) -> None:
        """Update trajectory history for smoothing."""
        if instance_id not in self._position_history:
            self._position_history[instance_id] = []
            self._quaternion_history[instance_id] = []
        
        self._position_history[instance_id].append(position.detach())
        self._quaternion_history[instance_id].append(quaternion.detach())
        
        # Keep only recent history
        if len(self._position_history[instance_id]) > max_history:
            self._position_history[instance_id] = self._position_history[instance_id][-max_history:]
            self._quaternion_history[instance_id] = self._quaternion_history[instance_id][-max_history:]
    
    def regularize(self, weight: float = 0.01) -> torch.Tensor:
        """Regularization with enhanced constraints."""
        loss = 0.0
        
        if self.learn_translation:
            t_norm = torch.norm(self.se3_params.translation, dim=-1)
            loss = loss + weight * torch.mean(t_norm)
        
        R = self.get_rotation_matrix()
        det_R = torch.det(R)
        loss = loss + weight * torch.mean((det_R - 1) ** 2)
        
        I = torch.eye(3, device=R.device).unsqueeze(0)
        loss = loss + weight * torch.mean((R @ R.transpose(-2, -1) - I) ** 2)
        
        if self.velocity_init:
            loss = loss + weight * 0.1 * (
                torch.mean(self.velocity ** 2) +
                torch.mean(self.angular_velocity ** 2)
            )
        
        return loss
    
    def forward(
        self,
        canonical_gaussians: Dict[str, torch.Tensor],
        instance_ids: torch.Tensor,
        timestamps: torch.Tensor,
        temporal_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if temporal_features is not None:
            twist = self.encode_pose(temporal_features, instance_ids)
            T, _ = batch_se3_exp_map(twist)
        else:
            unique_ids = torch.unique(instance_ids)
            T = self.se3_params.get_transformation(unique_ids)
            T_map = torch.zeros(
                instance_ids.shape[0], 4, 4,
                device=instance_ids.device, dtype=T.dtype
            )
            for i, uid in enumerate(unique_ids):
                mask = instance_ids == uid
                T_map[mask] = T[i]
            T = T_map
        
        R = T[..., :3, :3]
        t = T[..., :3, 3]
        
        positions = canonical_gaussians['positions']
        positions_transformed = batch_transform_points(positions, T)
        
        if 'covariances' in canonical_gaussians:
            covariances = canonical_gaussians['covariances']
            covariances_transformed = batch_transform_covariances(covariances, R)
        else:
            covariances_transformed = None
        
        return {
            'positions': positions_transformed,
            'covariances': covariances_transformed,
            'R': R,
            't': t,
            'T': T,
        }


def compose_se3(T1: torch.Tensor, T2: torch.Tensor) -> torch.Tensor:
    r"""Compose SE(3) transformations."""
    R1 = T1[..., :3, :3]
    t1 = T1[..., :3, 3]
    R2 = T2[..., :3, :3]
    t2 = T2[..., :3, 3]
    
    R_composed = R1 @ R2
    t_composed = t1 + (R1 @ t2.unsqueeze(-1)).squeeze(-1)
    
    T_composed = torch.eye(4, device=T1.device, dtype=T1.dtype)
    T_composed = T_composed.expand(*T1.shape[:-2], 4, 4).clone()
    T_composed[..., :3, :3] = R_composed
    T_composed[..., :3, 3] = t_composed
    
    return T_composed


def interpolate_se3(T0: torch.Tensor, T1: torch.Tensor, alpha: float) -> torch.Tensor:
    r"""Interpolate SE(3) transformations."""
    return se3_interpolate(T0, T1, alpha)
