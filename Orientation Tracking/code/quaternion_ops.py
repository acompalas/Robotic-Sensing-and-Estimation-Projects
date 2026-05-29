"""
Quaternion operations implemented in PyTorch for automatic differentiation.
All operations follow the convention q = [w, x, y, z] where w is the scalar part.
"""

import torch

# Small epsilon for numerical stability
EPS = 1e-6


def quat_mult(q, p):
    """
    Quaternion multiplication: q * p
    
    Args:
        q: quaternion tensor (..., 4) in [w, x, y, z] format
        p: quaternion tensor (..., 4) in [w, x, y, z] format
    Returns:
        quaternion product (..., 4)
    """
    qw, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    pw, px, py, pz = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
    
    w = qw * pw - qx * px - qy * py - qz * pz
    x = qw * px + qx * pw + qy * pz - qz * py
    y = qw * py - qx * pz + qy * pw + qz * px
    z = qw * pz + qx * py - qy * px + qz * pw
    
    return torch.stack([w, x, y, z], dim=-1)


def quat_conjugate(q):
    """
    Quaternion conjugate: q_bar = [w, -x, -y, -z]
    
    Args:
        q: quaternion tensor (..., 4)
    Returns:
        conjugate quaternion (..., 4)
    """
    conj = q.clone()
    conj[..., 1:] = -conj[..., 1:]
    return conj


def quat_inverse(q):
    """
    Quaternion inverse: q^{-1} = q_bar / ||q||^2
    
    Args:
        q: quaternion tensor (..., 4)
    Returns:
        inverse quaternion (..., 4)
    """
    q_conj = quat_conjugate(q)
    norm_sq = torch.sum(q * q, dim=-1, keepdim=True)
    return q_conj / (norm_sq + EPS)  # Prevent division by zero


def quat_norm(q):
    """
    Quaternion norm: ||q||
    
    Args:
        q: quaternion tensor (..., 4)
    Returns:
        norm tensor (...)
    """
    return torch.linalg.norm(q, dim=-1)


def quat_normalize(q):
    """
    Normalize quaternion to unit length
    
    Args:
        q: quaternion tensor (..., 4)
    Returns:
        normalized quaternion (..., 4)
    """
    return q / (torch.linalg.norm(q, dim=-1, keepdim=True) + EPS)  # Prevent division by zero


def quat_exp(v):
    """
    Quaternion exponential for pure quaternion [0, v]
    """
    theta = torch.linalg.norm(v, dim=-1, keepdim=True)
    
    # Handle small angles
    small_angle = theta < 1e-6  # Slightly larger epsilon for float32
    
    w_small = torch.ones_like(theta)
    xyz_small = v / 2.0
    
    w_normal = torch.cos(theta)
    xyz_normal = (v / (theta + 1e-8)) * torch.sin(theta)
    
    # Fix broadcasting for batch operations
    if v.dim() > 1:
        small_angle = small_angle.expand_as(v)
    
    w = torch.where(small_angle[..., :1] if v.dim() > 1 else small_angle, w_small, w_normal)
    xyz = torch.where(small_angle, xyz_small, xyz_normal)
    
    return torch.cat([w, xyz], dim=-1)


def quat_log(q):
    """
    Quaternion logarithm for unit quaternions
    log(q) = [0, (q_v/||q_v||) * arccos(q_s)]
    """
    # Ensure unit quaternion
    q = quat_normalize(q)
    
    w = q[..., 0:1]
    v = q[..., 1:]
    
    v_norm = torch.linalg.norm(v, dim=-1, keepdim=True)
    
    # Handle small v
    small_v = (v_norm < EPS).expand_as(v) if v.dim() > 1 else (v_norm < EPS)
    
    log_v_small = torch.zeros_like(v)
    
    # FIX 3: Clamp arccos input to valid range
    w_clamped = torch.clamp(w, -1.0 + EPS, 1.0 - EPS)  # ✓ Already have this!
    angle = torch.arccos(w_clamped)
    log_v_normal = (v / (v_norm + EPS)) * angle  # ✓ Already have epsilon!
    
    log_v = torch.where(small_v, log_v_small, log_v_normal)
    
    zeros = torch.zeros_like(w)
    return torch.cat([zeros, log_v], dim=-1)


def axis_angle_to_quat(axis, angle):
    """
    Convert axis-angle to quaternion
    q = [cos(theta/2), axis * sin(theta/2)]
    
    Args:
        axis: unit axis tensor (..., 3)
        angle: rotation angle tensor (...)
    Returns:
        quaternion (..., 4)
    """
    half_angle = angle / 2.0
    w = torch.cos(half_angle)
    xyz = axis * torch.sin(half_angle).unsqueeze(-1)
    return torch.cat([w.unsqueeze(-1), xyz], dim=-1)


def quat_rotate_vector(q, v):
    """
    Rotate vector v by quaternion q
    v' = q * [0, v] * q^{-1}
    
    Args:
        q: quaternion tensor (..., 4)
        v: 3D vector tensor (..., 3)
    Returns:
        rotated vector (..., 3)
    """
    # Convert v to pure quaternion [0, v]
    zeros = torch.zeros_like(v[..., 0:1])
    v_quat = torch.cat([zeros, v], dim=-1)
    
    # Compute q * [0, v] * q^{-1}
    q_inv = quat_inverse(q)
    temp = quat_mult(q, v_quat)
    result = quat_mult(temp, q_inv)
    
    # Extract vector part
    return result[..., 1:]