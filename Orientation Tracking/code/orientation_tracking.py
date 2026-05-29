"""
Orientation tracking using projected gradient descent.
VECTORIZED for GPU acceleration.
"""

import torch
import numpy as np
from quaternion_ops import (
    quat_mult, quat_inverse, quat_exp, quat_log, 
    quat_normalize
)


def set_seed(seed=42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: random seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def motion_model_vectorized(q_t, dt, omega_t):
    """
    Vectorized motion model: q_{t+1} = q_t * exp([0, tau_t omega_t/2])
    
    Args:
        q_t: quaternion tensor (N, 4)
        dt: time differences (N,) or scalar
        omega_t: angular velocity tensor (N, 3)
    Returns:
        predicted quaternions (N, 4)
    """
    if not isinstance(dt, torch.Tensor):
        dt = torch.tensor(dt, dtype=q_t.dtype, device=q_t.device)
    
    if dt.dim() == 1:
        dt = dt.unsqueeze(1)
    
    dq = quat_exp(dt * omega_t / 2.0)
    return quat_mult(q_t, dq)


def observation_model_vectorized(q_t):
    """
    Vectorized observation model: [0, a_t] = q_t^{-1} * [0,0,0,1] * q_t
    
    Args:
        q_t: quaternion tensor (N, 4)
    Returns:
        predicted acceleration tensor (N, 3)
    """
    N = q_t.shape[0]
    
    gravity_world = torch.zeros(N, 4, dtype=q_t.dtype, device=q_t.device)
    gravity_world[:, 3] = 1.0  # [0, 0, 0, 1] = [w, x, y, z] format
    
    q_inv = quat_inverse(q_t)
    temp = quat_mult(q_inv, gravity_world)
    result = quat_mult(temp, q_t)
    
    return result[:, 1:]  # Return vector part [x, y, z]


def compute_cost_vectorized(q_traj, t, gyro, acc):
    """
    VECTORIZED cost function with NaN debugging.
    """
    T = q_traj.shape[0]
    
    # Check inputs
    if torch.isnan(q_traj).any():
        print("NaN detected in q_traj input!")
        return torch.tensor(float('nan'), device=q_traj.device)
    
    # Motion model term
    dt = t[1:] - t[:-1]
    
    if torch.isnan(dt).any():
        print("NaN in dt!")
    
    f_pred = motion_model_vectorized(q_traj[:-1], dt, gyro[:, :-1].T)
    
    if torch.isnan(f_pred).any():
        print("NaN in f_pred after motion_model!")
        print(f"q_traj min/max: {q_traj.min():.6f} / {q_traj.max():.6f}")
        print(f"gyro min/max: {gyro.min():.6f} / {gyro.max():.6f}")
        return torch.tensor(float('nan'), device=q_traj.device)
    
    q_inv = quat_inverse(q_traj[1:])
    
    if torch.isnan(q_inv).any():
        print("NaN in q_inv!")
        return torch.tensor(float('nan'), device=q_traj.device)
    
    q_rel = quat_mult(q_inv, f_pred)
    
    if torch.isnan(q_rel).any():
        print("NaN in q_rel before normalize!")
        return torch.tensor(float('nan'), device=q_traj.device)
    
    q_rel = quat_normalize(q_rel)
    
    if torch.isnan(q_rel).any():
        print("NaN in q_rel after normalize!")
        print(f"q_rel had zeros: {(torch.linalg.norm(q_rel, dim=-1) < 1e-10).sum()}")
        return torch.tensor(float('nan'), device=q_traj.device)
    
    log_q_rel = quat_log(q_rel)
    
    if torch.isnan(log_q_rel).any():
        print("NaN in log_q_rel!")
        print(f"q_rel w values: min={q_rel[:, 0].min():.6f}, max={q_rel[:, 0].max():.6f}")
        return torch.tensor(float('nan'), device=q_traj.device)
    
    error_vec = 2.0 * log_q_rel[:, 1:]
    motion_cost = 0.5 * torch.sum(error_vec * error_vec)
    
    # Observation model term
    h_pred = observation_model_vectorized(q_traj)
    
    if torch.isnan(h_pred).any():
        print("NaN in h_pred!")
        return torch.tensor(float('nan'), device=q_traj.device)
    
    error = acc.T - h_pred
    obs_cost = 0.5 * torch.sum(error * error)
    
    total_cost = motion_cost + obs_cost
    
    if torch.isnan(total_cost):
        print(f"NaN in total cost! motion={motion_cost:.6f}, obs={obs_cost:.6f}")
    
    return total_cost


def projected_gradient_descent(q_init, t, gyro, acc, 
                               max_iter=100, lr=0.01, 
                               print_every=100, convergence_tol=1e-7,
                               device='auto', seed=42):
    """
    Projected gradient descent - VECTORIZED
    
    Args:
        q_init: numpy array (T, 4) - initial quaternion trajectory
        t: numpy array (T,) - timestamps
        gyro: numpy array (3, T) - gyro measurements [rad/s]
        acc: numpy array (3, T) - accelerometer measurements [g]
        max_iter: maximum number of iterations
        lr: learning rate
        print_every: print cost every N iterations
        convergence_tol: stop if cost change < tol
        device: 'auto', 'cpu', or 'cuda'
        seed: random seed for reproducibility (default: 42)
    
    Returns:
        optimized quaternion trajectory as numpy array (T, 4)
    """
    # Set seed for reproducibility
    set_seed(seed)
    
    # Determine device
    if device == 'auto':
        device_obj = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif device == 'cpu':
        device_obj = torch.device('cpu')
    elif device == 'cuda':
        if not torch.cuda.is_available():
            print("Warning: CUDA requested but not available, falling back to CPU")
            device_obj = torch.device('cpu')
        else:
            device_obj = torch.device('cuda')
    else:
        raise ValueError(f"Invalid device: {device}")
    
    # FIX 1: float32 for GPU, float64 for CPU
    if device_obj.type == 'cuda':
        dtype = torch.float32
        print(f"Using device: {device_obj} with float32 (VECTORIZED, seed={seed})")
    else:
        dtype = torch.float64
        print(f"Using device: {device_obj} with float64 (VECTORIZED, seed={seed})")
    
    q_traj = torch.tensor(q_init, dtype=dtype, device=device_obj, requires_grad=True)
    t_torch = torch.tensor(t, dtype=dtype, device=device_obj)
    gyro_torch = torch.tensor(gyro, dtype=dtype, device=device_obj)
    acc_torch = torch.tensor(acc, dtype=dtype, device=device_obj)
    
    prev_cost = float('inf')
    
    print(f"Starting projected gradient descent (max {max_iter} iterations)...")
    print(f"Convergence tolerance: {convergence_tol}")
    
    with torch.no_grad():
        initial_cost = compute_cost_vectorized(q_traj, t_torch, gyro_torch, acc_torch)
    print(f"Initial cost: {initial_cost.item():.6f}\n")
    
    for iteration in range(max_iter):
        if q_traj.grad is not None:
            q_traj.grad.zero_()
        
        cost = compute_cost_vectorized(q_traj, t_torch, gyro_torch, acc_torch)
        cost.backward()
        
        with torch.no_grad():
            q_traj.data = q_traj.data - lr * q_traj.grad
            q_traj.data = quat_normalize(q_traj.data)
        
        cost_val = cost.item()
        cost_change = abs(prev_cost - cost_val)
        
        if iteration % print_every == 0:
            print(f"Iter {iteration:4d}, Cost: {cost_val:.6f}, Change: {cost_change:.2e}")
        
        if iteration > 5 and cost_change < convergence_tol:
            print(f"\nConverged at iteration {iteration}!")
            print(f"  Cost change ({cost_change:.2e}) < tolerance ({convergence_tol:.2e})")
            break
        
        prev_cost = cost_val
    else:
        print(f"\nReached maximum iterations ({max_iter})")
        print(f"  Final cost change: {cost_change:.2e}")
    
    print(f"Final cost: {cost_val:.6f}")
    
    return q_traj.detach().cpu().numpy()


def optimize_orientation(q_init, t, gyro, acc, max_iter=100, lr=0.001, device='auto', seed=42):
    """
    Wrapper function for orientation optimization.
    
    Args:
        q_init: numpy array (T, 4) - initial quaternion trajectory
        t: numpy array (T,) - timestamps
        gyro: numpy array (3, T) - gyro measurements
        acc: numpy array (3, T) - accelerometer measurements
        max_iter: maximum iterations
        lr: learning rate
        device: 'auto', 'cpu', or 'cuda'
        seed: random seed for reproducibility
    
    Returns:
        optimized quaternion trajectory (T, 4)
    """
    return projected_gradient_descent(q_init, t, gyro, acc, 
                                     max_iter=max_iter, lr=lr, device=device, seed=seed)