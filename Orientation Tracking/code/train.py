"""
Main script for ECE276A Project 1: Orientation Tracking
Implements projected gradient descent for IMU-based orientation estimation.
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from transforms3d.euler import quat2euler, mat2euler

from calibrate_imu import (
    load_imu_raw, load_vicon, calibrate_imu, integrate_gyro
)
from orientation_tracking import optimize_orientation


# --- Paths ---
HERE = Path(__file__).resolve().parent
DATA = (HERE / ".." / "data").resolve()  
PLOTS_DIR = (DATA / "plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
TRAJECTORIES_DIR = (DATA / "trajectories")
TRAJECTORIES_DIR.mkdir(parents=True, exist_ok=True)

# --- Constants ---
STATIC_SECONDS = 3.0


def save_trajectory(q_traj, t, dataset_id, method="optimized"):
    """
    Save quaternion trajectory to file.
    
    Args:
        q_traj: quaternion trajectory (T, 4)
        t: timestamps (T,)
        dataset_id: dataset identifier (1-9 for train, 10-11 for test)
        method: trajectory type (e.g., 'optimized')
    """
    save_path = TRAJECTORIES_DIR / f"trajectory_{dataset_id}_{method}.npz"
    np.savez(save_path, quaternions=q_traj, timestamps=t)
    print(f"  Saved trajectory: {save_path}")


def plot_orientation_comparison(t_imu, rpy_estimated, tv, rpy_vicon, 
                                dataset, method_name, save_path):
    """
    Plot estimated vs VICON roll, pitch, yaw
    """
    roll_e, pitch_e, yaw_e = rpy_estimated.T
    
    roll_e = np.unwrap(roll_e)
    pitch_e = np.unwrap(pitch_e)
    yaw_e = np.unwrap(yaw_e)
    
    # Normalize timestamps to start at 0
    t_imu_normalized = t_imu - t_imu[0]
    
    fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    
    axs[0].plot(t_imu_normalized, roll_e, 'r-', label=f'{method_name}', linewidth=1.5)
    axs[0].set_ylabel('Roll (rad)', fontsize=12)
    axs[0].grid(True, alpha=0.3)
    axs[0].legend(fontsize=10)
    
    axs[1].plot(t_imu_normalized, pitch_e, 'r-', label=f'{method_name}', linewidth=1.5)
    axs[1].set_ylabel('Pitch (rad)', fontsize=12)
    axs[1].grid(True, alpha=0.3)
    axs[1].legend(fontsize=10)
    
    axs[2].plot(t_imu_normalized, yaw_e, 'r-', label=f'{method_name}', linewidth=1.5)
    axs[2].set_ylabel('Yaw (rad)', fontsize=12)
    axs[2].set_xlabel('Time (s)', fontsize=12)
    axs[2].grid(True, alpha=0.3)
    axs[2].legend(fontsize=10)
    
    # Plot VICON if available
    if tv is not None and rpy_vicon is not None:
        roll_v, pitch_v, yaw_v = rpy_vicon.T
        roll_v = np.unwrap(roll_v)
        pitch_v = np.unwrap(pitch_v)
        yaw_v = np.unwrap(yaw_v)
        
        # Normalize VICON timestamps to match IMU start time
        tv_normalized = tv - t_imu[0]
        
        axs[0].plot(tv_normalized, roll_v, 'b-', label='VICON', alpha=0.7, linewidth=1.5)
        axs[0].legend(fontsize=10)
        
        axs[1].plot(tv_normalized, pitch_v, 'b-', label='VICON', alpha=0.7, linewidth=1.5)
        axs[1].legend(fontsize=10)
        
        axs[2].plot(tv_normalized, yaw_v, 'b-', label='VICON', alpha=0.7, linewidth=1.5)
        axs[2].legend(fontsize=10)
    
    fig.suptitle(f'Dataset {dataset}: {method_name}', 
                fontsize=14, fontweight='bold')
    fig.tight_layout()
    
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Orientation tracking with projected gradient descent'
    )
    parser.add_argument('--dataset', type=int, required=True,
                       help='Dataset number (1-9 for train, 10-11 for test)')
    parser.add_argument('--split', type=str, default='train',
                       choices=['train', 'test'],
                       help='Dataset split: train or test (default: train)')
    parser.add_argument('--max_iter', type=int, default=5000,
                       help='Maximum optimization iterations (default: 5000)')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate (default: 0.01)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use: auto (default), cpu, or cuda')
    parser.add_argument('--save_trajectory', action='store_true',
                       help='Save quaternion trajectory to file')
    
    args = parser.parse_args()
    ds = args.dataset
    
    # Validate dataset number based on split
    if args.split == 'train' and not (1 <= ds <= 9):
        raise ValueError("Train dataset must be 1-9")
    elif args.split == 'test' and not (10 <= ds <= 11):
        raise ValueError("Test dataset must be 10-11")
    
    print("=" * 80)
    print(f"ORIENTATION TRACKING - Dataset {ds} ({args.split.upper()})")
    print("=" * 80)
    
    # -------------------------------------------------------------------------
    # 1. Load and calibrate IMU
    # -------------------------------------------------------------------------
    print("\n[1/3] Loading and calibrating IMU...")
    t, acc_c, gyro_c = load_imu_raw(ds, split=args.split)
    acc, gyro = calibrate_imu(t, acc_c, gyro_c, static_seconds=STATIC_SECONDS)

    # Normalize accelerometer to unit vectors 
    from calibrate_imu import normalize_accelerometer
    acc = normalize_accelerometer(acc)

    print(f"  IMU data: {len(t)} samples, {t[-1] - t[0]:.2f} seconds")
    
    # -------------------------------------------------------------------------
    # 2. Initialize quaternion trajectory
    # -------------------------------------------------------------------------
    print("\n[2/3] Initializing quaternion trajectory...")
    T = len(t)
    
    # Initialize with gyro integration for good starting point
    print("  Using gyro integration for initialization...")
    q_init = integrate_gyro(t, gyro)
    
    # But set first quaternion to identity as per assignment
    print("  Setting q0 = [1, 0, 0, 0] as per assignment...")
    q_init[0] = np.array([1.0, 0.0, 0.0, 0.0])
    
    print(f"  Initial trajectory shape: {q_init.shape}")
    
    # -------------------------------------------------------------------------
    # 3. Load VICON ground truth (only for train set)
    # -------------------------------------------------------------------------
    tv, rpy_vicon = None, None
    if args.split == 'train':
        print("\n  Loading VICON ground truth...")
        tv, Rv = load_vicon(ds)
        rpy_vicon = np.array([mat2euler(Rv[:, :, k], axes='sxyz') 
                             for k in range(Rv.shape[2])])
        print(f"  VICON data: {len(tv)} samples")
    else:
        print("\n  Skipping VICON (test set has no ground truth)")
    
    # -------------------------------------------------------------------------
    # 4. Optimize orientation with accelerometer
    # -------------------------------------------------------------------------
    print(f"\n[3/3] Running projected gradient descent...")
    print(f"  Max iterations: {args.max_iter}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Device: {args.device}")
    print()
    
    q_optimized = optimize_orientation(
        q_init, t, gyro, acc,
        max_iter=args.max_iter,
        lr=args.lr,
        device=args.device
    )
    
    # Save optimized trajectory if requested
    if args.save_trajectory:
        save_trajectory(q_optimized, t, ds, method="optimized")
    
    # Convert to RPY for plotting
    print("\n  Converting to RPY for visualization...")
    rpy_optimized = np.array([quat2euler(q_optimized[k], axes='sxyz') 
                             for k in range(q_optimized.shape[0])])
    
    # Plot optimized result
    print("  Plotting optimized result...")
    plot_orientation_comparison(
        t, rpy_optimized, tv, rpy_vicon, ds,
        "Optimized (Gyro + Accel)",
        PLOTS_DIR / f"optimized_ds{ds}.png"
    )
    
    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)


if __name__ == "__main__":
    main()