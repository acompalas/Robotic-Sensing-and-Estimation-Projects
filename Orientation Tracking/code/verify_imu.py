"""
Verify IMU calibration by comparing gyro-only integration with VICON.
This is the sanity check mentioned in the assignment.
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from transforms3d.euler import mat2euler, quat2euler

from calibrate_imu import (
    load_imu_raw, load_vicon, calibrate_imu, integrate_gyro
)

# --- paths ---
HERE = Path(__file__).resolve().parent
DATA = (HERE / ".." / "data").resolve()
PLOTS_DIR = (DATA / "plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# --- constants ---
STATIC_SECONDS = 3.0


def main():
    parser = argparse.ArgumentParser(
        description='Verify IMU calibration using gyro-only integration'
    )
    parser.add_argument("--dataset", type=int, required=True, 
                       choices=range(1, 10),
                       help="Dataset number (1-9)")
    args = parser.parse_args()

    ds = args.dataset

    print("=" * 80)
    print(f"IMU CALIBRATION VERIFICATION - Dataset {ds}")
    print("=" * 80)

    # Load and calibrate IMU
    print("\n[1/3] Loading and calibrating IMU...")
    t, acc_c, gyro_c = load_imu_raw(ds)
    acc, gyro = calibrate_imu(t, acc_c, gyro_c, static_seconds=STATIC_SECONDS)
    print(f"  Total samples: {len(t)}")
    print(f"  Duration: {t[-1] - t[0]:.2f} seconds")

    # Gyro-only integration
    print("\n[2/3] Performing gyro-only integration...")
    q_gyro = integrate_gyro(t, gyro)
    print(f"  Quaternion trajectory shape: {q_gyro.shape}")

    # Convert to RPY
    rpy_imu = np.array([quat2euler(q_gyro[k], axes="sxyz") 
                       for k in range(q_gyro.shape[0])])
    roll_i, pitch_i, yaw_i = rpy_imu.T
    roll_i, pitch_i, yaw_i = np.unwrap(roll_i), np.unwrap(pitch_i), np.unwrap(yaw_i)

    # Load VICON ground truth
    print("\n[3/3] Loading VICON ground truth...")
    tv, Rv = load_vicon(ds)
    rpy_v = np.array([mat2euler(Rv[:, :, k], axes="sxyz") 
                     for k in range(Rv.shape[2])])
    roll_v, pitch_v, yaw_v = rpy_v.T
    roll_v, pitch_v, yaw_v = np.unwrap(roll_v), np.unwrap(pitch_v), np.unwrap(yaw_v)
    print(f"  VICON samples: {len(tv)}")

    # Normalize timestamps to start at 0
    t_normalized = t - t[0]
    tv_normalized = tv - t[0]

    # Plot comparison with improved style
    print("\n[4/4] Plotting results...")
    fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    axs[0].plot(t_normalized, roll_i, 'r-', label="IMU gyro-integrated", linewidth=1.5)
    axs[0].plot(tv_normalized, roll_v, 'b-', label="VICON", alpha=0.7, linewidth=1.5)
    axs[0].set_ylabel("Roll (rad)", fontsize=12)
    axs[0].grid(True, alpha=0.3)
    axs[0].legend(fontsize=10)

    axs[1].plot(t_normalized, pitch_i, 'r-', label="IMU gyro-integrated", linewidth=1.5)
    axs[1].plot(tv_normalized, pitch_v, 'b-', label="VICON", alpha=0.7, linewidth=1.5)
    axs[1].set_ylabel("Pitch (rad)", fontsize=12)
    axs[1].grid(True, alpha=0.3)
    axs[1].legend(fontsize=10)

    axs[2].plot(t_normalized, yaw_i, 'r-', label="IMU gyro-integrated", linewidth=1.5)
    axs[2].plot(tv_normalized, yaw_v, 'b-', label="VICON", alpha=0.7, linewidth=1.5)
    axs[2].set_ylabel("Yaw (rad)", fontsize=12)
    axs[2].set_xlabel("Time (s)", fontsize=12)
    axs[2].grid(True, alpha=0.3)
    axs[2].legend(fontsize=10)

    fig.suptitle(f"Dataset {ds}: Gyro-only integration vs VICON", 
                fontsize=14, fontweight='bold')
    fig.tight_layout()

    out = PLOTS_DIR / f"imu_vs_vicon_{ds}.png"
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print(f"  Saved: {out}")

    print("\n" + "=" * 80)
    print("VERIFICATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()