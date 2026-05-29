"""
IMU calibration utilities.
Provides functions to load, calibrate, and integrate IMU data.
"""

from pathlib import Path
import numpy as np
from transforms3d.quaternions import qmult

from load_data import read_data

# --- paths ---
HERE = Path(__file__).resolve().parent
DATA = (HERE / ".." / "data").resolve()

# --- calibration constants ---
VREF_MV = 3300.0
ADC_MAX = 1023.0
ACC_SENS_MV_PER_G = 330.0
GYRO_SENS_4X_MV_PER_DEG_S = 3.33


def scale_factor(mv_per_unit):
    """Convert ADC sensitivity to units per count."""
    return VREF_MV / (ADC_MAX * mv_per_unit)


def gyro_mv_per_rad_s(mv_per_deg_s):
    """Convert gyro sensitivity from mV/(deg/s) to mV/(rad/s)."""
    return mv_per_deg_s * (180.0 / np.pi)


def load_imu_raw(dataset: int, split: str = 'train'):
    """
    Load raw IMU data for a given dataset.
    
    Args:
        dataset: Dataset number (1-9 for train, 10-11 for test)
        split: 'train' or 'test'
    
    Returns:
        t: timestamps (N,)
        acc_c: raw accelerometer counts (3, N)
        gyro_c: raw gyroscope counts (3, N)
    """
    ifile = DATA / f"{split}set" / "imu" / f"imuRaw{dataset}.p"
    
    if not ifile.exists():
        raise FileNotFoundError(f"IMU file not found: {ifile}")
    
    imud = read_data(str(ifile))

    vals = np.array(imud["vals"] if isinstance(imud, dict) and "vals" in imud else imud)

    # Fix orientation: want 7xN
    if vals.shape[0] != 7 and vals.shape[1] == 7:
        vals = vals.T
    if vals.shape[0] != 7:
        raise ValueError(f"Unexpected IMU shape {vals.shape}; expected 7xN or Nx7.")

    t = np.squeeze(vals[0, :]).astype(np.float64)
    acc_c = vals[1:4, :].astype(np.float64)
    gyro_c = vals[4:7, :].astype(np.float64)

    return t, acc_c, gyro_c

def load_vicon(dataset: int):
    """
    Load VICON ground truth data for a given dataset.
    
    Args:
        dataset: Dataset number (1-9)
    
    Returns:
        tv: VICON timestamps (N,)
        R: Rotation matrices (3, 3, N)
    """
    vfile = DATA / "trainset" / "vicon" / f"viconRot{dataset}.p"
    vd = read_data(str(vfile))

    if not isinstance(vd, dict):
        raise ValueError("Unexpected VICON format; expected dict with 'rots' and 'ts'.")

    R = np.array(vd["rots"]).astype(np.float64)
    tv = np.squeeze(np.array(vd["ts"])).astype(np.float64)

    if R.ndim != 3 or R.shape[0] != 3 or R.shape[1] != 3:
        raise ValueError(f"Unexpected VICON rot shape {R.shape}; expected 3x3xN.")
    if tv.ndim != 1 or tv.shape[0] != R.shape[2]:
        raise ValueError(f"VICON time shape {tv.shape} does not match rots {R.shape}.")

    return tv, R


def static_mask(t, seconds=2.0):
    """
    Create a boolean mask for the static period.
    
    Args:
        t: timestamps
        seconds: duration of static period
    
    Returns:
        Boolean array indicating static samples
    """
    return t <= (t[0] + seconds)


def calibrate_imu(t, acc_c, gyro_c, static_seconds=3.0, verbose=True):
    """
    Calibrate IMU by removing biases estimated from static period.
    
    During the static period:
    - Gyroscope should read [0, 0, 0] rad/s
    - Accelerometer should read [0, 0, 1] g (gravity)
    
    Args:
        t: timestamps (N,)
        acc_c: raw accelerometer counts (3, N)
        gyro_c: raw gyroscope counts (3, N)
        static_seconds: duration of static period for calibration
        verbose: print calibration info
    
    Returns:
        acc: calibrated accelerometer [g] (3, N)
        gyro: calibrated gyroscope [rad/s] (3, N)
    """
    # Scale factors
    acc_scale = scale_factor(ACC_SENS_MV_PER_G)  # g/count
    gyro_scale = scale_factor(gyro_mv_per_rad_s(GYRO_SENS_4X_MV_PER_DEG_S))  # rad/s/count

    # Static mask
    m = static_mask(t, static_seconds)

    # Estimate biases from static period
    gyro_bias = gyro_c[:, m].mean(axis=1, keepdims=True)
    acc_bias = acc_c[:, m].mean(axis=1, keepdims=True)

    # Apply calibration
    gyro = gyro_scale * (gyro_c - gyro_bias)
    acc = acc_scale * (acc_c - acc_bias) + np.array([[0.0], [0.0], [1.0]])

    if verbose:
        print(f"  Static window: {m.sum()} samples, duration: {t[m][-1] - t[m][0]:.3f} s")
        print(f"  Gyro bias (counts): [{gyro_bias[0,0]:.1f}, {gyro_bias[1,0]:.1f}, {gyro_bias[2,0]:.1f}]")
        print(f"  Acc bias (counts):  [{acc_bias[0,0]:.1f}, {acc_bias[1,0]:.1f}, {acc_bias[2,0]:.1f}]")

    return acc, gyro


def quat_exp(v):
    """
    Quaternion exponential for a pure quaternion [0, v].
    exp([0, v]) = [cos(theta), (v/theta)sin(theta)] where theta = ||v||
    
    Args:
        v: 3D vector
    
    Returns:
        quaternion [w, x, y, z]
    """
    theta = np.linalg.norm(v)
    if theta < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = v / theta
    return np.hstack([np.cos(theta), axis * np.sin(theta)])


def integrate_gyro(t, gyro):
    """
    Integrate gyroscope measurements to obtain orientation trajectory.
    Uses motion model: q_{t+1} = q_t ◦ exp([0, tau_t omega_t/2])
    
    Args:
        t: timestamps (N,)
        gyro: gyroscope measurements [rad/s] (3, N)
    
    Returns:
        q: quaternion trajectory (N, 4) in [w, x, y, z] format
    """
    N = gyro.shape[1]
    q = np.zeros((N, 4), dtype=np.float64)
    q[0] = np.array([1.0, 0.0, 0.0, 0.0])

    for k in range(N - 1):
        dt = t[k + 1] - t[k]
        dq = quat_exp(dt * gyro[:, k] / 2.0)
        q[k + 1] = qmult(q[k], dq)
        q[k + 1] /= np.linalg.norm(q[k + 1])

    # Enforce quaternion sign continuity (q and -q represent same rotation)
    for k in range(1, N):
        if np.dot(q[k], q[k - 1]) < 0:
            q[k] *= -1

    return q

def normalize_accelerometer(acc):
    """
    Normalize accelerometer to unit vectors (direction only).
    This removes the effect of linear acceleration magnitude.
    
    Args:
        acc: accelerometer measurements (3, N) in [g]
    
    Returns:
        acc_normalized: unit vectors (3, N)
    """
    acc_norm = np.linalg.norm(acc, axis=0, keepdims=True)
    return acc / (acc_norm + 1e-10)