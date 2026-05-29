import numpy as np
from scipy.linalg import expm
from pr3_utils import axangle2pose, axangle2adtwist


def imu_prediction(v_t, w_t, timestamps):
    """
    Task 1: EKF prediction step using SE(3) kinematics.

    Mean propagation:
        mu_{t+1|t} = mu_{t|t} @ exp(tau_t * u_hat_t)

    Covariance propagation:
        A_t = exp(-tau_t * curlywedge(u_t))   (6x6)
        Sigma_{t+1|t} = A_t @ Sigma_{t|t} @ A_t.T + tau_t * W

    Inputs:
        v_t:        linear velocities in IMU body frame
        w_t:        angular velocities in IMU body frame
        timestamps: UNIX timestamps in seconds

    Outputs:
        poses:  (T, 4, 4) world_T_imu poses in SE(3)
        covs:   (T, 6, 6) pose covariances
    """
    v_t        = np.asarray(v_t)
    w_t        = np.asarray(w_t)
    timestamps = np.asarray(timestamps).reshape(-1)

    # Normalize to (T, 3) in case data is (3, T)
    if v_t.ndim == 2 and v_t.shape[0] == 3 and v_t.shape[1] != 3:
        v_t = v_t.T
    if w_t.ndim == 2 and w_t.shape[0] == 3 and w_t.shape[1] != 3:
        w_t = w_t.T

    T_steps = v_t.shape[0]
    print(f"  T_steps={T_steps}, v_t={v_t.shape}, w_t={w_t.shape}")

    poses = np.zeros((T_steps, 4, 4))
    covs  = np.zeros((T_steps, 6, 6))

    poses[0] = np.eye(4)
    covs[0]  = np.eye(6) * 1e-4

    # Process noise — tune in Task 4
    W = np.eye(6)
    W[:3, :3] *= 1e-4
    W[3:, 3:] *= 1e-4

    for t in range(T_steps - 1):
        tau = timestamps[t+1] - timestamps[t]
        u   = np.concatenate([v_t[t], w_t[t]])   # (6,) [v; omega]

        # Mean propagation: T_{t+1} = T_t @ exp(tau * u_hat)
        U = axangle2pose((tau * u)[None, :])[0]   # explicit batch dim
        poses[t+1] = poses[t] @ U

        # Covariance propagation: A_t = exp(-tau * curlywedge(u))
        A_t = expm(-tau * axangle2adtwist(u[None, :])[0])
        covs[t+1] = A_t @ covs[t] @ A_t.T + tau * W

    return poses, covs