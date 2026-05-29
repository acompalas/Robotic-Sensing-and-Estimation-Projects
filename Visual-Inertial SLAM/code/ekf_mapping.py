import numpy as np
from pr3_utils import projection, projectionJacobian, inversePose


# Regular camera -> optical frame rotation
# IMU convention: x=forward, y=left, z=up
# Optical convention: x=right, y=down, z=forward
oTr = np.array([[0, -1,  0, 0],
                [0,  0, -1, 0],
                [1,  0,  0, 0],
                [0,  0,  0, 1]], dtype=float)

# P in R^{3x4}: used in Jacobian as P.T (4x3)
# d(m_hom)/d(m) = P.T  (from lecture chain rule derivation)
P = np.eye(4)[:3, :]   # (3, 4)


def get_OTI_l(extL_T_imu):
    """IMU -> left camera optical frame. oTi = oTr @ inv(ITL)"""
    return oTr @ inversePose(extL_T_imu[None])[0]


def get_OTI_r(extR_T_imu):
    """IMU -> right camera optical frame. oTi = oTr @ inv(ITR)"""
    return oTr @ inversePose(extR_T_imu[None])[0]


def is_valid_obs(z_i):
    """
    Valid observation: reject only the exact [-1,-1,-1,-1] sentinel.
    Valid pixels can be slightly negative near image borders.
    Use isclose for float safety.
    """
    return not np.all(np.isclose(z_i, -1.0))


def triangulate_stereo(K_l, K_r, OTI_l, OTI_r, T_world_imu, z_i,
                        min_disparity=2.0, max_depth=50.0):
    """
    Initialize landmark via stereo triangulation.
    Uses disparity in left optical frame.

    Returns m_world (3,) or None if geometry is degenerate.

    Args:
        min_disparity: minimum pixel disparity (ul - ur) to accept
        max_depth:     maximum depth in meters to accept
    """
    ul, vl, ur, vr = z_i

    disparity = ul - ur
    if abs(disparity) < min_disparity:
        return None

    fu_l = K_l[0, 0]
    fv_l = K_l[1, 1]
    cu_l = K_l[0, 2]
    cv_l = K_l[1, 2]

    # Baseline: distance between optical centers in left optical frame
    c_l_imu = inversePose(OTI_l[None])[0][:3, 3]
    c_r_imu = inversePose(OTI_r[None])[0][:3, 3]
    b_vec   = OTI_l[:3, :3] @ (c_l_imu - c_r_imu)
    b       = abs(b_vec[0])

    # Depth in left optical frame
    z_depth = fu_l * b / disparity
    if z_depth <= 0 or z_depth > max_depth:
        return None

    # 3D point in left optical frame
    x_opt = (ul - cu_l) * z_depth / fu_l
    y_opt = (vl - cv_l) * z_depth / fv_l
    m_opt = np.array([x_opt, y_opt, z_depth, 1.0])

    # left optical -> IMU -> world
    T_imu_opt = inversePose(OTI_l[None])[0]
    m_world   = T_world_imu @ T_imu_opt @ m_opt

    return m_world[:3]


def project_single(K, OTI, T_imu_world, m_j):
    """
    Project landmark into one camera. Returns (2,) [u,v] or None if behind camera.
    """
    m_hom = np.append(m_j, 1.0)
    q = OTI @ T_imu_world @ m_hom
    if q[2] <= 1e-6:
        return None
    p = K @ projection(q[None])[0][:3]
    return p[:2]


def project_stereo(K_l, K_r, OTI_l, OTI_r, T_imu_world, m_j):
    """
    Project landmark into stereo [ul, vl, ur, vr].
    Returns (4,) or None if behind either camera.
    """
    zl = project_single(K_l, OTI_l, T_imu_world, m_j)
    zr = project_single(K_r, OTI_r, T_imu_world, m_j)
    if zl is None or zr is None:
        return None
    return np.array([zl[0], zl[1], zr[0], zr[1]])


def landmark_jacobian_stereo(K_l, K_r, OTI_l, OTI_r, T_imu_world, m_j):
    """
    Jacobian of stereo observation w.r.t. landmark position m_j.

    For each camera c in {L, R} (chain rule from lecture):
        q_c  = OTI_c @ T_imu_world @ m_hom
        H_c  = K_c[:2,:] @ (dpi/dq)(q_c)[:3,:] @ OTI_c @ T_imu_world @ P.T

    Stacked result: H_j = [H_l; H_r]  (4x3)
    """
    m_hom = np.append(m_j, 1.0)

    # Left camera
    q_l   = OTI_l @ T_imu_world @ m_hom
    Jpi_l = projectionJacobian(q_l[None])[0][:3, :]
    H_l   = K_l[:2, :] @ Jpi_l @ OTI_l @ T_imu_world @ P.T   # (2, 3)

    # Right camera
    q_r   = OTI_r @ T_imu_world @ m_hom
    Jpi_r = projectionJacobian(q_r[None])[0][:3, :]
    H_r   = K_r[:2, :] @ Jpi_r @ OTI_r @ T_imu_world @ P.T   # (2, 3)

    return np.vstack([H_l, H_r])   # (4, 3)


def landmark_mapping(poses, features, K_l, K_r, extL_T_imu, extR_T_imu,
                     subsample=5, V_diag=4.0,
                     min_disparity=2.0, max_depth=50.0, innov_gate=50.0):
    """
    Task 3: EKF landmark mapping with fixed IMU trajectory.

    State:      mu_m in R^{3M}  (stacked landmark positions)
    Covariance: per-landmark 3x3 blocks (block-decoupled approximation).
    No prediction step (landmarks are static).

    Per-landmark EKF update:
        S_inn   = H_j @ Sigma_j @ H_j.T + V
        K_j     = Sigma_j @ H_j.T @ inv(S_inn)
        mu_j   += K_j @ (z_j - z_hat_j)
        Sigma_j = (I - K_j @ H_j) @ Sigma_j

    Inputs:
        poses:         (T, 4, 4)  world_T_imu from Task 1
        features:      (4, M, T)  pixel observations
        K_l, K_r:      (3, 3)     camera intrinsics
        extL_T_imu:    (4, 4)     ITL: left cam -> IMU
        extR_T_imu:    (4, 4)     ITR: right cam -> IMU
        subsample:     int         use every N-th landmark
        V_diag:        float       per-pixel observation noise variance
        min_disparity: float       minimum disparity for triangulation (pixels)
        max_depth:     float       maximum depth for triangulation (meters)
        innov_gate:    float       maximum innovation norm to accept update (pixels)

    Outputs:
        mu_m:    (M_used, 3)   estimated landmark positions in world frame
        lm_ids:  (M_used,)     indices into original feature array
    """
    T_steps = features.shape[2]
    M_total = features.shape[1]

    V = np.eye(4) * V_diag

    OTI_l = get_OTI_l(extL_T_imu)
    OTI_r = get_OTI_r(extR_T_imu)
    print(f"  OTI_l:\n{OTI_l}")
    print(f"  OTI_r:\n{OTI_r}")

    # --- Initialize landmarks via stereo triangulation ---
    lm_ids_all  = np.arange(0, M_total, subsample)
    n_candidates = len(lm_ids_all)
    print(f"  Initializing from {n_candidates} candidates "
          f"(min_disp={min_disparity}, max_depth={max_depth}m)...")

    mu_list    = []
    sigma_list = []
    lm_ids     = []

    for i, lm_idx in enumerate(lm_ids_all):
        if i % 100 == 0:
            print(f"  Initializing... {i}/{n_candidates} candidates processed, "
                  f"{len(mu_list)} initialized so far")
        for t in range(T_steps):
            z_i = features[:, lm_idx, t]
            if not is_valid_obs(z_i):
                continue
            m_world = triangulate_stereo(
                K_l, K_r, OTI_l, OTI_r, poses[t], z_i,
                min_disparity=min_disparity, max_depth=max_depth
            )
            if m_world is None:
                continue
            mu_list.append(m_world)
            sigma_list.append(np.eye(3) * 1.0)
            lm_ids.append(lm_idx)
            break

    mu_m   = np.array(mu_list)
    sigmas = np.array(sigma_list)
    lm_ids = np.array(lm_ids)
    M      = mu_m.shape[0]
    print(f"  {M} landmarks initialized")

    # --- EKF update loop ---
    for t in range(T_steps):
        T_imu_world = inversePose(poses[t:t+1])[0]

        for j, lm_idx in enumerate(lm_ids):
            z_i = features[:, lm_idx, t]
            if not is_valid_obs(z_i):
                continue

            z_hat = project_stereo(K_l, K_r, OTI_l, OTI_r, T_imu_world, mu_m[j])
            if z_hat is None:
                continue

            H_j   = landmark_jacobian_stereo(K_l, K_r, OTI_l, OTI_r, T_imu_world, mu_m[j])
            innov = z_i.astype(float) - z_hat

            if np.linalg.norm(innov) > innov_gate:
                continue

            S_inn = H_j @ sigmas[j] @ H_j.T + V
            K_j   = sigmas[j] @ H_j.T @ np.linalg.inv(S_inn)
            mu_m[j]   += K_j @ innov
            sigmas[j]  = (np.eye(3) - K_j @ H_j) @ sigmas[j]

        if t % 500 == 0:
            print(f"  t={t}/{T_steps}")

    return mu_m, lm_ids