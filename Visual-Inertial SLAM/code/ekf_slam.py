import numpy as np
from scipy.linalg import expm
from scipy.sparse import eye as speye, lil_matrix, csr_matrix, bmat
import scipy.sparse as sp
from pr3_utils import axangle2pose, axangle2adtwist, inversePose, projectionJacobian
from ekf_mapping import (get_OTI_l, get_OTI_r, is_valid_obs,
                          triangulate_stereo, project_stereo, P)


def odot(s):
    """
    The odot operator from lecture slide 24, adapted to our [v; omega] convention.

    Our twist vector convention (matching axangle2twist in pr3_utils):
        xi = [v (3,); omega (3,)]  i.e. translation first, rotation second

    For homogeneous s = [s1, s2, s3, s4] in R^4, the odot operator satisfies:
        xi_hat @ s = s^odot @ xi

    With xi = [v; omega], the odot matrix is:
        s^odot = [ s4*I    0      ] in R^{4x6}   (translation cols)
                 [ 0    -skew(s[:3]) ]             (rotation cols)
                 [ 0  0  0  0  0  0 ]

    Returns (4, 6) matrix.
    """
    s1, s2, s3, s4 = s
    return np.array([
        [s4,  0,  0,    0,  s3, -s2],
        [ 0, s4,  0,  -s3,   0,  s1],
        [ 0,  0, s4,   s2, -s1,   0],
        [ 0,  0,  0,    0,   0,   0]
    ], dtype=float)   # (4, 6)


def pose_jacobian_stereo(K_l, K_r, OTI_l, OTI_r, T_imu_world, m_j):
    """
    Jacobian of stereo observation w.r.t. IMU pose perturbation delta_xi.

    From lecture slide 24/25 (separate left/right cameras):
        H_i^L = -K_l[:2,:] @ Jpi_l[:3,:] @ OTI_l @ (T^{-1} @ m_hom)^odot
        H_i^R = -K_r[:2,:] @ Jpi_r[:3,:] @ OTI_r @ (T^{-1} @ m_hom)^odot

    Returns H_pose: (4, 6)
    """
    m_hom = np.append(m_j, 1.0)
    s     = T_imu_world @ m_hom       # (4,) point in IMU frame
    s_od  = odot(s)                   # (4, 6)

    # Left
    q_l   = OTI_l @ s
    Jpi_l = projectionJacobian(q_l[None])[0][:3, :]   # (3, 4)
    H_l   = -K_l[:2, :] @ Jpi_l @ OTI_l @ s_od       # (2, 6)

    # Right
    q_r   = OTI_r @ s
    Jpi_r = projectionJacobian(q_r[None])[0][:3, :]   # (3, 4)
    H_r   = -K_r[:2, :] @ Jpi_r @ OTI_r @ s_od       # (2, 6)

    return np.vstack([H_l, H_r])   # (4, 6)


def landmark_jacobian_stereo(K_l, K_r, OTI_l, OTI_r, T_imu_world, m_j):
    """
    Jacobian of stereo observation w.r.t. landmark position m_j.

    H_lm = [K_l[:2,:] @ Jpi_l[:3,:] @ OTI_l @ T_imu_world @ P.T]
            [K_r[:2,:] @ Jpi_r[:3,:] @ OTI_r @ T_imu_world @ P.T]

    Returns H_lm: (4, 3)
    """
    m_hom = np.append(m_j, 1.0)

    # Left
    q_l   = OTI_l @ T_imu_world @ m_hom
    Jpi_l = projectionJacobian(q_l[None])[0][:3, :]
    H_l   = K_l[:2, :] @ Jpi_l @ OTI_l @ T_imu_world @ P.T   # (2, 3)

    # Right
    q_r   = OTI_r @ T_imu_world @ m_hom
    Jpi_r = projectionJacobian(q_r[None])[0][:3, :]
    H_r   = K_r[:2, :] @ Jpi_r @ OTI_r @ T_imu_world @ P.T   # (2, 3)

    return np.vstack([H_l, H_r])   # (4, 3)


def visual_inertial_slam(v_t, w_t, timestamps, features,
                         K_l, K_r, extL_T_imu, extR_T_imu,
                         subsample=20, V_diag=2.0,
                         W_trans=1e-6, W_rot=1e-6,
                         min_disparity=2.0, max_depth=50.0,
                         innov_gate=20.0, delta_xi_gate=0.3):
    """
    Task 4: Visual-Inertial SLAM with full joint covariance (sparse).

    Joint state:
        x = [delta_xi (6,); m_1 (3,); ...; m_M (3,)]
        Sigma in R^{(6+3M) x (6+3M)}  stored as scipy sparse csr_matrix

    Per-timestep loop:
        1. Predict: propagate pose mean and full joint covariance
        2. Joint update using visible initialized landmarks
           H is sparse (4*N_obs x 6+3M), only pose+observed landmark cols nonzero
        3. Initialize newly seen landmarks using corrected pose
        4. Expand joint covariance for new landmarks

    Prediction:
        F = [F_pose  0], Q = [tau*W  0]
            [0       I]      [0      0]
        Sigma_{t+1|t} = F @ Sigma @ F.T + Q

    Joint update:
        H sparse (4*N_obs x n), shrunk to observed landmarks only (per TA)
        S = H @ Sigma @ H.T + I_kron_V   (dense, small: 4*N_obs x 4*N_obs)
        K = Sigma @ H.T @ inv(S)          (sparse x dense -> dense)
        delta_x = K @ innov
        Sigma = (I-KH) @ Sigma @ (I-KH).T + K @ R @ K.T  (Joseph form)

    Pose mean update on SE(3):
        T <- T_pred @ exp(delta_xi_hat)

    Landmark mean update (Euclidean):
        m_j <- m_j + delta_m_j
    """
    v_t        = np.asarray(v_t)
    w_t        = np.asarray(w_t)
    timestamps = np.asarray(timestamps).reshape(-1)

    if v_t.ndim == 2 and v_t.shape[0] == 3 and v_t.shape[1] != 3:
        v_t = v_t.T
    if w_t.ndim == 2 and w_t.shape[0] == 3 and w_t.shape[1] != 3:
        w_t = w_t.T

    T_steps = v_t.shape[0]
    M_total = features.shape[1]

    V = np.eye(4) * V_diag

    # Process noise (pose block only)
    W = np.eye(6)
    W[:3, :3] *= W_trans
    W[3:, 3:] *= W_rot

    OTI_l = get_OTI_l(extL_T_imu)
    OTI_r = get_OTI_r(extR_T_imu)

    lm_candidates = np.arange(0, M_total, subsample)

    # --- State initialization ---
    T_cur  = np.eye(4)          # world_T_imu (SE3 mean)
    lm_col = {}                 # lm_idx -> 0-based landmark index
    mu_lm  = []                 # list of (3,) means

    # Joint covariance: sparse, starts as 6x6
    Sigma = csr_matrix(np.eye(6) * 1e-4)

    poses_out    = np.zeros((T_steps, 4, 4))
    poses_out[0] = np.eye(4)

    print(f"  Running VI-SLAM for {T_steps} timesteps, subsample={subsample}...")

    for t in range(T_steps - 1):
        tau = timestamps[t+1] - timestamps[t]
        u   = np.concatenate([v_t[t], w_t[t]])
        M   = len(mu_lm)
        n   = 6 + 3 * M

        # -------------------------------------------------- #
        # Step 1: Predict                                     #
        # -------------------------------------------------- #
        T_pred = T_cur @ axangle2pose((tau * u)[None, :])[0]
        F_pose = expm(-tau * axangle2adtwist(u[None, :])[0])  # (6,6)

        # Build sparse block F (n x n)
        if M == 0:
            F_sp = csr_matrix(F_pose)
        else:
            F_sp = bmat([
                [csr_matrix(F_pose), None           ],
                [None,               speye(3*M, format='csr')]
            ], format='csr')

        # Build sparse Q (n x n)
        Q_pose = csr_matrix(tau * W)
        if M == 0:
            Q_sp = Q_pose
        else:
            Q_sp = bmat([
                [Q_pose, None                          ],
                [None,   csr_matrix((3*M, 3*M))        ]
            ], format='csr')

        Sigma = F_sp @ Sigma @ F_sp.T + Q_sp
        Sigma = 0.5 * (Sigma + Sigma.T)   # enforce symmetry

        # -------------------------------------------------- #
        # Step 2: Joint update using initialized landmarks    #
        # -------------------------------------------------- #
        T_imu_world = inversePose(T_pred[None])[0]

        # Collect visible initialized landmarks (shrink H per TA guidance)
        obs_list = []
        for lm_idx, col_idx in lm_col.items():
            z_i = features[:, lm_idx, t+1]
            if not is_valid_obs(z_i):
                continue
            z_hat = project_stereo(K_l, K_r, OTI_l, OTI_r,
                                   T_imu_world, mu_lm[col_idx])
            if z_hat is None:
                continue
            innov_i = z_i.astype(float) - z_hat
            # Innovation gating: skip landmarks with large prediction error
            if np.linalg.norm(innov_i) > innov_gate:
                continue
            obs_list.append((col_idx, z_i.astype(float), z_hat))

        if len(obs_list) > 0:
            N_obs = len(obs_list)

            # Build sparse H (4*N_obs x n) - lil for efficient construction
            H_lil = lil_matrix((4 * N_obs, n))
            innov = np.zeros(4 * N_obs)

            for k, (col_idx, z_i, z_hat) in enumerate(obs_list):
                m_j = mu_lm[col_idx]
                # Pose block
                H_lil[4*k:4*k+4, :6] = pose_jacobian_stereo(
                    K_l, K_r, OTI_l, OTI_r, T_imu_world, m_j
                )
                # Landmark block
                lm_start = 6 + 3 * col_idx
                H_lil[4*k:4*k+4, lm_start:lm_start+3] = landmark_jacobian_stereo(
                    K_l, K_r, OTI_l, OTI_r, T_imu_world, m_j
                )
                innov[4*k:4*k+4] = z_i - z_hat

            H   = H_lil.tocsr()
            R   = np.kron(np.eye(N_obs), V)              # (4N, 4N) dense - small

            # S = H @ Sigma @ H.T + R  -> dense (4N x 4N), small and invertible
            SigHT = Sigma @ H.T                          # sparse (n x 4N)
            S     = (H @ SigHT).toarray() + R            # dense  (4N x 4N)
            K     = SigHT @ np.linalg.inv(S)             # dense  (n x 4N)

            delta_x = K @ innov                          # dense (n,)

            # Sanity gate: if pose correction is too large, skip update
            if np.linalg.norm(delta_x[:6]) > delta_xi_gate:
                T_cur = T_pred
                Sigma = csr_matrix(Sigma.toarray())   # keep as csr, no update
                if t % 500 == 0 or True:
                    pass
                continue  # skip this timestep's update

            # Joseph form covariance update
            KH    = K @ H.toarray()                      # dense (n x n)
            IKH   = np.eye(n) - KH
            Sigma_dense = IKH @ Sigma.toarray() @ IKH.T + K @ R @ K.T
            Sigma_dense = 0.5 * (Sigma_dense + Sigma_dense.T)
            Sigma = csr_matrix(Sigma_dense)

            # Apply pose correction on SE(3)
            T_cur = T_pred @ axangle2pose(delta_x[:6][None, :])[0]

            # Apply landmark corrections
            for lm_idx, col_idx in lm_col.items():
                lm_start = 6 + 3 * col_idx
                mu_lm[col_idx] += delta_x[lm_start:lm_start+3]
        else:
            T_cur = T_pred

        poses_out[t+1] = T_cur

        # -------------------------------------------------- #
        # Step 3: Initialize new landmarks, expand state     #
        # -------------------------------------------------- #
        new_lms = []
        for lm_idx in lm_candidates:
            if lm_idx in lm_col:
                continue
            z_i = features[:, lm_idx, t+1]
            if not is_valid_obs(z_i):
                continue
            m_world = triangulate_stereo(
                K_l, K_r, OTI_l, OTI_r, T_cur, z_i,
                min_disparity=min_disparity, max_depth=max_depth
            )
            if m_world is None:
                continue
            new_lms.append((lm_idx, m_world))

        if len(new_lms) > 0:
            n_new       = len(new_lms)
            n_old       = 6 + 3 * M
            n_new_total = n_old + 3 * n_new

            # Expand Sigma: copy old block, init new landmark blocks
            # Zero cross-covariances for new landmarks (practical approximation)
            Sigma_new = lil_matrix((n_new_total, n_new_total))
            Sigma_new[:n_old, :n_old] = Sigma
            for i in range(n_new):
                idx = n_old + 3 * i
                Sigma_new[idx:idx+3, idx:idx+3] = speye(3) * 1.0
            Sigma = Sigma_new.tocsr()

            for lm_idx, m_world in new_lms:
                col_idx = len(mu_lm)
                lm_col[lm_idx] = col_idx
                mu_lm.append(m_world)

        if t % 500 == 0:
            print(f"  t={t}/{T_steps}, landmarks={len(mu_lm)}, "
                  f"state_dim={6+3*len(mu_lm)}")

    # Convert to arrays
    lm_ids_out = np.array(sorted(lm_col.keys()))
    mu_m_out   = np.array([mu_lm[lm_col[i]] for i in lm_ids_out])

    return poses_out, mu_m_out, lm_ids_out