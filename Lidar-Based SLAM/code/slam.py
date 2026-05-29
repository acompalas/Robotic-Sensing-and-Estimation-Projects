import numpy as np
import matplotlib.pyplot as plt
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, "..", "data")

# Data loading
def load_data(dataset=20, data_dir=None):
    if data_dir is None:
        data_dir = DATA_DIR
    with np.load(f"{data_dir}/Encoders{dataset}.npz") as d:
        enc_counts = d["counts"]        # (4, N)  FR FL RR RL
        enc_stamps = d["time_stamps"]   # (N,)

    with np.load(f"{data_dir}/Imu{dataset}.npz") as d:
        imu_av     = d["angular_velocity"]      # (3, M)
        imu_stamps = d["time_stamps"]           # (M,)

    with np.load(f"{data_dir}/Hokuyo{dataset}.npz") as d:
        lidar_ranges    = d["ranges"]           # (1081, K)
        lidar_stamps    = d["time_stamps"]      # (K,)
        lidar_angle_min = float(d["angle_min"])
        lidar_angle_max = float(d["angle_max"])
        lidar_range_min = float(d["range_min"])
        lidar_range_max = float(d["range_max"])

    with np.load(f"{data_dir}/Kinect{dataset}.npz") as d:
        disp_stamps = d["disparity_time_stamps"]
        rgb_stamps  = d["rgb_time_stamps"]

    lidar_angles = np.linspace(lidar_angle_min, lidar_angle_max, 1081)

    return dict(
        enc_counts=enc_counts, enc_stamps=enc_stamps,
        imu_av=imu_av, imu_stamps=imu_stamps,
        lidar_ranges=lidar_ranges, lidar_stamps=lidar_stamps,
        lidar_angles=lidar_angles,
        lidar_range_min=lidar_range_min, lidar_range_max=lidar_range_max,
        disp_stamps=disp_stamps, rgb_stamps=rgb_stamps
    )


# Part 1 – Encoder + IMU odometry
METERS_PER_TICK = 0.0022   # wheel diameter 0.254 m, 360 ticks/rev -> pi*0.254/360

def compute_odometry(data):
    """
    Returns poses (N, 3) as [x, y, theta] at each encoder timestamp,
    using exact differential-drive integration.
    v  from encoders (right/left wheel average)
    omega  from IMU yaw rate (z-axis), interpolated to encoder timestamps
    """
    enc_counts = data["enc_counts"]   # (4, N)  FR FL RR RL
    enc_stamps = data["enc_stamps"]   # (N,)
    imu_av     = data["imu_av"]       # (3, M)
    imu_stamps = data["imu_stamps"]   # (M,)

    N = enc_stamps.shape[0]

    # wheel distances per time step 
    # Right wheels: FR(0) + RR(2),  Left wheels: FL(1) + RL(3)
    d_right = (enc_counts[0] + enc_counts[2]) / 2.0 * METERS_PER_TICK
    d_left  = (enc_counts[1] + enc_counts[3]) / 2.0 * METERS_PER_TICK

    # Linear velocity estimate: distance per time step divided by dt
    # tau*v = (d_right + d_left)/2
    tau_v = (d_right + d_left) / 2.0   # displacement in metres this step

    # IMU yaw rate interpolated to encoder stamps 
    imu_yaw_rate = imu_av[2]   # z-axis is yaw
    omega = np.interp(enc_stamps, imu_stamps, imu_yaw_rate)  # (N,)

    # time intervals
    dt = np.diff(enc_stamps)   # (N-1,)

    # exact differential-drive integration 
    poses = np.zeros((N, 3))   # [x, y, theta]  
    for i in range(N - 1):
        x, y, th = poses[i]
        tau   = dt[i]

        # Skip bad timesteps
        if tau < 1e-9:
            poses[i + 1] = poses[i]
            continue

        v = tau_v[i] / tau

        # Midpoint yaw rate 
        w = 0.5 * (omega[i] + omega[i + 1])

        # exact integration (sinc form from lecture notes)
        dth = w * tau
        if abs(w) < 1e-9:          # straight line
            dx = v * tau * np.cos(th)
            dy = v * tau * np.sin(th)
        else:
            dx = v * tau * np.sinc(dth / (2 * np.pi)) * np.cos(th + dth / 2)
            dy = v * tau * np.sinc(dth / (2 * np.pi)) * np.sin(th + dth / 2)

        # Wrap theta to (-pi, pi] for numerical cleanliness.
        # Theta is the robot's heading in the WORLD frame,
        # measured CCW from the +x axis. omega is the body-frame yaw rate
        # from the IMU z-axis (also CCW positive for our robot).
        new_th = (th + dth + np.pi) % (2 * np.pi) - np.pi
        poses[i + 1] = [x + dx, y + dy, new_th]

    return poses


# Part 2b – 2D LiDAR scan matching via ICP
# Cap LiDAR range for occupancy mapping. 
LIDAR_MAX_RANGE_OCC = None   # metres (None = use sensor range_max)

def lidar_scan_to_xy(ranges, angles, range_min, range_max, return_hit_mask=False):
    """
    Convert a single LiDAR scan to valid (x, y) points in the SENSOR frame.

    Default (return_hit_mask=False): returns only real hits, excludes no-returns.
    Used by ICP — no-return ghost points at max range confuse correspondence matching.

    With return_hit_mask=True: also includes no-return beams (capped at effective_max)
    for free-space carving in occupancy mapping, and returns a boolean hit mask.

    ranges : (1081,) range values in metres
    angles : (1081,) angle values in radians
    returns: (N, 2) points in sensor frame
             if return_hit_mask: also (N,) bool mask (True=real hit, False=no-return)
    """
    effective_max = (range_max if LIDAR_MAX_RANGE_OCC is None
                     else min(range_max, LIDAR_MAX_RANGE_OCC))
    effective_min = max(range_min, 0.1)   # per TA: filter <0.1m to reduce ICP jumps

    if not return_hit_mask:
        # ICP mode: real hits only
        valid = (ranges > effective_min) & (ranges < effective_max)
        r = ranges[valid]
        a = angles[valid]
        return np.column_stack([r * np.cos(a), r * np.sin(a)])

    # Occupancy mode: real hits + no-return beams 
    finite         = np.isfinite(ranges)
    hit_mask_full  = finite & (ranges > effective_min) & (ranges < effective_max)
    noret_mask     = finite & (ranges >= effective_max)
    valid          = hit_mask_full | noret_mask
    r_capped       = np.where(noret_mask, effective_max, ranges)
    r = r_capped[valid]
    a = angles[valid]
    pts = np.column_stack([r * np.cos(a), r * np.sin(a)])
    return pts, hit_mask_full[valid]


def icp_2d(source, target, T_init=None, max_iter=50, tol=1e-6, max_dist=0.5):
    """
    2D ICP — finds SE(2) transform T* such that T* @ source ≈ target.

    Works entirely in 2D (x, y). The math is identical to 3D ICP but
    restricted to the plane:
      - Correspondences: nearest neighbour in 2D
      - Best-fit: SVD on 2x2 cross-covariance → 2D rotation + translation
      - Output: 3x3 homogeneous SE(2) matrix

    source   : (N, 2)
    target   : (M, 2)
    T_init   : (3, 3) SE(2) initial guess, identity if None
    returns  : T (3,3), mse (float)
    """
    from scipy.spatial import KDTree

    if T_init is None:
        T_init = np.eye(3)

    # Apply initial guess to source
    def apply_T2(T, pts):
        # pts: (N,2), T: (3,3)
        h = np.hstack([pts, np.ones((len(pts), 1))])  # (N,3)
        return (T @ h.T).T[:, :2]

    src = apply_T2(T_init, source)
    T_total = T_init.copy()
    tree = KDTree(target)
    prev_mse = np.inf

    for _ in range(max_iter):
        # Correspondences 
        dist, idx = tree.query(src)

        # Outlier rejection 
        mask = dist < max_dist
        if mask.sum() < 4:
            break

        A = src[mask]
        B = target[idx[mask]]
        mse = np.mean(dist[mask] ** 2)

        # Best-fit 2D rigid transform (SVD/Kabsch in 2D) 
        cA = A.mean(axis=0)
        cB = B.mean(axis=0)
        Ac = A - cA
        Bc = B - cB

        H = Ac.T @ Bc          # (2,2) cross-covariance
        U, S, Vt = np.linalg.svd(H)

        # Fix reflection
        d = np.linalg.det(Vt.T @ U.T)
        R2 = Vt.T @ np.diag([1, d]) @ U.T   # (2,2) rotation
        t2 = cB - R2 @ cA                    # (2,) translation

        # Pack into 3x3 SE(2)
        T_step = np.eye(3)
        T_step[:2, :2] = R2
        T_step[:2,  2] = t2

        # Update
        src     = apply_T2(T_step, src)
        T_total = T_step @ T_total

        if abs(prev_mse - mse) < tol:
            break
        prev_mse = mse

    # Final MSE with same mask
    dist_f, _ = tree.query(src)
    mask_f = dist_f < max_dist
    final_mse = np.mean(dist_f[mask_f] ** 2) if mask_f.sum() > 0 else np.inf

    return T_total, final_mse


def pose_to_se2(pose):
    """
    Convert [x, y, theta] pose vector to 3x3 SE(2) homogeneous matrix.
    T = | R  t |
        | 0  1 |
    """
    x, y, th = pose
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s, x],
                     [s,  c, y],
                     [0,  0, 1]])


def se2_to_pose(T):
    """
    Extract [x, y, theta] from 3x3 SE(2) matrix.
    """
    x   = T[0, 2]
    y   = T[1, 2]
    th  = np.arctan2(T[1, 0], T[0, 0])
    return np.array([x, y, th])


# LiDAR-to-body extrinsic in SE(2) 
# From RobotConfiguration.pdf: Hokuyo is mounted forward of robot center.
# x offset ~ (514.35 - 330.20) mm = 184.15 mm ~ 0.18415 m forward,
# y offset = 0, yaw = 0 (0-angle is forward, same as body frame).
_lx, _ly, _lth = 0.18415, 0.0, 0.0
T_BODY_LIDAR = np.array([
    [np.cos(_lth), -np.sin(_lth), _lx],
    [np.sin(_lth),  np.cos(_lth), _ly],
    [0,             0,             1  ]
])
T_LIDAR_BODY = np.linalg.inv(T_BODY_LIDAR)


def interp_pose(t, enc_stamps, odom_poses):
    """
    Interpolate [x, y, theta] at time t.
    Theta is unwrapped before interpolation to avoid the +-pi wrap bug.
    """
    x = np.interp(t, enc_stamps, odom_poses[:, 0])
    y = np.interp(t, enc_stamps, odom_poses[:, 1])
    theta_unwrapped = np.unwrap(odom_poses[:, 2])
    theta = np.interp(t, enc_stamps, theta_unwrapped)
    theta = (theta + np.pi) % (2 * np.pi) - np.pi
    return np.array([x, y, theta])


def compute_icp_poses(data, odom_poses, max_dist=0.3, every_nth=10):
    """
    Refine odometry poses using 2D LiDAR scan matching (ICP).

    For each consecutive pair of LiDAR scans:
      1. Interpolate odometry at LiDAR timestamps (with theta unwrap fix)
      2. Compute relative odometry transform, conjugated by LiDAR extrinsic
      3. Run 2D ICP in LiDAR frame, convert result back to body frame
      4. Chain: T_{t+1} = T_t * tT_{t+1}

    every_nth=10 for debugging, set to 1 for final run.
    """
    lidar_ranges    = data["lidar_ranges"]
    lidar_stamps    = data["lidar_stamps"]
    lidar_angles    = data["lidar_angles"]
    lidar_range_min = data["lidar_range_min"]
    lidar_range_max = data["lidar_range_max"]
    enc_stamps      = data["enc_stamps"]

    indices = np.arange(0, lidar_stamps.shape[0], every_nth)
    M = len(indices)
    icp_poses = np.zeros((M, 3))

    # Start ICP trajectory from the interpolated odometry pose at the first
    # LiDAR stamp so ICP and odometry live in the same global frame
    icp_poses[0] = interp_pose(lidar_stamps[indices[0]], enc_stamps, odom_poses)

    print(f"  Running 2D ICP on {M} LiDAR scans (every_nth={every_nth})...")

    ICP_MSE_REJECT = 0.05   # if ICP MSE exceeds this, fall back to odometry

    for k in range(M - 1):
        i  = indices[k]
        i1 = indices[k + 1]

        # Odometry relative transform
        # Always compute this first — used as ICP init AND as fallback
        odom_i  = interp_pose(lidar_stamps[i],  enc_stamps, odom_poses)
        odom_i1 = interp_pose(lidar_stamps[i1], enc_stamps, odom_poses)
        T_body_t   = pose_to_se2(odom_i)
        T_body_t1  = pose_to_se2(odom_i1)
        T_rel_body = np.linalg.inv(T_body_t) @ T_body_t1

        # 2D point clouds in LiDAR frame
        scan_src = lidar_scan_to_xy(lidar_ranges[:, i],  lidar_angles,
                                    lidar_range_min, lidar_range_max)
        scan_tgt = lidar_scan_to_xy(lidar_ranges[:, i1], lidar_angles,
                                    lidar_range_min, lidar_range_max)

        T_curr = pose_to_se2(icp_poses[k])

        # If scan too sparse, fall back to odometry (not frozen pose)
        if len(scan_src) < 10 or len(scan_tgt) < 10:
            icp_poses[k + 1] = se2_to_pose(T_curr @ T_rel_body)
            continue

        # Conjugate odometry to LiDAR frame for ICP initialisation
        T_rel_lidar = T_LIDAR_BODY @ T_rel_body @ T_BODY_LIDAR

        # 2D ICP in LiDAR frame 
        T_rel_icp, mse = icp_2d(scan_tgt, scan_src,
                                 T_init=T_rel_lidar,
                                 max_iter=50,
                                 max_dist=max_dist)

        # If ICP MSE too high, trust odometry instead
        if mse > ICP_MSE_REJECT:
            T_rel_icp_body = T_rel_body   # fall back to odometry
        else:
            # Convert ICP result back to body frame
            T_rel_icp_body = T_BODY_LIDAR @ T_rel_icp @ T_LIDAR_BODY

        # Chain poses 
        icp_poses[k + 1] = se2_to_pose(T_curr @ T_rel_icp_body)

        if (k + 1) % 50 == 0:
            print(f"    scan {k+1}/{M-1}, mse={mse:.5f}")

    return icp_poses, indices




# Part 3 – Occupancy and texture mapping
# Map parameters — 5cm resolution, 40m x 40m
MAP_RES  = 0.05    # metres per cell
MAP_MIN  = -40.0   # metres
MAP_MAX  =  40.0   # metres

def make_map():
    """Initialise an empty log-odds occupancy grid."""
    size = int(np.ceil((MAP_MAX - MAP_MIN) / MAP_RES))
    return {
        'log_odds': np.zeros((size, size), dtype=np.float32),
        'res': MAP_RES,
        'min': MAP_MIN,
        'size': size
    }

def world_to_cell(x, y, m):
    """Convert world (x,y) in metres to integer grid cell (cx, cy)."""
    cx = np.floor((x - m['min']) / m['res']).astype(int)
    cy = np.floor((y - m['min']) / m['res']).astype(int)
    return cx, cy

def in_bounds(cx, cy, m):
    return (cx >= 0) & (cx < m['size']) & (cy >= 0) & (cy < m['size'])

# Log-odds update values — chosen for the Hokuyo UTM-30LX.
LOG_ODDS_OCC  =  2.0
LOG_ODDS_FREE = -0.5


def update_map(m, robot_pose, scan_xy_lidar, hit_mask=None):
    """
    Update occupancy grid from one LiDAR scan.

    robot_pose    : [x, y, theta] of robot body in world frame
    scan_xy_lidar : (N, 2) points in LiDAR sensor frame
    hit_mask      : (N,) boolean — True = real obstacle hit, False = no-return
                    If None, all points treated as real hits.
    """
    from pr2_utils import bresenham2D

    if hit_mask is None:
        hit_mask = np.ones(len(scan_xy_lidar), dtype=bool)

    # Transform: LiDAR -> body -> world 
    T_world_body  = pose_to_se2(robot_pose)
    T_world_lidar = T_world_body @ T_BODY_LIDAR

    N    = len(scan_xy_lidar)
    ones = np.ones((N, 1))
    pts_lidar_h = np.hstack([scan_xy_lidar, ones])
    pts_world   = (T_world_lidar @ pts_lidar_h.T).T[:, :2]

    # Robot position in cells
    rx, ry = world_to_cell(robot_pose[0], robot_pose[1], m)
    rx, ry = int(rx), int(ry)

    for n, (px, py) in enumerate(pts_world):
        ex, ey = world_to_cell(px, py, m)
        ex, ey = int(ex), int(ey)

        if not in_bounds(ex, ey, m):
            continue

        # Cast ray as free for all beams (including no-returns)
        ray  = bresenham2D(rx, ry, ex, ey).astype(int)
        rxs  = ray[0]
        rys  = ray[1]

        if hit_mask[n]:
            # Real hit: free cells exclude endpoint, mark endpoint occupied
            free_mask = in_bounds(rxs[1:-1], rys[1:-1], m)
            m['log_odds'][rys[1:-1][free_mask], rxs[1:-1][free_mask]] += LOG_ODDS_FREE
            m['log_odds'][ey, ex] += LOG_ODDS_OCC
        else:
            # No-return: endpoint is not an obstacle — carve it as free too
            free_mask = in_bounds(rxs[1:], rys[1:], m)
            m['log_odds'][rys[1:][free_mask], rxs[1:][free_mask]] += LOG_ODDS_FREE

    np.clip(m['log_odds'], -10, 10, out=m['log_odds'])


def build_occupancy_map(data, poses, lidar_indices, snapshot_tag=None):
    """
    Build occupancy grid from a trajectory + LiDAR scans.

    poses         : (M, 3) robot poses in world frame
    lidar_indices : (M,)   indices into lidar_ranges for each pose
    snapshot_tag  : if set, saves intermediate maps at 25/50/75/100% progress
                   e.g. "Dataset 20 ICP Occupancy" -> saves PNGs at each step
    """
    print(f"  Building occupancy map ({len(poses)} poses)...")
    m = make_map()

    lidar_ranges    = data["lidar_ranges"]
    lidar_angles    = data["lidar_angles"]
    lidar_range_min = data["lidar_range_min"]
    lidar_range_max = data["lidar_range_max"]

    N = len(poses)
    # Save intermediate snapshots at 25%, 50%, 75%, 100% of poses
    snapshot_steps = {int(N * f) for f in [0.25, 0.50, 0.75, 1.0]}

    for k, (pose, idx) in enumerate(zip(poses, lidar_indices)):
        scan, hit_mask = lidar_scan_to_xy(lidar_ranges[:, idx], lidar_angles,
                                          lidar_range_min, lidar_range_max,
                                          return_hit_mask=True)
        update_map(m, pose, scan, hit_mask)

        if (k + 1) % 100 == 0:
            print(f"    pose {k+1}/{N}")

        if snapshot_tag and (k + 1) in snapshot_steps:
            pct = int(100 * (k + 1) / N)
            plot_occupancy_map(m, title=f"{snapshot_tag} {pct}pct")

    return m


def plot_occupancy_map(m, title="Occupancy Map", trajectory=None, traj_label="trajectory"):
    """
    Display log-odds map as probability via logistic sigmoid.
    Optionally overlays a trajectory (N, 3) in world frame.
    """
    prob = 1.0 / (1.0 + np.exp(-m['log_odds']))   # sigmoid
    plt.figure(figsize=(10, 10))
    plt.imshow(prob, origin='lower', cmap='gray_r',
               vmin=0, vmax=1,
               extent=[MAP_MIN, MAP_MAX, MAP_MIN, MAP_MAX])
    plt.colorbar(label='P(occupied)')
    if trajectory is not None:
        plt.plot(trajectory[:, 0], trajectory[:, 1],
                 'r-', linewidth=1.0, label=traj_label, alpha=0.8)
        plt.plot(trajectory[0, 0],  trajectory[0, 1],  'go', markersize=6, label="start")
        plt.plot(trajectory[-1, 0], trajectory[-1, 1], 'rs', markersize=6, label="end")
        plt.legend(loc='upper right')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title(title)
    plt.tight_layout()
    fname = title.replace(" ", "_") + ".png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved: {fname}")


# Kinect camera parameters
# Depth camera position relative to robot center
KINECT_POS   = np.array([0.18, 0.005, 0.36])   # x, y, z in metres
KINECT_ROLL  = 0.0
KINECT_PITCH = 0.36    # radians (18 deg tilt downward)
KINECT_YAW   = 0.021   # radians

# Intrinsic matrix
K_DEPTH = np.array([[585.05, 0,      242.94],
                    [0,      585.05, 315.84],
                    [0,      0,      1     ]])

def rot_x(a):
    return np.array([[1, 0,       0      ],
                     [0, np.cos(a), -np.sin(a)],
                     [0, np.sin(a),  np.cos(a)]])

def rot_y(a):
    return np.array([[ np.cos(a), 0, np.sin(a)],
                     [0,          1, 0         ],
                     [-np.sin(a), 0, np.cos(a)]])

def rot_z(a):
    return np.array([[np.cos(a), -np.sin(a), 0],
                     [np.sin(a),  np.cos(a), 0],
                     [0,          0,          1]])

# Rotation from depth camera frame to body frame: R = Rz * Ry * Rx
# The Kinect optical frame has Z-forward, Y-down, X-right.
# We first rotate optical->body->aligned, then apply the physical mount angles.
# Optical->standard body: x_body=z_opt, y_body=-x_opt, z_body=-y_opt
R_OPT_TO_BODY = np.array([[ 0,  0,  1],
                           [-1,  0,  0],
                           [ 0, -1,  0]])
R_MOUNT = rot_z(KINECT_YAW) @ rot_y(KINECT_PITCH) @ rot_x(KINECT_ROLL)
R_BODY_DEPTH = R_MOUNT @ R_OPT_TO_BODY

# Full 4x4 body←depth transform
T_BODY_DEPTH = np.eye(4)
T_BODY_DEPTH[:3, :3] = R_BODY_DEPTH
T_BODY_DEPTH[:3,  3] = KINECT_POS


def disparity_to_depth_and_rgb_pixels(disp_img, row_step=1, col_step=1):
    """
    Convert disparity image to depth + RGB pixel locations.
    Formulas from assignment spec.

    disp_img  : (H, W) disparity values (may be subsampled)
    row_step  : original-image row stride used for subsampling (default 1)
    col_step  : original-image col stride used for subsampling (default 1)
    returns   : depth (H,W), rgb_i (H,W), rgb_j (H,W) in original pixel coords
    """
    d  = disp_img.astype(np.float32)
    # Disparity values where dd <= 0 are invalid (d >= 1088 gives dd <= 0).
    valid_disp = d < (3.31 / 0.00304)   # dd > 0 iff d < 1088.5
    dd = -0.00304 * d + 3.31

    # Compute depth only for valid pixels; NaN elsewhere
    valid = valid_disp & (dd > 1e-6)
    depth = np.full_like(dd, np.nan, dtype=np.float32)
    depth[valid] = 1.03 / dd[valid]

    # NaN dd for invalid pixels so rgb coords are also NaN 
    dd = np.where(valid, dd, np.nan)

    # Pixel grid in original image coordinates
    # Per Piazza: rgb_i = column index, rgb_j = row index
    H, W = disp_img.shape
    row_grid, col_grid = np.meshgrid(np.arange(H) * row_step,
                                     np.arange(W) * col_step,
                                     indexing='ij')

    rgb_i = (526.37 * col_grid + 19276 - 7877.07 * dd) / 585.051   
    rgb_j = (526.37 * row_grid + 16662) / 585.051                   

    return depth, rgb_i, rgb_j   # keep float; cast to int after masking


def build_texture_map(data, poses, lidar_stamps_for_poses,
                      dataset=20, floor_thresh=0.1,
                      depth_min=0.5, depth_max=5.0):
    """
    Build 2D floor texture map from Kinect RGBD images.

    For each Kinect frame:
      1. Find closest robot pose by timestamp
      2. Convert disparity → depth → 3D points in depth camera frame
      3. Transform to world frame using robot pose
      4. Keep only floor points (|z_world| < floor_thresh)
      5. Paint grid cells with RGB colour
    """
    import os

    disp_dir = os.path.join(DATA_DIR, "dataRGBD", f"Disparity{dataset}")
    rgb_dir  = os.path.join(DATA_DIR, "dataRGBD", f"RGB{dataset}")
    if not os.path.exists(disp_dir) or not os.path.exists(rgb_dir):
        print(f"  Kinect data not found at {disp_dir} / {rgb_dir}, skipping texture map.")
        return None

    disp_stamps = data["disp_stamps"]
    rgb_stamps  = data["rgb_stamps"]

    # Texture map — stores RGB as float [0,1]
    size = int(np.ceil((MAP_MAX - MAP_MIN) / MAP_RES))
    tex_r = np.zeros((size, size), dtype=np.float32)
    tex_g = np.zeros((size, size), dtype=np.float32)
    tex_b = np.zeros((size, size), dtype=np.float32)
    tex_count = np.zeros((size, size), dtype=np.int32)

    print(f"  Building texture map ({len(disp_stamps)} frames)...")

    for k, t_disp in enumerate(disp_stamps):
        # Files are 1-indexed: disparity{dataset}_{k+1}.png, rgb{dataset}_{k+1}.png
        disp_path = os.path.join(disp_dir, f"disparity{dataset}_{k+1}.png")
        rgb_path  = os.path.join(rgb_dir,  f"rgb{dataset}_{k+1}.png")
        if not os.path.exists(disp_path) or not os.path.exists(rgb_path):
            continue

        import cv2
        disp_img = cv2.imread(disp_path, cv2.IMREAD_ANYDEPTH).astype(np.float32)
        rgb_img  = cv2.imread(rgb_path,  cv2.IMREAD_COLOR)
        rgb_img  = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)  # OpenCV loads BGR

        # Interpolate ICP pose at Kinect timestamp for smooth texture registration
        robot_pose = interp_pose(t_disp, lidar_stamps_for_poses, poses)

        H, W = disp_img.shape

        # Subsample for performance (every 2nd pixel in each dimension)
        STEP = 2
        disp_sub = disp_img[::STEP, ::STEP]
        H_sub, W_sub = disp_sub.shape

        # Disparity → depth + RGB pixel coords (on subsampled grid)
        depth, rgb_i, rgb_j = disparity_to_depth_and_rgb_pixels(disp_sub, row_step=STEP, col_step=STEP)

        # Mask invalid depths: isfinite guard + configurable Kinect range
        depth_valid_mask = np.isfinite(depth) & (depth > depth_min) & (depth < depth_max)

        # 3D points in depth camera frame using subsampled pixel coordinates
        u = np.arange(W_sub) * STEP   # map back to original pixel coords
        v = np.arange(H_sub) * STEP
        uu, vv = np.meshgrid(u, v)
        Z = depth
        X = (uu - K_DEPTH[0, 2]) * Z / K_DEPTH[0, 0]
        Y = (vv - K_DEPTH[1, 2]) * Z / K_DEPTH[1, 1]

        pts_depth = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

        # Apply depth validity mask (removes near-zero dd → huge depth garbage)
        depth_flat = depth_valid_mask.flatten()
        pts_depth  = pts_depth[depth_flat]

        # Transform to world frame: world = T_world_body * T_body_depth * pts
        T_world_body = np.eye(4)
        T_world_body[:2, :2] = pose_to_se2(robot_pose)[:2, :2]
        T_world_body[0, 3]   = robot_pose[0]
        T_world_body[1, 3]   = robot_pose[1]

        T_world_depth = T_world_body @ T_BODY_DEPTH

        pts_h = np.hstack([pts_depth, np.ones((len(pts_depth), 1))])
        pts_world = (T_world_depth @ pts_h.T).T  # (H*W, 4)

        # Keep floor points: z_world near 0
        floor_mask = np.abs(pts_world[:, 2]) < floor_thresh
        floor_pts  = pts_world[floor_mask]

        # Get corresponding RGB — apply same depth+floor masks, cast to int after
        rgb_i_flat = rgb_i.flatten()[depth_flat][floor_mask].astype(int)
        rgb_j_flat = rgb_j.flatten()[depth_flat][floor_mask].astype(int)
        H_rgb, W_rgb = rgb_img.shape[:2]
        # rgb_j = row index  check against H_rgb
        # rgb_i = col index  check against W_rgb
        rgb_valid = ((rgb_j_flat >= 0) & (rgb_j_flat < H_rgb) &
                     (rgb_i_flat >= 0) & (rgb_i_flat < W_rgb))

        floor_pts  = floor_pts[rgb_valid]
        rgb_i_flat = rgb_i_flat[rgb_valid]
        rgb_j_flat = rgb_j_flat[rgb_valid]

        # Map to grid cells — use consistent map dict, explicit int cast
        m_tmp = {'min': MAP_MIN, 'res': MAP_RES, 'size': size}
        cx, cy = world_to_cell(floor_pts[:, 0], floor_pts[:, 1], m_tmp)
        cx = cx.astype(int)
        cy = cy.astype(int)
        cell_valid = in_bounds(cx, cy, m_tmp)

        cx     = cx[cell_valid]
        cy     = cy[cell_valid]
        # Per Piazza: rgb_i is column index, rgb_j is row index
        # numpy image indexing is [row, col] = [rgb_j, rgb_i]
        colors = rgb_img[rgb_j_flat[cell_valid],
                         rgb_i_flat[cell_valid]].astype(np.float32) / 255.0

        # Store as [row=y, col=x] to match occupancy grid convention
        tex_r[cy, cx] += colors[:, 0]
        tex_g[cy, cx] += colors[:, 1]
        tex_b[cy, cx] += colors[:, 2]
        tex_count[cy, cx] += 1

        if (k + 1) % 50 == 0:
            print(f"    frame {k+1}/{len(disp_stamps)}")

    # Average colours
    mask = tex_count > 0
    tex_r[mask] /= tex_count[mask]
    tex_g[mask] /= tex_count[mask]
    tex_b[mask] /= tex_count[mask]

    texture = np.stack([tex_r, tex_g, tex_b], axis=-1)
    return texture


def plot_texture_map(texture, title="Texture Map"):
    plt.figure(figsize=(10, 10))
    # Stored as [row=y, col=x], so no transpose needed
    plt.imshow(texture, origin='lower',
               extent=[MAP_MIN, MAP_MAX, MAP_MIN, MAP_MAX])
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title(title)
    plt.tight_layout()
    fname = title.replace(" ", "_") + ".png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved: {fname}")

# Plotting helpers
def plot_trajectory(poses, title="Robot Trajectory"):
    plt.figure(figsize=(8, 8))
    plt.plot(poses[:, 0], poses[:, 1], 'b-', linewidth=0.8, label="trajectory")
    plt.plot(poses[0, 0],  poses[0, 1],  'go', markersize=8, label="start")
    plt.plot(poses[-1, 0], poses[-1, 1], 'rs', markersize=8, label="end")
    plt.axis("equal")
    plt.grid(True)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(title.replace(" ", "_") + ".png", dpi=150)
    plt.show()
    print(f"Saved: {title.replace(' ', '_')}.png")

# Part 4 – GTSAM Pose Graph Optimization
def build_pose_graph(data, icp_poses, icp_indices,
                     loop_interval=10,
                     proximity_radius=2.0,
                     loop_mse_threshold=0.01,
                     min_pose_gap=50,
                     loop_icp_max_dist=0.3):
    """
    Build and optimize a GTSAM pose graph from ICP poses.

    Odometry factors: consecutive ICP relative poses.
    Loop closure factors (two types):
      (a) Fixed-interval: every loop_interval poses, add ICP edge between
          pose k and pose k+loop_interval.
      (b) Proximity-based: for each pose pair (i,j) where j > i+min_pose_gap
          and euclidean distance < proximity_radius, run ICP. If MSE <
          loop_mse_threshold, add edge.
    Consistency check: only add loop closure if ICP MSE < loop_mse_threshold.

    Returns optimized poses (M, 3).
    """
    import gtsam
    from gtsam import NonlinearFactorGraph, Values, Pose2
    from gtsam import PriorFactorPose2, BetweenFactorPose2
    from gtsam.symbol_shorthand import X

    M               = len(icp_poses)
    lidar_ranges    = data["lidar_ranges"]
    lidar_angles    = data["lidar_angles"]
    lidar_range_min = data["lidar_range_min"]
    lidar_range_max = data["lidar_range_max"]

    # Noise models 
    # Odometry: per Piazza suggestion 0.1, 0.1, 0.05
    odom_sigmas  = np.array([0.1, 0.1, 0.05])   # x, y, theta (m, m, rad)
    odom_noise   = gtsam.noiseModel.Diagonal.Sigmas(odom_sigmas)

    # Loop closure: looser than odometry — ICP may have small residual error.
    # Wrap in Huber robust kernel to prevent a single bad loop edge from corrupting the entire solution.
    loop_sigmas  = np.array([0.3, 0.3, 0.1])
    _loop_base   = gtsam.noiseModel.Diagonal.Sigmas(loop_sigmas)
    try:
        _huber     = gtsam.noiseModel.mEstimator.Huber.Create(1.345)
        loop_noise = gtsam.noiseModel.Robust.Create(_huber, _loop_base)
    except Exception:
        loop_noise = _loop_base

    # Prior on first pose
    prior_sigmas = np.array([1e-6, 1e-6, 1e-6])
    prior_noise  = gtsam.noiseModel.Diagonal.Sigmas(prior_sigmas)

    # Build graph 
    graph  = NonlinearFactorGraph()
    values = Values()

    # Insert all initial pose estimates
    for k in range(M):
        x, y, th = icp_poses[k]
        values.insert(X(k), Pose2(x, y, th))

    # Prior on first pose
    x0, y0, th0 = icp_poses[0]
    graph.add(PriorFactorPose2(X(0), Pose2(x0, y0, th0), prior_noise))

    # Odometry factors 
    print("  Adding odometry factors...")
    for k in range(M - 1):
        T_k   = pose_to_se2(icp_poses[k])
        T_k1  = pose_to_se2(icp_poses[k + 1])
        T_rel = se2_to_pose(np.linalg.inv(T_k) @ T_k1)
        graph.add(BetweenFactorPose2(
            X(k), X(k + 1),
            Pose2(T_rel[0], T_rel[1], T_rel[2]),
            odom_noise))
    print(f"    Added {M-1} odometry factors")

    # Helper: get scan for pose index k
    def get_scan(k):
        idx = icp_indices[k]
        return lidar_scan_to_xy(lidar_ranges[:, idx], lidar_angles,
                                lidar_range_min, lidar_range_max)

    # Helper: run ICP between poses i and j, initialized from current graph estimate
    def icp_between(i, j):
        scan_i = get_scan(i)
        scan_j = get_scan(j)
        if len(scan_i) < 10 or len(scan_j) < 10:
            return None, np.inf
        # Initialize from current graph estimate (values), not frozen icp_poses
        pi = values.atPose2(X(i))
        pj = values.atPose2(X(j))
        T_i = pose_to_se2([pi.x(), pi.y(), pi.theta()])
        T_j = pose_to_se2([pj.x(), pj.y(), pj.theta()])
        T_rel_body  = np.linalg.inv(T_i) @ T_j
        T_rel_lidar = T_LIDAR_BODY @ T_rel_body @ T_BODY_LIDAR
        T_rel_icp, mse = icp_2d(scan_j, scan_i,
                                 T_init=T_rel_lidar,
                                 max_iter=50,
                                 max_dist=loop_icp_max_dist)
        T_rel_body_icp = T_BODY_LIDAR @ T_rel_icp @ T_LIDAR_BODY
        return se2_to_pose(T_rel_body_icp), mse

    # (a) Fixed-interval loop closure 
    print(f"  Adding fixed-interval loop closures (every {loop_interval} poses)...")
    n_fixed = 0
    mse_fixed_accepted = []
    for k in range(0, M - loop_interval, loop_interval):
        j = k + loop_interval
        T_rel, mse = icp_between(k, j)
        if mse < loop_mse_threshold:
            graph.add(BetweenFactorPose2(
                X(k), X(j),
                Pose2(T_rel[0], T_rel[1], T_rel[2]),
                loop_noise))
            n_fixed += 1
            mse_fixed_accepted.append(mse)
    print(f"    Added {n_fixed} fixed-interval loop closure factors")
    if mse_fixed_accepted:
        print(f"    Fixed-interval MSE: min={min(mse_fixed_accepted):.4f}, "
              f"max={max(mse_fixed_accepted):.4f}, "
              f"mean={np.mean(mse_fixed_accepted):.4f}")

    # (b) Proximity-based loop closure 
    print(f"  Adding proximity-based loop closures "
          f"(radius={proximity_radius}m, min_gap={min_pose_gap})...")
    from scipy.spatial import cKDTree
    xy = icp_poses[:, :2]
    kd_tree = cKDTree(xy)

    # query_pairs returns unique (i,j) pairs with i<j and dist<radius — no duplicates
    candidate_pairs = [(i, j) for i, j in kd_tree.query_pairs(proximity_radius)
                       if j - i >= min_pose_gap]
    print(f"    {len(candidate_pairs)} candidate pairs to evaluate...")

    n_prox = 0
    mse_accepted = []
    mse_rejected = []
    for i, j in candidate_pairs:
        T_rel, mse = icp_between(i, j)
        if mse < loop_mse_threshold:
            graph.add(BetweenFactorPose2(
                X(i), X(j),
                Pose2(T_rel[0], T_rel[1], T_rel[2]),
                loop_noise))
            n_prox += 1
            mse_accepted.append(mse)
        else:
            mse_rejected.append(mse)
    print(f"    Added {n_prox} proximity-based loop closure factors")
    if mse_accepted:
        print(f"    Accepted MSE: min={min(mse_accepted):.4f}, "
              f"max={max(mse_accepted):.4f}, "
              f"mean={np.mean(mse_accepted):.4f}")
    if mse_rejected:
        print(f"    Rejected MSE: min={min(mse_rejected):.4f}, "
              f"max={max(mse_rejected):.4f}, "
              f"mean={np.mean(mse_rejected):.4f} "
              f"({len(mse_rejected)} pairs rejected)")

    # Optimize 
    print("  Optimizing pose graph...")
    params = gtsam.LevenbergMarquardtParams()
    params.setVerbosity('ERROR')
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, values, params)
    result = optimizer.optimize()

    # Extract optimized poses
    opt_poses = np.zeros((M, 3))
    for k in range(M):
        p = result.atPose2(X(k))
        opt_poses[k] = [p.x(), p.y(), p.theta()]

    print(f"  Optimization complete.")
    print(f"  Initial error: {graph.error(values):.4f}")
    print(f"  Final error:   {graph.error(result):.4f}")

    return opt_poses


# ─────────────────────────────────────────────
# Main — CLI flags
# ─────────────────────────────────────────────
# Usage:
#   python3 slam.py                                      # run all parts
#   python3 slam.py --parts 1                            # only Part 1
#   python3 slam.py --parts 2                            # only Part 2 (loads Part 1 cache)
#   python3 slam.py --parts 3                            # only Part 3 (loads Part 1+2 cache)
#   python3 slam.py --parts 4                            # only Part 4 (loads Part 1+2 cache)
#   python3 slam.py --parts 3 4                          # Parts 3 and 4
#   python3 slam.py --parts 3 --dataset 20               # Part 3, dataset 20 only
#   python3 slam.py --every-nth 10                       # override ICP subsampling
#   python3 slam.py --max-dist 0.3                       # override ICP max correspondence distance
#   python3 slam.py --parts 3 --lidar-max-range 12       # tune occupancy range cap
#   python3 slam.py --parts 3 --log-odds-free -0.25      # tune free-space update strength
#   python3 slam.py --parts 4 --loop-interval 10         # fixed-interval loop closure gap
#   python3 slam.py --parts 4 --loop-radius 2.0          # proximity loop closure radius
#   python3 slam.py --parts 4 --loop-mse 0.01            # ICP MSE acceptance threshold
#
# Caching: completed parts are saved as cache_ds{N}_part{K}.npz next to slam.py.
# If a required part's cache is missing it runs automatically.

def cache_path(dataset, part):
    return os.path.join(SCRIPT_DIR, f"cache_ds{dataset}_part{part}.npz")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ECE 276A SLAM pipeline")
    parser.add_argument("--parts", type=int, nargs="+", default=[1, 2, 3, 4],
                        choices=[1, 2, 3, 4],
                        help="Which parts to run (default: all)")
    parser.add_argument("--dataset", type=int, nargs="+", default=[20, 21],
                        choices=[20, 21],
                        help="Which datasets to process (default: both)")
    parser.add_argument("--every-nth", type=int, default=1,
                        help="ICP scan subsampling (default: 1 = all scans)")
    parser.add_argument("--max-dist", type=float, default=0.2,
                        help="ICP max correspondence distance in metres (default: 0.2)")
    parser.add_argument("--force", action="store_true",
                        help="Ignore cache and recompute all requested parts")
    parser.add_argument("--loop-interval", type=int, default=10,
                        help="Fixed-interval loop closure gap in poses (default: 10)")
    parser.add_argument("--loop-radius", type=float, default=2.0,
                        help="Proximity loop closure radius in metres (default: 2.0)")
    parser.add_argument("--loop-mse", type=float, default=0.01,
                        help="ICP MSE threshold for loop closure acceptance (default: 0.01)")
    parser.add_argument("--min-pose-gap", type=int, default=50,
                        help="Min pose index gap for proximity loop closure (default: 50)")
    parser.add_argument("--lidar-max-range", type=float, default=None,
                        help="Override LIDAR_MAX_RANGE_OCC for occupancy mapping "
                             "(default: use value in code). Try 10, 12, 15.")
    parser.add_argument("--log-odds-free", type=float, default=None,
                        help="Override LOG_ODDS_FREE (default: use value in code). "
                             "Try -0.2, -0.25, -0.3 for crisper walls.")
    args = parser.parse_args()

    # Apply CLI overrides to module-level tuning constants
    if args.lidar_max_range is not None:
        LIDAR_MAX_RANGE_OCC = args.lidar_max_range
        print(f"  [override] LIDAR_MAX_RANGE_OCC = {LIDAR_MAX_RANGE_OCC} m")
    if args.log_odds_free is not None:
        LOG_ODDS_FREE = args.log_odds_free
        print(f"  [override] LOG_ODDS_FREE = {LOG_ODDS_FREE}")

    run_parts = set(args.parts)

    for dataset in args.dataset:
        print(f"\n{'='*50}")
        print(f"=== Dataset {dataset} ===")
        print(f"{'='*50}")
        data = load_data(dataset)

        # Part 1: odometry 
        p1_cache = cache_path(dataset, 1)
        need_p1  = (1 in run_parts) or args.force or not os.path.exists(p1_cache)

        if need_p1:
            print("\n[Part 1] Running odometry...")
            odom_poses = compute_odometry(data)
            np.savez(p1_cache, odom_poses=odom_poses)
            print(f"  Poses: {odom_poses.shape}")
            print(f"  Final: x={odom_poses[-1,0]:.2f} m, "
                  f"y={odom_poses[-1,1]:.2f} m, "
                  f"θ={np.degrees(odom_poses[-1,2]):.1f}°")
            plot_trajectory(odom_poses,
                            title=f"Dataset {dataset} Odometry Trajectory")
        else:
            print(f"\n[Part 1] Loading cache: {p1_cache}")
            odom_poses = np.load(p1_cache)["odom_poses"]
            print(f"  Loaded: {odom_poses.shape}")

        # Part 2: ICP scan matching 
        p2_cache = cache_path(dataset, 2)
        need_p2  = (2 in run_parts) or args.force or not os.path.exists(p2_cache)

        if need_p2:
            print(f"\n[Part 2] Running ICP "
                  f"(every_nth={args.every_nth}, max_dist={args.max_dist})...")
            icp_poses, icp_indices = compute_icp_poses(
                data, odom_poses,
                max_dist=args.max_dist,
                every_nth=args.every_nth
            )
            np.savez(p2_cache, icp_poses=icp_poses, icp_indices=icp_indices)
            print(f"  Poses: {icp_poses.shape}")
            print(f"  Final: x={icp_poses[-1,0]:.2f} m, "
                  f"y={icp_poses[-1,1]:.2f} m, "
                  f"θ={np.degrees(icp_poses[-1,2]):.1f}°")
            plot_trajectory(icp_poses,
                            title=f"Dataset {dataset} ICP Trajectory")

            # Combined odometry vs ICP plot
            plt.figure(figsize=(10, 10))
            plt.plot(odom_poses[:, 0], odom_poses[:, 1],
                     'b-', linewidth=0.8, alpha=0.5, label="Odometry")
            plt.plot(icp_poses[:, 0], icp_poses[:, 1],
                     'r-', linewidth=0.8, label="ICP")
            plt.plot(icp_poses[0, 0], icp_poses[0, 1], 'go', markersize=8, label="start")
            plt.plot(icp_poses[-1, 0], icp_poses[-1, 1], 'rs', markersize=8, label="end")
            plt.axis("equal")
            plt.grid(True)
            plt.legend()
            plt.xlabel("x (m)")
            plt.ylabel("y (m)")
            plt.title(f"Dataset {dataset} Odometry vs ICP Trajectory")
            plt.tight_layout()
            fname = f"Dataset_{dataset}_Odometry_vs_ICP_Trajectory.png"
            plt.savefig(fname, dpi=150)
            plt.close()
            print(f"  Saved: {fname}")
        else:
            print(f"\n[Part 2] Loading cache: {p2_cache}")
            p2          = np.load(p2_cache)
            icp_poses   = p2["icp_poses"]
            icp_indices = p2["icp_indices"]
            print(f"  Loaded: {icp_poses.shape}")

        icp_lidar_stamps = data["lidar_stamps"][icp_indices]

        # Part 3: occupancy + texture mapping 
        if 3 in run_parts:
            print("\n[Part 3] Building occupancy map...")

            # First-scan sanity check (assignment requirement)
            print("  First-scan sanity check...")
            m0 = make_map()
            first_scan, first_hit_mask = lidar_scan_to_xy(
                data["lidar_ranges"][:, icp_indices[0]],
                data["lidar_angles"],
                data["lidar_range_min"], data["lidar_range_max"],
                return_hit_mask=True
            )
            update_map(m0, icp_poses[0], first_scan, first_hit_mask)
            plot_occupancy_map(m0,
                               title=f"Dataset {dataset} First-Scan Occupancy")

            # Full occupancy map — pass indices directly (no timestamp search)
            occ_map = build_occupancy_map(data, icp_poses, icp_indices,
                                          snapshot_tag=f"Dataset {dataset} ICP Occupancy")
            plot_occupancy_map(occ_map,
                               title=f"Dataset {dataset} Occupancy Map",
                               trajectory=icp_poses,
                               traj_label="ICP trajectory")

            # Texture map
            texture = build_texture_map(data, icp_poses, icp_lidar_stamps,
                                        dataset=dataset)
            if texture is not None:
                plot_texture_map(texture,
                                 title=f"Dataset {dataset} Texture Map")

        # Part 4: GTSAM pose graph optimization
        if 4 in run_parts:
            print("\n[Part 4] Running GTSAM pose graph optimization...")
            p4_cache = cache_path(dataset, 4)
            need_p4  = (4 in run_parts and args.force) or not os.path.exists(p4_cache)

            if need_p4:
                opt_poses = build_pose_graph(
                    data, icp_poses, icp_indices,
                    loop_interval      = args.loop_interval,
                    proximity_radius   = args.loop_radius,
                    loop_mse_threshold = args.loop_mse,
                    min_pose_gap       = args.min_pose_gap,
                    loop_icp_max_dist  = args.max_dist
                )
                np.savez(p4_cache, opt_poses=opt_poses)
                print(f"  Optimized poses: {opt_poses.shape}")
                print(f"  Final: x={opt_poses[-1,0]:.2f} m, "
                      f"y={opt_poses[-1,1]:.2f} m, "
                      f"θ={np.degrees(opt_poses[-1,2]):.1f}°")
            else:
                print(f"  Loading cache: {p4_cache}")
                opt_poses = np.load(p4_cache)["opt_poses"]
                print(f"  Loaded: {opt_poses.shape}")

            # Plot optimized trajectory vs ICP
            plt.figure(figsize=(10, 10))
            plt.plot(icp_poses[:, 0], icp_poses[:, 1],
                     'b-', linewidth=0.8, alpha=0.5, label="ICP")
            plt.plot(opt_poses[:, 0], opt_poses[:, 1],
                     'r-', linewidth=0.8, label="GTSAM optimized")
            plt.plot(opt_poses[0, 0], opt_poses[0, 1], 'go', markersize=8, label="start")
            plt.plot(opt_poses[-1, 0], opt_poses[-1, 1], 'rs', markersize=8, label="end")
            plt.axis("equal")
            plt.grid(True)
            plt.legend()
            plt.xlabel("x (m)")
            plt.ylabel("y (m)")
            plt.title(f"Dataset {dataset} GTSAM Trajectory")
            plt.tight_layout()
            fname = f"Dataset_{dataset}_GTSAM_Trajectory.png"
            plt.savefig(fname, dpi=150)
            plt.close()
            print(f"  Saved: {fname}")

            # Rebuild occupancy + texture maps with optimized poses
            print("  Rebuilding occupancy map with optimized poses...")
            opt_map = build_occupancy_map(data, opt_poses, icp_indices,
                                          snapshot_tag=f"Dataset {dataset} GTSAM Occupancy")
            plot_occupancy_map(opt_map,
                               title=f"Dataset {dataset} GTSAM Occupancy Map",
                               trajectory=opt_poses,
                               traj_label="GTSAM trajectory")

            opt_lidar_stamps = data["lidar_stamps"][icp_indices]
            print("  Rebuilding texture map with optimized poses...")
            opt_texture = build_texture_map(data, opt_poses, opt_lidar_stamps,
                                            dataset=dataset)
            if opt_texture is not None:
                plot_texture_map(opt_texture,
                                 title=f"Dataset {dataset} GTSAM Texture Map")