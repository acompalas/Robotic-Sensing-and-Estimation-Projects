import numpy as np
from scipy.spatial import KDTree


# ICP helpers
def find_correspondences(source, target_tree):
    """
    For each point in source, find the nearest neighbour in target.
    Uses a KD-tree for efficiency (O(N log N) vs O(N^2) brute force).

    source      : (N, 3)
    target_tree : KDTree built from target points
    returns     : distances (N,), indices (N,)
    """
    distances, indices = target_tree.query(source, workers=-1)
    return distances, indices


def best_fit_transform(A, B):
    """
    Compute the least-squares rigid-body transform T* that minimises
        sum ||T*A_i - B_i||^2
    where A and B are corresponding point sets (same size).

    This is the SVD-based solution (Lecture 4 / Kabsch algorithm):
      1. Centre both point clouds
      2. Compute cross-covariance H = A_c^T B_c
      3. SVD: H = U S V^T
      4. R = V U^T  (handle reflection: det(R) must be +1)
      5. t = mean(B) - R * mean(A)

    A, B : (N, 3)
    returns: T (4,4) SE(3) homogeneous transform
    """
    assert A.shape == B.shape

    # Centroids
    cA = A.mean(axis=0)
    cB = B.mean(axis=0)

    # Centre the point clouds
    Ac = A - cA
    Bc = B - cB

    # Cross-covariance matrix
    H = Ac.T @ Bc   # (3, 3)

    # SVD
    U, S, Vt = np.linalg.svd(H)

    # Rotation — fix reflection if det < 0
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1, 1, d])          # flips last singular vector if needed
    R = Vt.T @ D @ U.T              # (3, 3)

    # Translation
    t = cB - R @ cA                 # (3,)

    # Pack into 4x4 homogeneous transform
    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = t
    return T


def apply_transform(T, points):
    """
    Apply 4x4 homogeneous transform T to (N,3) point cloud.
    Returns (N,3).
    """
    N = points.shape[0]
    pts_h = np.hstack([points, np.ones((N, 1))])   # (N, 4)
    return (T @ pts_h.T).T[:, :3]                  # (N, 3)


def icp(source, target, T_init=None, max_iter=50, tol=1e-6, max_dist=None):
    """
    Iterative Closest Point (3-D).

    Finds T such that T @ source ≈ target.

    source   : (N, 3) — the model / canonical point cloud
    target   : (M, 3) — the observed / measured point cloud
    T_init   : (4,4)  — initial guess (identity if None)
    max_iter : int    — maximum iterations
    tol      : float  — convergence threshold on mean-square error change
    max_dist : float  — reject correspondences farther than this (outlier filter)

    returns  : T (4,4), mean_sq_error (float)
    """
    if T_init is None:
        T_init = np.eye(4)

    T_total = T_init.copy()
    src = apply_transform(T_init, source)   # source in current estimate frame

    # Build KD-tree once on the fixed target
    tree = KDTree(target)

    prev_mse = np.inf

    for iteration in range(max_iter):
        # Step 1: find nearest neighbours 
        distances, indices = find_correspondences(src, tree)

        # Step 2: optionally reject outlier correspondences 
        if max_dist is not None:
            mask = distances < max_dist
        else:
            mask = np.ones(len(distances), dtype=bool)

        if mask.sum() < 6:
            # Too few correspondences — stop
            break

        A = src[mask]                   # current source points
        B = target[indices[mask]]       # matched target points

        # Step 3: compute MSE 
        mse = np.mean(distances[mask] ** 2)

        # Step 4: compute best-fit transform for this iteration 
        T_step = best_fit_transform(A, B)

        # Step 5: update source and accumulate total transform 
        src      = apply_transform(T_step, src)
        T_total  = T_step @ T_total

        # Step 6: check convergence 
        if abs(prev_mse - mse) < tol:
            break
        prev_mse = mse

    # Final MSE uses same max_dist mask — consistent with per-iteration MSE
    # so that yaw initialisation comparisons are apples-to-apples
    final_distances, _ = find_correspondences(src, tree)
    if max_dist is not None:
        final_mask = final_distances < max_dist
        final_mse = np.mean(final_distances[final_mask] ** 2) if final_mask.sum() > 0 else np.inf
    else:
        final_mse = np.mean(final_distances ** 2)

    return T_total, final_mse


# Warm-up: pose initialisation by yaw discretisation
def make_yaw_transform(yaw, translation=None):
    """
    Build a 4x4 transform that rotates by `yaw` around the Z axis,
    optionally with a translation.
    """
    T = np.eye(4)
    c, s = np.cos(yaw), np.sin(yaw)
    T[:3, :3] = np.array([[c, -s, 0],
                           [s,  c, 0],
                           [0,  0, 1]])
    if translation is not None:
        T[:3, 3] = translation
    return T


def icp_with_yaw_init(source, target, n_yaw=36, max_iter=50, max_dist=0.10):
    """
    Run ICP from multiple initial yaw angles and return the result
    with the lowest MSE (best correspondence fit).

    "assume the object only rotates around the z axis
    and initialise rotations by discretising the yaw angles."

    We also initialise the translation so the source centroid matches
    the target centroid — this centers both clouds before rotating.
    """
    # Initial translation: align centroids
    t_init = target.mean(axis=0) - source.mean(axis=0)

    yaw_angles = np.linspace(0, 2 * np.pi, n_yaw, endpoint=False)

    best_T   = np.eye(4)
    best_mse = np.inf

    for yaw in yaw_angles:
        T_init = make_yaw_transform(yaw, translation=t_init)
        T, mse = icp(source, target, T_init=T_init,
                     max_iter=max_iter, max_dist=max_dist)
        if mse < best_mse:
            best_mse = mse
            best_T   = T

    return best_T, best_mse


# Main: warm-up test 
def visualize_icp_result(source_pc, target_pc, T):
    """
    Try Open3D visualization first (works on Windows/Mac with GPU).
    Falls back to saving a matplotlib PNG if Open3D fails (WSL/headless).
    """
    try:
        import open3d as o3d
        source_pcd = o3d.geometry.PointCloud()
        source_pcd.points = o3d.utility.Vector3dVector(source_pc.reshape(-1, 3))
        source_pcd.paint_uniform_color([0, 0, 1])

        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(target_pc.reshape(-1, 3))
        target_pcd.paint_uniform_color([1, 0, 0])

        source_pcd.transform(T)
        o3d.visualization.draw_geometries([source_pcd, target_pcd])
    except Exception as e:
        print(f"  Open3D visualization failed ({e}), falling back to matplotlib.")
        save_icp_result(source_pc, target_pc, T, "icp_result")


def save_icp_result(source_pc, target_pc, T, filename):
    """
    Save a 3-view (XY, XZ, YZ) plot of the ICP result to a PNG.
    Works in WSL without a display server.
    Blue = transformed source (canonical model)
    Red  = target (observed point cloud)
    """
    import matplotlib.pyplot as plt

    src_t = apply_transform(T, source_pc)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    views = [(0, 1, "XY"), (0, 2, "XZ"), (1, 2, "YZ")]

    for ax, (xi, yi, label) in zip(axes, views):
        step = max(1, len(src_t) // 2000)   # downsample for speed
        ax.scatter(target_pc[::step, xi], target_pc[::step, yi],
                   s=1, c='red',  alpha=0.5, label='target')
        ax.scatter(src_t[::step,  xi], src_t[::step,  yi],
                   s=1, c='blue', alpha=0.5, label='source (aligned)')
        ax.set_xlabel(label[0])
        ax.set_ylabel(label[1])
        ax.set_title(label)
        ax.axis('equal')
        ax.legend(markerscale=5)

    plt.suptitle(filename)
    plt.tight_layout()
    plt.savefig(filename + ".png", dpi=120)
    plt.close()
    print(f"  Saved: {filename}.png")


if __name__ == "__main__":
    from utils import read_canonical_model, load_pc

    for obj_name in ["liq_container"]:
        print(f"\n=== {obj_name} ===")
        source_pc = read_canonical_model(obj_name)

        for i in range(4):
            target_pc = load_pc(obj_name, i)
            print(f"  PC {i}: source={source_pc.shape}, target={target_pc.shape}")

            T, mse = icp_with_yaw_init(source_pc, target_pc,
                                        n_yaw=36, max_iter=50, max_dist=0.10)

            print(f"  MSE={mse:.6f}")
            print(f"  R=\n{T[:3,:3]}")
            print(f"  t={T[:3,3]}")

            visualize_icp_result(source_pc, target_pc, T)
            save_icp_result(source_pc, target_pc, T, f"icp_{obj_name}_{i}")