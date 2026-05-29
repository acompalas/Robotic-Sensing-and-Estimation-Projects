"""
Panorama generation using estimated orientation trajectories.
ECE276A Project 1 - Part 2
Using CYLINDRICAL projection 
"""

import argparse
from pathlib import Path
import numpy as np
import cv2
from scipy.spatial.transform import Rotation


# --- Paths ---
HERE = Path(__file__).resolve().parent
DATA = (HERE / ".." / "data").resolve()
TRAJECTORIES_DIR = DATA / "trajectories"
PANORAMAS_DIR = DATA / "panoramas"
PANORAMAS_DIR.mkdir(parents=True, exist_ok=True)

# Datasets with camera data
TRAIN_CAM_DATASETS = [1, 2, 8, 9]
TEST_CAM_DATASETS = [10, 11]


def load_camera_data(dataset, split='train'):
    """
    Load camera images and timestamps.
    
    Args:
        dataset: dataset number
        split: 'train' or 'test'
    
    Returns:
        cam_timestamps: (N,) array of camera timestamps
        cam_images: list of N RGB images (H, W, 3)
    """
    import pickle
    
    cam_file = DATA / f"{split}set" / "cam" / f"cam{dataset}.p"
    
    if not cam_file.exists():
        raise FileNotFoundError(f"Camera file not found: {cam_file}")
    
    with open(cam_file, 'rb') as f:
        cam_data = pickle.load(f, encoding='latin1')
    
    # Extract data
    if isinstance(cam_data, dict):
        cam_timestamps = cam_data['ts'].flatten()
        cam_images_raw = cam_data['cam']
    else:
        cam_timestamps = cam_data[0]['ts'].flatten()
        cam_images_raw = cam_data[0]['cam']
    
    print(f"  Raw data shape: {cam_images_raw.shape}")
    
    # Handle shape (H, W, C, N) -> (N, H, W, C)
    if cam_images_raw.ndim == 4 and cam_images_raw.shape[2] == 3:
        num_images = cam_images_raw.shape[3]
        height = cam_images_raw.shape[0]  
        width = cam_images_raw.shape[1]
        channels = cam_images_raw.shape[2]
        
        print(f"  Interpreting as: {num_images} images of {height}x{width}x{channels}")
        cam_images_raw = np.transpose(cam_images_raw, (3, 0, 1, 2))
    
    print(f"  Reshaped to: {cam_images_raw.shape}")
    
    # Match timestamps and images
    if cam_images_raw.shape[0] != len(cam_timestamps):
        min_len = min(cam_images_raw.shape[0], len(cam_timestamps))
        cam_images_raw = cam_images_raw[:min_len]
        cam_timestamps = cam_timestamps[:min_len]
    
    cam_images = [cam_images_raw[i] for i in range(cam_images_raw.shape[0])]
    
    # Ensure uint8
    if cam_images[0].dtype != np.uint8:
        max_val = max(img.max() for img in cam_images)
        if max_val <= 1.0:
            cam_images = [(img * 255).astype(np.uint8) for img in cam_images]
        else:
            cam_images = [img.astype(np.uint8) for img in cam_images]
    
    print(f"  Loaded {len(cam_images)} camera images")
    print(f"  Image shape: {cam_images[0].shape}")
    print(f"  Duration: {cam_timestamps[-1] - cam_timestamps[0]:.2f} seconds")
    
    return cam_timestamps, cam_images


def load_trajectory(dataset, method='optimized'):
    """Load saved quaternion trajectory."""
    traj_file = TRAJECTORIES_DIR / f"trajectory_{dataset}_{method}.npz"
    
    if not traj_file.exists():
        raise FileNotFoundError(f"Trajectory file not found: {traj_file}")
    
    data = np.load(traj_file)
    t = data['timestamps']
    q_traj = data['quaternions']
    
    print(f"  Loaded trajectory: {q_traj.shape[0]} quaternions")
    print(f"  Duration: {t[-1] - t[0]:.2f} seconds")
    
    return t, q_traj


def load_vicon_as_quaternions(dataset):
    """
    Load VICON data and convert rotation matrices to quaternions.
    
    Args:
        dataset: dataset number
    
    Returns:
        vicon_timestamps: (T,) array of timestamps
        q_traj: (T, 4) array of quaternions [w, x, y, z]
    """
    import pickle
    
    vicon_file = DATA / "trainset" / "vicon" / f"viconRot{dataset}.p"
    
    if not vicon_file.exists():
        raise FileNotFoundError(f"VICON file not found: {vicon_file}")
    
    with open(vicon_file, 'rb') as f:
        vicon_data = pickle.load(f, encoding='latin1')
    
    # Extract data
    if isinstance(vicon_data, dict):
        vicon_timestamps = vicon_data['ts'].flatten()
        R_vicon = vicon_data['rots']  # (3, 3, T)
    else:
        vicon_timestamps = vicon_data[0]['ts'].flatten()
        R_vicon = vicon_data[0]['rots']
    
    # Convert rotation matrices to quaternions
    T = R_vicon.shape[2]
    q_traj = np.zeros((T, 4))
    
    for i in range(T):
        R = R_vicon[:, :, i]
        rot = Rotation.from_matrix(R)
        q_xyzw = rot.as_quat()  # [x, y, z, w]
        q_traj[i] = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]  # [w, x, y, z]
    
    print(f"  Loaded VICON: {T} rotation matrices")
    print(f"  Converted to quaternions [w, x, y, z]")
    print(f"  Duration: {vicon_timestamps[-1] - vicon_timestamps[0]:.2f} seconds")
    
    return vicon_timestamps, q_traj


def find_closest_quaternion(cam_timestamp, imu_timestamps, q_traj):
    """Find closest-in-the-past quaternion."""
    valid_indices = np.where(imu_timestamps <= cam_timestamp)[0]
    
    if len(valid_indices) == 0:
        return q_traj[0]
    
    idx = valid_indices[-1]
    return q_traj[idx]


def quaternion_to_rotation_matrix(q):
    """Convert quaternion [w,x,y,z] to rotation matrix."""
    rot = Rotation.from_quat([q[1], q[2], q[3], q[0]])  # scipy uses [x,y,z,w]
    return rot.as_matrix()


def stitch_panorama_vectorized(cam_timestamps, cam_images, imu_timestamps, q_traj,
                               pano_width=1920, pano_height=960):
    """
    Vectorized panorama stitching with CYLINDRICAL projection.
    
    Key fixes:
    1. Using cylindrical projection (not full sphere)
    2. Correct camera-to-IMU transformation
    3. Proper vertical mapping using z-coordinate
    """
    panorama = np.zeros((pano_height, pano_width, 3), dtype=np.uint8)
    
    img_height, img_width = cam_images[0].shape[:2]
    
    print(f"\nStitching panorama ({pano_width}x{pano_height})...")
    print(f"Using CYLINDRICAL projection...")
    print(f"Processing {len(cam_images)} images (VECTORIZED)...")
    
    # Camera FOV (TOTAL, not ±)
    fov_h = 60.0  # total horizontal FOV
    fov_v = 45.0  # total vertical FOV
    
    fx = img_width / (2 * np.tan(np.radians(fov_h) / 2))
    fy = img_height / (2 * np.tan(np.radians(fov_v) / 2))
    
    print(f"  Focal lengths: fx={fx:.2f}, fy={fy:.2f}")
    
    # Pre-compute pixel grid
    v_grid, u_grid = np.meshgrid(np.arange(img_height), np.arange(img_width), indexing='ij')
    u_flat = u_grid.flatten().astype(float)
    v_flat = v_grid.flatten().astype(float)
    
    # Camera rays in camera frame (optical axis = z forward)
    x_cam = (u_flat - img_width/2) / fx
    y_cam = (v_flat - img_height/2) / fy
    z_cam = np.ones_like(x_cam)
    
    rays_cam = np.stack([x_cam, y_cam, z_cam], axis=1)
    rays_cam = rays_cam / np.linalg.norm(rays_cam, axis=1, keepdims=True)
    
    # Camera to IMU frame transformation
    # Camera frame: z forward (optical axis), x right, y down
    # IMU frame: x forward, y left, z up
    R_cam_to_imu = np.array([
        [0,  0,  1],  # IMU x = Camera z (optical axis forward)
        [-1, 0,  0],  # IMU y = -Camera x (left)
        [0, -1,  0]   # IMU z = -Camera y (up)
    ])
    
    rays_imu = (R_cam_to_imu @ rays_cam.T).T
    
    for idx, (cam_ts, img) in enumerate(zip(cam_timestamps, cam_images)):
        if idx % 100 == 0:
            print(f"  Processing image {idx+1}/{len(cam_images)}...")
        
        # Get quaternion (IMU to world rotation)
        q = find_closest_quaternion(cam_ts, imu_timestamps, q_traj)
        R_imu_to_world = quaternion_to_rotation_matrix(q)
        
        # Transform rays to world frame
        rays_world = (R_imu_to_world @ rays_imu.T).T
        rays_world = rays_world / np.linalg.norm(rays_world, axis=1, keepdims=True)
        
        # Extract world coordinates
        x_w, y_w, z_w = rays_world[:, 0], rays_world[:, 1], rays_world[:, 2]
        
        # CYLINDRICAL PROJECTION
        # Horizontal: azimuth angle around vertical axis
        theta = np.arctan2(y_w, x_w)  
        
        # Vertical: use z-component directly (height on cylinder)
        # z = 1 (straight up) -> top of panorama
        # z = 0 (horizontal) -> middle
        # z = -1 (straight down) -> bottom
        
        # Map to panorama coordinates
        pano_u = ((theta + np.pi) / (2 * np.pi) * pano_width).astype(int)
        
        # For cylindrical: map z directly to height
        # Normalize z from [-1, 1] to [0, 1] then to panorama height
        z_norm = (z_w + 1) / 2  # map [-1, 1] to [0, 1]
        pano_v = ((1 - z_norm) * pano_height).astype(int)  # flip so top = high z
        
        # Clip to valid range
        pano_u = np.clip(pano_u, 0, pano_width - 1)
        pano_v = np.clip(pano_v, 0, pano_height - 1)
        
        # Assign pixels 
        panorama[pano_v, pano_u] = img[v_flat.astype(int), u_flat.astype(int)]
    
    print("  Done!")
    return panorama


def main():
    parser = argparse.ArgumentParser(
        description='Generate panorama from camera images and orientation trajectory'
    )
    parser.add_argument('--dataset', type=int, required=True,
                       help='Dataset number (1,2,8,9 for train; 10,11 for test)')
    parser.add_argument('--split', type=str, default='train',
                       choices=['train', 'test'],
                       help='Dataset split (default: train)')
    parser.add_argument('--method', type=str, default='optimized',
                       choices=['optimized', 'vicon'],
                       help='Trajectory method: optimized or vicon ground truth')
    parser.add_argument('--width', type=int, default=1920,
                       help='Panorama width in pixels (default: 1920)')
    parser.add_argument('--height', type=int, default=960,
                       help='Panorama height in pixels (default: 960)')
    
    args = parser.parse_args()
    
    # Validate dataset
    if args.split == 'train' and args.dataset not in TRAIN_CAM_DATASETS:
        raise ValueError(f"Dataset {args.dataset} does not have camera data. "
                        f"Train datasets with camera: {TRAIN_CAM_DATASETS}")
    elif args.split == 'test' and args.dataset not in TEST_CAM_DATASETS:
        raise ValueError(f"Dataset {args.dataset} does not have camera data. "
                        f"Test datasets with camera: {TEST_CAM_DATASETS}")
    
    # VICON only available for train set
    if args.method == 'vicon' and args.split != 'train':
        raise ValueError("VICON ground truth only available for train datasets")
    
    print("=" * 80)
    print(f"PANORAMA GENERATION - Dataset {args.dataset}")
    print("=" * 80)
    
    # Load camera data
    print("\n[1/3] Loading camera data...")
    cam_timestamps, cam_images = load_camera_data(args.dataset, split=args.split)
    
    # Load trajectory
    print("\n[2/3] Loading orientation trajectory...")
    if args.method == 'vicon':
        print("  Using VICON ground truth...")
        imu_timestamps, q_traj = load_vicon_as_quaternions(args.dataset)
    else:
        imu_timestamps, q_traj = load_trajectory(args.dataset, method='optimized')
    
    # Stitch panorama
    print("\n[3/3] Stitching panorama...")
    panorama = stitch_panorama_vectorized(
        cam_timestamps, cam_images,
        imu_timestamps, q_traj,
        pano_width=args.width,
        pano_height=args.height
    )
    
    # Save panorama
    output_path = PANORAMAS_DIR / f"panorama_ds{args.dataset}_{args.method}.png"
    cv2.imwrite(str(output_path), cv2.cvtColor(panorama, cv2.COLOR_RGB2BGR))
    print(f"\nSaved panorama: {output_path}")
    
    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)


if __name__ == "__main__":
    main()