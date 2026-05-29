import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from pr3_utils import load_data, visualize_trajectory_2d
from ekf_predict import imu_prediction
from ekf_mapping import landmark_mapping
from ekf_slam import visual_inertial_slam


parser = argparse.ArgumentParser(description="ECE276A PR3 - Visual-Inertial SLAM")
# Dataset
parser.add_argument("--dataset",       type=str,   default="00",  help="Dataset index: 00, 01, 02")
# Task 2 flag (only needed for dataset02)
parser.add_argument("--compute-features", action="store_true",
                    help="Force recompute Task 2 features (dataset02 only). "
                         "By default, uses cached features if available.")
# Task 3 parameters
parser.add_argument("--subsample_t3",  type=int,   default=5,     help="Task 3 landmark subsampling")
parser.add_argument("--V_t3",          type=float, default=4.0,   help="Task 3 observation noise (pixels)")
# Task 4 parameters
parser.add_argument("--subsample_t4",  type=int,   default=20,    help="Task 4 landmark subsampling")
parser.add_argument("--V",             type=float, default=2.0,   help="Task 4 observation noise (pixels)")
parser.add_argument("--W_trans",       type=float, default=1e-6,  help="IMU translational process noise")
parser.add_argument("--W_rot",         type=float, default=1e-6,  help="IMU rotational process noise")
# Shared filtering parameters
parser.add_argument("--max_depth",     type=float, default=50.0,  help="Max triangulation depth (meters)")
parser.add_argument("--min_disp",      type=float, default=2.0,   help="Min stereo disparity (pixels)")
parser.add_argument("--innov_gate_t3", type=float, default=50.0,  help="Task 3 innovation gate (pixels)")
parser.add_argument("--innov_gate_t4", type=float, default=20.0,  help="Task 4 innovation gate (pixels)")
parser.add_argument("--xi_gate",       type=float, default=0.3,   help="Task 4 delta_xi gate")
# Skip flags for testing (all tasks run by default)
parser.add_argument("--no-task3",      action="store_true",       help="Skip Task 3")
parser.add_argument("--no-task4",      action="store_true",       help="Skip Task 4")
args = parser.parse_args()


if __name__ == '__main__':

    filename = f"../data/dataset{args.dataset}/dataset{args.dataset}.npy"
    v_t, w_t, timestamps, features, K_l, K_r, extL_T_imu, extR_T_imu = load_data(filename)

    print(f"Dataset {args.dataset} loaded:")
    print(f"  v_t:        {v_t.shape}")
    print(f"  w_t:        {w_t.shape}")
    print(f"  timestamps: {timestamps.shape}")
    print(f"  features:   {features.shape}  (4 x M x T)")

    # ------------------------------------------------------------------ #
    #  Task 2: Feature Detection and Tracking (dataset02 only)           #
    # ------------------------------------------------------------------ #
    features_path = f"../data/dataset{args.dataset}/dataset{args.dataset}_features_computed.npy"

    if args.dataset == "02":
        if args.compute_features or not os.path.exists(features_path):
            # Case 1 or 3: generate features
            print("\n--- Task 2: Feature Detection and Tracking ---")
            from feature_tracker import load_imgs, compute_features, visualize_features
            imgs_path = f"../data/dataset{args.dataset}/dataset{args.dataset}_imgs.npy"
            print(f"  Loading images from {imgs_path}...")
            left_imgs, right_imgs = load_imgs(imgs_path)
            features = compute_features(left_imgs, right_imgs)
            print(f"  Computed features shape: {features.shape}")
            np.save(features_path, features)
            print(f"  Saved computed features to {features_path}")
        else:
            # Case 2: load existing features
            print(f"\n--- Task 2: Loading precomputed features from {features_path} ---")
            from feature_tracker import load_imgs, visualize_features
            imgs_path = f"../data/dataset{args.dataset}/dataset{args.dataset}_imgs.npy"
            left_imgs, right_imgs = load_imgs(imgs_path)
            features = np.load(features_path)
            print(f"  Loaded features shape: {features.shape}")

        # Always visualize for dataset02
        visualize_features(
            left_imgs, right_imgs, features,
            t=15, dt=100,
            save_path=f"task2_dataset{args.dataset}.png"
        )

    # ------------------------------------------------------------------ #
    #  Task 1: IMU Localization via EKF Prediction                        #                       #
    # ------------------------------------------------------------------ #
    print("\n--- Task 1: IMU Prediction ---")
    poses, covs = imu_prediction(v_t, w_t, timestamps)
    print(f"  poses shape: {poses.shape}")

    fig1, ax1 = visualize_trajectory_2d(poses, path_name="IMU Dead Reckoning", show_ori=True)
    ax1.set_title("Task 1: IMU Localization via EKF Prediction")
    plt.tight_layout()
    plt.savefig(f"task1_dataset{args.dataset}.png", dpi=150)
    plt.close()
    print(f"  Saved task1_dataset{args.dataset}.png")

    # ------------------------------------------------------------------ #
    #  Task 3: Landmark Mapping via EKF Update                           #
    # ------------------------------------------------------------------ #
    if not args.no_task3:
        print("\n--- Task 3: Landmark Mapping ---")
        mu_m, lm_ids = landmark_mapping(
            poses, features, K_l, K_r, extL_T_imu, extR_T_imu,
            subsample=args.subsample_t3,
            V_diag=args.V_t3,
            min_disparity=args.min_disp,
            max_depth=args.max_depth,
            innov_gate=args.innov_gate_t3
        )
        print(f"  Landmarks estimated: {mu_m.shape[0]}")

        fig3, ax3 = visualize_trajectory_2d(poses, path_name="IMU Trajectory", show_ori=False)
        ax3.scatter(mu_m[:, 0], mu_m[:, 1], s=1, c='k', alpha=0.5, label="Landmarks")
        ax3.set_title("Task 3: Landmark Mapping (fixed trajectory)")
        ax3.legend()
        plt.tight_layout()
        plt.savefig(f"task3_dataset{args.dataset}.png", dpi=150)
        plt.close()
        print(f"  Saved task3_dataset{args.dataset}.png")

    # ------------------------------------------------------------------ #
    #  Task 4: Visual-Inertial SLAM                                       #
    # ------------------------------------------------------------------ #
    if not args.no_task4:
        print("\n--- Task 4: Visual-Inertial SLAM ---")
        print(f"  subsample={args.subsample_t4}, V={args.V}, "
              f"W_trans={args.W_trans}, W_rot={args.W_rot}, "
              f"max_depth={args.max_depth}, min_disp={args.min_disp}")
        poses_slam, mu_m_slam, lm_ids_slam = visual_inertial_slam(
            v_t, w_t, timestamps, features,
            K_l, K_r, extL_T_imu, extR_T_imu,
            subsample=args.subsample_t4,
            V_diag=args.V,
            W_trans=args.W_trans,
            W_rot=args.W_rot,
            min_disparity=args.min_disp,
            max_depth=args.max_depth,
            innov_gate=args.innov_gate_t4,
            delta_xi_gate=args.xi_gate
        )
        print(f"  Landmarks estimated: {mu_m_slam.shape[0]}")

        fig4, ax4 = visualize_trajectory_2d(poses, path_name="IMU Only", show_ori=False)
        ax4.plot(poses_slam[:, 0, 3], poses_slam[:, 1, 3],
                 'b-', label="SLAM Trajectory")
        ax4.scatter(mu_m_slam[:, 0], mu_m_slam[:, 1],
                    s=1, c='k', alpha=0.3, label="Landmarks")
        ax4.set_title("Task 4: Visual-Inertial SLAM")
        ax4.legend()
        plt.tight_layout()
        plt.savefig(f"task4_dataset{args.dataset}.png", dpi=150)
        plt.close()
        print(f"  Saved task4_dataset{args.dataset}.png")