import numpy as np
import cv2 as cv


def load_imgs(imgs_path):
    """
    Load stereo images from dataset02_imgs.npy.
    The file is a pickled dict with keys 'cam_imgs_L' and 'cam_imgs_R',
    each a list of (H, W) uint8 grayscale arrays.

    Returns:
        left_imgs:  list of (H, W) uint8 arrays, length T
        right_imgs: list of (H, W) uint8 arrays, length T
    """
    data = np.load(imgs_path, allow_pickle=True).item()
    left_imgs  = data['cam_imgs_L']
    right_imgs = data['cam_imgs_R']
    assert len(left_imgs) == len(right_imgs), "Left/right image count mismatch"
    print(f"  Loaded {len(left_imgs)} stereo pairs, "
          f"image size: {left_imgs[0].shape}")
    return left_imgs, right_imgs


def visualize_features(left_imgs, right_imgs, features, t=0, dt=50,
                       max_display=100, save_path=None):
    """
    Visualize features matching Fig. 2 in the project spec.

    Top row:    left image | right image
                BOTH images show blue+red dots for the same features
                green lines connect blue->red within each image
    Bottom row: left[t] | left[t+dt]
                BOTH images show blue+red dots for the same features
                green lines connect blue->red within each image
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    T = features.shape[2]
    t2 = min(t + dt, T - 1)

    left_t  = left_imgs[t]
    right_t = right_imgs[t]
    left_t2 = left_imgs[t2]

    # Valid stereo at time t
    valid_stereo = np.where(~np.isclose(features[0, :, t], -1.0))[0]
    if len(valid_stereo) > max_display:
        valid_stereo = valid_stereo[:max_display]

    # Valid at both t and t+dt
    valid_t  = ~np.isclose(features[0, :, t],  -1.0)
    valid_t2 = ~np.isclose(features[0, :, t2], -1.0)
    valid_temporal = np.where(valid_t & valid_t2)[0]
    if len(valid_temporal) > max_display:
        valid_temporal = valid_temporal[:max_display]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ---- Top row: stereo matches ----
    # Both panels show both blue (left cam) and red (right cam) features
    # Green lines connect them within each panel
    for col, img in enumerate([left_t, right_t]):
        axes[0, col].imshow(img, cmap='gray')
        title = f"Left image (t={t})" if col == 0 else f"Right image (t={t})"
        axes[0, col].set_title(title)
        for j in valid_stereo:
            lx, ly = features[0, j, t], features[1, j, t]
            rx, ry = features[2, j, t], features[3, j, t]
            axes[0, col].plot(lx, ly, 'b.', markersize=4)
            axes[0, col].plot(rx, ry, 'r.', markersize=4)
            axes[0, col].plot([lx, rx], [ly, ry], 'g-', linewidth=0.5, alpha=0.6)
        axes[0, col].axis('off')

    axes[0, 0].legend(handles=[
        mpatches.Patch(color='blue', label='Left features'),
        mpatches.Patch(color='red',  label='Right features'),
    ], fontsize=8)

    # ---- Bottom row: temporal tracks ----
    # Both panels show both blue (t) and red (t+dt) features
    # Green lines connect them within each panel
    for col, img in enumerate([left_t, left_t2]):
        axes[1, col].imshow(img, cmap='gray')
        title = f"Left image (t={t})" if col == 0 else f"Left image (t={t2})"
        axes[1, col].set_title(title)
        for j in valid_temporal:
            lx0, ly0 = features[0, j, t],  features[1, j, t]
            lx1, ly1 = features[0, j, t2], features[1, j, t2]
            axes[1, col].plot(lx0, ly0, 'b.', markersize=4)
            axes[1, col].plot(lx1, ly1, 'r.', markersize=4)
            axes[1, col].plot([lx0, lx1], [ly0, ly1], 'g-', linewidth=0.5, alpha=0.6)
        axes[1, col].axis('off')

    axes[1, 0].legend(handles=[
        mpatches.Patch(color='blue', label=f't={t}'),
        mpatches.Patch(color='red',  label=f't={t2}'),
    ], fontsize=8)

    plt.suptitle("Task 2: Feature Detection and Tracking", fontsize=13)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  Saved {save_path}")
    else:
        plt.show()


def compute_features(left_imgs, right_imgs,
                     redetect_interval=5, max_corners=200,
                     quality_level=0.01, min_distance=10,
                     win_size=(15, 15), max_level=2):
    """
    Task 2: Feature detection and tracking for stereo image sequences.

    Produces a features array of shape (4, M, T) matching the format
    of the provided datasets, where:
        features[0, j, t] = lx_j  (left image x pixel of feature j at time t)
        features[1, j, t] = ly_j  (left image y pixel)
        features[2, j, t] = rx_j  (right image x pixel)
        features[3, j, t] = ry_j  (right image y pixel)
        features[:, j, t] = [-1,-1,-1,-1] if feature j not visible at time t

    Two steps per timestep:
        (a) Stereo matching: optical flow from left -> right image
        (b) Temporal tracking: optical flow from left[t-1] -> left[t]

    New features detected every `redetect_interval` frames using
    Shi-Tomasi corner detection (goodFeaturesToTrack).

    Inputs:
        left_imgs:           list of (H, W) uint8 grayscale arrays
        right_imgs:          list of (H, W) uint8 grayscale arrays
        redetect_interval:   detect new corners every N frames
        max_corners:         max corners per detection
        quality_level:       Shi-Tomasi quality threshold (0-1)
        min_distance:        min pixel distance between corners
        win_size:            optical flow window size
        max_level:           optical flow pyramid levels

    Output:
        features: (4, M, T) array
    """
    T = len(left_imgs)
    H, W = left_imgs[0].shape[:2]

    lk_params = dict(
        winSize=win_size,
        maxLevel=max_level,
        criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01)
    )

    gftt_params = dict(
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance,
        blockSize=7
    )

    # feature_tracks[j] = dict: t -> (lx, ly, rx, ry)
    feature_tracks = []
    # active_tracks: list of (global_j, np.array([x, y]))
    active_tracks  = []

    prev_left_gray = None

    for t in range(T):
        left_img  = left_imgs[t]
        right_img = right_imgs[t]

        # Already uint8 grayscale per data exploration
        if left_img.ndim == 3:
            left_gray  = cv.cvtColor(left_img,  cv.COLOR_BGR2GRAY)
            right_gray = cv.cvtColor(right_img, cv.COLOR_BGR2GRAY)
        else:
            left_gray  = left_img
            right_gray = right_img

        # -------------------------------------------------- #
        # Step (b): Temporal tracking left[t-1] -> left[t]  #
        # -------------------------------------------------- #
        survived = []
        if prev_left_gray is not None and len(active_tracks) > 0:
            prev_pts = np.array(
                [tr[1] for tr in active_tracks], dtype=np.float32
            ).reshape(-1, 1, 2)

            next_pts, status, _ = cv.calcOpticalFlowPyrLK(
                prev_left_gray, left_gray, prev_pts, None, **lk_params
            )

            for i, (j, _) in enumerate(active_tracks):
                if status[i, 0] == 1:
                    pt = next_pts[i, 0]
                    if 0 <= pt[0] < W and 0 <= pt[1] < H:
                        survived.append((j, pt))

        active_tracks = survived

        # -------------------------------------------------- #
        # Detect new features periodically                   #
        # -------------------------------------------------- #
        if t % redetect_interval == 0:
            new_corners = cv.goodFeaturesToTrack(left_gray, **gftt_params)

            if new_corners is not None:
                new_corners = new_corners.reshape(-1, 2)

                existing_pts = np.array(
                    [tr[1] for tr in active_tracks], dtype=np.float32
                ) if active_tracks else np.empty((0, 2))

                for pt in new_corners:
                    if len(existing_pts) > 0:
                        dists = np.linalg.norm(existing_pts - pt, axis=1)
                        if dists.min() < min_distance:
                            continue
                    j = len(feature_tracks)
                    feature_tracks.append({})
                    active_tracks.append((j, pt))

        # -------------------------------------------------- #
        # Step (a): Stereo matching left[t] -> right[t]     #
        # -------------------------------------------------- #
        if len(active_tracks) > 0:
            left_pts = np.array(
                [tr[1] for tr in active_tracks], dtype=np.float32
            ).reshape(-1, 1, 2)

            right_pts, status_r, _ = cv.calcOpticalFlowPyrLK(
                left_gray, right_gray, left_pts, None, **lk_params
            )

            still_active = []
            for i, (j, lpt) in enumerate(active_tracks):
                if status_r[i, 0] == 1:
                    rpt = right_pts[i, 0]
                    if 0 <= rpt[0] < W and 0 <= rpt[1] < H:
                        # Stereo sanity check 1: positive disparity
                        disp = lpt[0] - rpt[0]
                        if disp <= 0:
                            still_active.append((j, lpt))
                            continue
                        # Stereo sanity check 2: small vertical mismatch
                        if abs(lpt[1] - rpt[1]) > 2.0:
                            still_active.append((j, lpt))
                            continue
                        # Valid stereo observation
                        feature_tracks[j][t] = (
                            float(lpt[0]), float(lpt[1]),
                            float(rpt[0]), float(rpt[1])
                        )
                        still_active.append((j, lpt))
                    else:
                        still_active.append((j, lpt))
                else:
                    still_active.append((j, lpt))

            active_tracks = still_active

        prev_left_gray = left_gray.copy()

        if t % 100 == 0:
            print(f"  t={t}/{T}, active_tracks={len(active_tracks)}, "
                  f"total_features={len(feature_tracks)}")

    # -------------------------------------------------- #
    # Pack into (4, M, T) array                         #
    # -------------------------------------------------- #
    M = len(feature_tracks)
    print(f"  Total features: {M}, timesteps: {T}")

    features = np.full((4, M, T), -1.0, dtype=np.float64)

    for j, track in enumerate(feature_tracks):
        for t, (lx, ly, rx, ry) in track.items():
            features[0, j, t] = lx
            features[1, j, t] = ly
            features[2, j, t] = rx
            features[3, j, t] = ry

    return features