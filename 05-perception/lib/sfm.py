from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import least_squares

from lib.camera import project, project_jacobian


@dataclass
class SfmResult:
    points_3d: np.ndarray
    R: np.ndarray
    t: np.ndarray
    inlier_pts1: np.ndarray
    inlier_pts2: np.ndarray
    K: np.ndarray


@dataclass
class BundleAdjustmentResult:
    cameras_init: np.ndarray
    landmarks_init: np.ndarray
    cameras_opt: np.ndarray
    landmarks_opt: np.ndarray
    observations: np.ndarray
    cam_indices: np.ndarray
    lm_indices: np.ndarray
    hessian: np.ndarray
    hessian_reduced: np.ndarray
    errors_per_iter: list[float] = field(default_factory=list)


def two_view_sfm(
    img1: np.ndarray, img2: np.ndarray, K: np.ndarray,
) -> SfmResult:
    """Minimal two-view SfM: features -> E-matrix -> pose -> triangulation."""
    orb = cv2.ORB_create(2000)
    kp1, des1 = orb.detectAndCompute(
        cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if img1.ndim == 3 else img1, None,
    )
    kp2, des2 = orb.detectAndCompute(
        cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if img2.ndim == 3 else img2, None,
    )
    if des1 is None or des2 is None:
        raise ValueError("Could not compute enough descriptors for two-view SfM.")

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    if len(good) < 8:
        raise ValueError("Need at least 8 good matches for the two-view SfM demo.")

    pts1 = np.float64([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float64([kp2[m.trainIdx].pt for m in good])

    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    mask = mask.ravel().astype(bool)
    pts1_in, pts2_in = pts1[mask], pts2[mask]

    _, R, t, mask2 = cv2.recoverPose(E, pts1_in, pts2_in, K)
    mask2 = mask2.ravel() > 0
    pts1_in, pts2_in = pts1_in[mask2], pts2_in[mask2]

    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t])

    pts4d = cv2.triangulatePoints(P1, P2, pts1_in.T, pts2_in.T)
    pts3d = (pts4d[:3] / pts4d[3:4]).T

    return SfmResult(
        points_3d=pts3d, R=R, t=t.ravel(),
        inlier_pts1=pts1_in, inlier_pts2=pts2_in, K=K,
    )


def bundle_adjustment_demo(
    n_cameras: int = 6,
    n_landmarks: int = 40,
    noise_px: float = 1.0,
    rng: np.random.Generator | None = None,
) -> BundleAdjustmentResult:
    """Synthetic BA: generate cameras + landmarks, add noise, optimise.

    Returns result with initial/final state and Hessian structure.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    f = 500.0
    K = np.array([[f, 0, 320], [0, f, 240], [0, 0, 1]])

    landmarks_gt = rng.uniform(-2, 2, (n_landmarks, 3))
    landmarks_gt[:, 2] = rng.uniform(3, 8, n_landmarks)

    cameras_gt = np.zeros((n_cameras, 6))
    for i in range(n_cameras):
        cameras_gt[i, :3] = rng.normal(0, 0.05, 3)
        cameras_gt[i, 3:] = np.array([rng.uniform(-1, 1), rng.uniform(-0.5, 0.5), 0]) * 0.3 * i

    cam_indices = []
    lm_indices = []
    observations = []
    for j in range(n_cameras):
        rvec = cameras_gt[j, :3]
        tvec = cameras_gt[j, 3:]
        R, _ = cv2.Rodrigues(rvec)
        for k in range(n_landmarks):
            X_c = R @ landmarks_gt[k] + tvec
            if X_c[2] < 0.1:
                continue
            uv = project(X_c, K)
            if 0 <= uv[0] < 640 and 0 <= uv[1] < 480:
                uv_noisy = uv + rng.normal(0, noise_px, 2)
                cam_indices.append(j)
                lm_indices.append(k)
                observations.append(uv_noisy)

    cam_indices = np.array(cam_indices)
    lm_indices = np.array(lm_indices)
    observations = np.array(observations)

    cameras_init = cameras_gt + rng.normal(0, 0.02, cameras_gt.shape)
    landmarks_init = landmarks_gt + rng.normal(0, 0.1, landmarks_gt.shape)
    cameras_fixed = cameras_gt[[0]].copy()
    cameras_init[0] = cameras_fixed[0]

    n_obs = len(observations)
    n_cam_params = (n_cameras - 1) * 6
    n_lm_params = n_landmarks * 3

    def unpack_state(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cams_free = x[:n_cam_params].reshape(n_cameras - 1, 6)
        lms = x[n_cam_params:].reshape(n_landmarks, 3)
        cams = np.vstack([cameras_fixed, cams_free])
        return cams, lms

    def residuals(x: np.ndarray) -> np.ndarray:
        cams, lms = unpack_state(x)
        res = np.empty(n_obs * 2)
        for i in range(n_obs):
            j, k_ = cam_indices[i], lm_indices[i]
            R, _ = cv2.Rodrigues(cams[j, :3])
            X_c = R @ lms[k_] + cams[j, 3:]
            uv = project(X_c, K)
            res[2 * i: 2 * i + 2] = observations[i] - uv
        return res

    x0 = np.concatenate([cameras_init[1:].ravel(), landmarks_init.ravel()])
    errors_per_iter: list[float] = []

    def record_error(x: np.ndarray) -> None:
        r = residuals(x)
        errors_per_iter.append(np.sqrt(np.mean(r**2)))

    x = x0.copy()
    record_error(x)
    for _ in range(6):
        result = least_squares(residuals, x, method="trf", max_nfev=5)
        x = result.x
        record_error(x)
        if result.cost < 1e-12:
            break

    cameras_opt, landmarks_opt = unpack_state(x)
    cameras_init_full = np.vstack([cameras_fixed, cameras_init[1:]])

    # Build approximate Hessian structure (J^T J sparsity)
    J = result.jac if result.jac is not None else np.zeros((n_obs * 2, n_cam_params + n_lm_params))
    H = J.T @ J
    # Schur complement on landmark block
    Htt = H[:n_cam_params, :n_cam_params]
    Htx = H[:n_cam_params, n_cam_params:]
    Hxx = H[n_cam_params:, n_cam_params:]
    diag_inv = np.zeros_like(Hxx)
    for k_ in range(n_landmarks):
        s = slice(k_ * 3, k_ * 3 + 3)
        blk = Hxx[s, s]
        det = np.linalg.det(blk)
        if abs(det) > 1e-12:
            diag_inv[s, s] = np.linalg.inv(blk)
    H_reduced = Htt - Htx @ diag_inv @ Htx.T

    return BundleAdjustmentResult(
        cameras_init=cameras_init_full,
        landmarks_init=landmarks_init,
        cameras_opt=cameras_opt,
        landmarks_opt=landmarks_opt,
        observations=observations,
        cam_indices=cam_indices,
        lm_indices=lm_indices,
        hessian=H,
        hessian_reduced=H_reduced,
        errors_per_iter=errors_per_iter,
    )
