from __future__ import annotations

import cv2
import numpy as np


def rectify_stereo(
    imgL: np.ndarray,
    imgR: np.ndarray,
    K: np.ndarray,
    baseline: float,
    dist: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Rectify a stereo pair assuming identical cameras with horizontal baseline."""
    h, w = imgL.shape[:2]
    if dist is None:
        dist = np.zeros(5)

    R = np.eye(3)
    T = np.array([baseline, 0.0, 0.0])
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K, dist, K, dist, (w, h), R, T, alpha=0,
    )
    map1L, map2L = cv2.initUndistortRectifyMap(K, dist, R1, P1, (w, h), cv2.CV_32FC1)
    map1R, map2R = cv2.initUndistortRectifyMap(K, dist, R2, P2, (w, h), cv2.CV_32FC1)

    rectL = cv2.remap(imgL, map1L, map2L, cv2.INTER_LINEAR)
    rectR = cv2.remap(imgR, map1R, map2R, cv2.INTER_LINEAR)
    return rectL, rectR


def compute_disparity(
    imgL: np.ndarray,
    imgR: np.ndarray,
    num_disparities: int = 96,
    block_size: int = 11,
) -> np.ndarray:
    """Compute disparity using SGBM. Returns float32 disparity in pixels."""
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * 3 * block_size**2,
        P2=32 * 3 * block_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
    )
    disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
    return disp


def disparity_to_depth(
    disparity: np.ndarray, f: float, baseline: float,
) -> np.ndarray:
    """Convert disparity to depth. Invalid pixels (d<=0) become inf."""
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = np.where(disparity > 0, f * baseline / disparity, np.inf)
    return depth


def triangulate_points(
    pts1: np.ndarray,
    pts2: np.ndarray,
    P1: np.ndarray,
    P2: np.ndarray,
) -> np.ndarray:
    """Triangulate 3D points from 2D correspondences and projection matrices.

    pts1, pts2: (N, 2). P1, P2: (3, 4).
    Returns: (N, 3) 3D points.
    """
    pts4d = cv2.triangulatePoints(P1, pts1.T.astype(np.float64), P2, pts2.T.astype(np.float64))
    pts3d = (pts4d[:3] / pts4d[3:4]).T
    return pts3d
