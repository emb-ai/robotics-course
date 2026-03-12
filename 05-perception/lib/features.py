from __future__ import annotations

import cv2
import numpy as np


def harris_corner_response(
    img: np.ndarray,
    k: float = 0.04,
    block_size: int = 5,
    ksize: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Harris corner response and per-pixel eigenvalues.

    Returns (response, eigenvalues) where eigenvalues has shape (H, W, 2).
    """
    img = img.astype(np.float32)
    Ix = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=ksize)
    Iy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=ksize)

    Ixx = cv2.GaussianBlur(Ix * Ix, (block_size, block_size), 0)
    Iyy = cv2.GaussianBlur(Iy * Iy, (block_size, block_size), 0)
    Ixy = cv2.GaussianBlur(Ix * Iy, (block_size, block_size), 0)

    det = Ixx * Iyy - Ixy**2
    trace = Ixx + Iyy
    response = det - k * trace**2

    # eigenvalues of 2x2 structure tensor per pixel
    discriminant = np.sqrt(np.maximum((Ixx - Iyy)**2 + 4 * Ixy**2, 0))
    lam1 = 0.5 * (trace + discriminant)
    lam2 = 0.5 * (trace - discriminant)
    eigenvalues = np.stack([lam1, lam2], axis=-1)

    return response, eigenvalues


def detect_and_describe_orb(
    img: np.ndarray, n_features: int = 1000,
) -> tuple[list[cv2.KeyPoint], np.ndarray]:
    """Detect ORB keypoints and compute descriptors."""
    orb = cv2.ORB_create(nfeatures=n_features)
    kp, des = orb.detectAndCompute(img, None)
    if des is None:
        des = np.empty((0, 32), dtype=np.uint8)
    return kp, des


def match_features_ransac(
    kp1: list[cv2.KeyPoint],
    des1: np.ndarray,
    kp2: list[cv2.KeyPoint],
    des2: np.ndarray,
    ratio_thresh: float = 0.75,
    ransac_thresh: float = 3.0,
) -> tuple[list[cv2.DMatch], np.ndarray]:
    """Match ORB descriptors with ratio test + RANSAC via fundamental matrix.

    Returns (good_matches, inlier_mask).
    """
    if len(des1) < 8 or len(des2) < 8:
        return [], np.array([], dtype=np.uint8)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw = bf.knnMatch(des1, des2, k=2)

    good: list[cv2.DMatch] = []
    for pair in raw:
        if len(pair) == 2 and pair[0].distance < ratio_thresh * pair[1].distance:
            good.append(pair[0])

    if len(good) < 8:
        return good, np.ones(len(good), dtype=np.uint8)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

    _, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, ransac_thresh)
    if mask is None:
        mask = np.ones(len(good), dtype=np.uint8)
    return good, mask.ravel()
