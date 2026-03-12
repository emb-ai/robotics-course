from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np


@dataclass
class PinholeCamera:
    fx: float
    fy: float
    cx: float
    cy: float
    skew: float = 0.0
    dist_coeffs: np.ndarray = field(default_factory=lambda: np.zeros(5))

    @property
    def K(self) -> np.ndarray:
        return np.array([
            [self.fx, self.skew, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0],
        ])

    @classmethod
    def from_matrix(cls, K: np.ndarray, dist: np.ndarray | None = None) -> PinholeCamera:
        return cls(
            fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2],
            skew=K[0, 1],
            dist_coeffs=dist if dist is not None else np.zeros(5),
        )


def project(X_c: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Project 3D points in camera frame to 2D pixel coordinates.

    X_c: (N, 3) or (3,)  — points in camera frame.
    K:   (3, 3)           — intrinsic matrix.
    Returns: (N, 2) or (2,) pixel coordinates.
    """
    single = X_c.ndim == 1
    X_c = np.atleast_2d(X_c)
    Z = X_c[:, 2:3]
    X_norm = X_c[:, :2] / Z
    uv = (K[:2, :2] @ X_norm.T).T + K[:2, 2]
    return uv.squeeze() if single else uv


def project_jacobian(X_c: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Jacobian d(pi)/d(X_c) for a single point. Returns (2, 3)."""
    fx, fy = K[0, 0], K[1, 1]
    Xc, Yc, Zc = X_c[0], X_c[1], X_c[2]
    return np.array([
        [fx / Zc, 0.0, -fx * Xc / Zc**2],
        [0.0, fy / Zc, -fy * Yc / Zc**2],
    ])


def apply_distortion(
    x_n: np.ndarray, y_n: np.ndarray,
    k1: float = 0, k2: float = 0, k3: float = 0,
    p1: float = 0, p2: float = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply Brown-Conrady distortion to normalised coordinates."""
    r2 = x_n**2 + y_n**2
    radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
    x_d = x_n * radial + 2 * p1 * x_n * y_n + p2 * (r2 + 2 * x_n**2)
    y_d = y_n * radial + p1 * (r2 + 2 * y_n**2) + 2 * p2 * x_n * y_n
    return x_d, y_d


def undistort_points(
    pts: np.ndarray, K: np.ndarray, dist: np.ndarray,
) -> np.ndarray:
    """Wrapper around cv2.undistortPoints. pts: (N, 2)."""
    pts_cv = pts.reshape(-1, 1, 2).astype(np.float64)
    out = cv2.undistortPoints(pts_cv, K, dist, P=K)
    return out.reshape(-1, 2)


def calibrate_camera_from_checkerboard(
    image_paths: Sequence[Path | str],
    board_size: tuple[int, int] = (9, 6),
    square_size: float = 0.025,
) -> tuple[np.ndarray, np.ndarray, float, list, list, dict[str, object]]:
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2) * square_size

    obj_points: list[np.ndarray] = []
    img_points: list[np.ndarray] = []
    img_size: tuple[int, int] | None = None

    for p in image_paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if img_size is None:
            img_size = (img.shape[1], img.shape[0])
        found, corners = cv2.findChessboardCorners(img, board_size, None)
        if found:
            corners_refined = cv2.cornerSubPix(
                img, corners, (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
            )
            obj_points.append(objp)
            img_points.append(corners_refined)

    if not obj_points:
        raise ValueError("No checkerboard corners found in any image.")

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, img_size, None, None,
    )
    per_view_rms = []
    for obj, img, rvec, tvec in zip(obj_points, img_points, rvecs, tvecs):
        reproj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        residual = img.reshape(-1, 2) - reproj.reshape(-1, 2)
        per_view_rms.append(float(np.sqrt(np.mean(np.sum(residual**2, axis=1)))))

    calib_data = {
        "board_size": board_size,
        "square_size": square_size,
        "obj_points": obj_points,
        "img_points": img_points,
        "image_size": img_size,
        "per_view_rms": per_view_rms,
    }
    mean_reproj_error = float(np.mean(per_view_rms))
    return K, dist, mean_reproj_error, rvecs, tvecs, calib_data


# ---------------------------------------------------------------------------
# Fisheye camera model
# ---------------------------------------------------------------------------

@dataclass
class FisheyeCamera:
    """Kannala–Brandt fisheye camera model (cv2.fisheye)."""
    K: np.ndarray              # (3, 3) float64
    D: np.ndarray              # (4, 1) distortion coefficients
    image_size: tuple[int, int]  # (width, height)


def calibrate_fisheye_from_frames(
    frames: Sequence[np.ndarray] | None = None,
    source: int | str = 0,
    board_size: tuple[int, int] = (9, 6),
    square_size: float = 1.0,
    scale_factor: int = 3,
    max_samples: int = 20,
    min_samples: int = 8,
    frame_skip: int = 15,
    display_fn: Callable[[np.ndarray, np.ndarray | None], None] | None = None,
) -> FisheyeCamera:
    """Calibrate a fisheye camera using a checkerboard target.

    If `frames` is provided, calibrates offline from that list.
    Otherwise opens `source` (device index or path) and runs interactively
    until KeyboardInterrupt or the stream ends.

    `display_fn(raw_frame, undistorted_or_None)` is called each iteration
    for notebook live-preview.
    """
    objp = np.zeros((1, board_size[0] * board_size[1], 3), np.float64)
    objp[0, :, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2) * square_size

    obj_points: list[np.ndarray] = []
    img_points: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None

    best_K: np.ndarray | None = None
    best_D: np.ndarray | None = None
    best_rms: float = float("inf")

    sample_buf: deque[tuple[np.ndarray, np.ndarray]] = deque(maxlen=max_samples)

    sb_flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_ACCURACY
    calib_flags = (
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        | cv2.fisheye.CALIB_CHECK_COND
        | cv2.fisheye.CALIB_FIX_SKEW
    )
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

    def _detect(frame: np.ndarray) -> np.ndarray | None:
        """Returns corners as (1, N, 2) float64 for cv2.fisheye.calibrate, or None."""
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (w // scale_factor, h // scale_factor))
        gray_s = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        found, corners_s = cv2.findChessboardCornersSB(gray_s, board_size, sb_flags)
        if not found:
            try:
                eq = cv2.equalizeHist(gray_s)
            except Exception:
                eq = gray_s
            th = cv2.adaptiveThreshold(
                gray_s, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            for attempt in [gray_s, eq, th, 255 - th]:
                found, corners_s = cv2.findChessboardCorners(
                    attempt, board_size,
                    cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
                )
                if found:
                    break
        if not found or corners_s is None:
            return None
        corners_full = corners_s * scale_factor
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        refined = cv2.cornerSubPix(gray_full, corners_full.astype(np.float32), (5, 5), (-1, -1), term)
        return refined.reshape(1, -1, 2).astype(np.float64)

    def _recalibrate() -> None:
        nonlocal best_K, best_D, best_rms
        if len(sample_buf) < min_samples:
            return
        objs = [objp] * len(sample_buf)
        imgs = [c for _, c in sample_buf]
        sz = sample_buf[0][0]
        try:
            K_tmp = np.eye(3, dtype=np.float64)
            D_tmp = np.zeros((4, 1), dtype=np.float64)
            rvecs_tmp = [np.zeros((1, 1, 3)) for _ in objs]
            tvecs_tmp = [np.zeros((1, 1, 3)) for _ in objs]
            rms, K_tmp, D_tmp, _, _ = cv2.fisheye.calibrate(
                objs, imgs, sz, K_tmp, D_tmp, rvecs_tmp, tvecs_tmp, calib_flags, term,
            )
            if rms < best_rms:
                best_rms = rms
                best_K = K_tmp.copy()
                best_D = D_tmp.copy()
        except cv2.error:
            pass

    def _undistort_preview(frame: np.ndarray) -> np.ndarray | None:
        if best_K is None or image_size is None:
            return None
        newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            best_K, best_D, image_size, np.eye(3), balance=0.0)
        return cv2.fisheye.undistortImage(frame, best_K, best_D, Knew=newK)

    if frames is not None:
        for idx, frame in enumerate(frames):
            if image_size is None:
                image_size = (frame.shape[1], frame.shape[0])
            if idx % frame_skip != 0:
                continue
            corners = _detect(frame)
            annotated = frame.copy()
            if corners is not None:
                cv2.drawChessboardCorners(
                    annotated, board_size,
                    corners.reshape(-1, 1, 2).astype(np.float32), True,
                )
                sample_buf.append((image_size, corners))
                _recalibrate()
            if display_fn:
                display_fn(annotated, _undistort_preview(frame))
    else:
        # Live: open camera
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera source: {source!r}")
        frame_idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if image_size is None:
                    image_size = (frame.shape[1], frame.shape[0])
                corners = _detect(frame)
                annotated = frame.copy()
                if corners is not None:
                    cv2.drawChessboardCorners(
                        annotated, board_size,
                        corners.reshape(-1, 1, 2).astype(np.float32), True,
                    )
                    if frame_idx % frame_skip == 0:
                        sample_buf.append((image_size, corners))
                        _recalibrate()
                frame_idx += 1
                if display_fn:
                    display_fn(annotated, _undistort_preview(frame))
        except KeyboardInterrupt:
            pass
        finally:
            cap.release()

    if best_K is None or image_size is None:
        raise RuntimeError(
            "Fisheye calibration failed: not enough valid checkerboard detections."
        )
    return FisheyeCamera(K=best_K, D=best_D, image_size=image_size)


def undistort_fisheye_image(
    img: np.ndarray,
    cam: FisheyeCamera,
    balance: float = 0.0,
) -> np.ndarray:
    """Undistort a fisheye image using the Kannala–Brandt model."""
    newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        cam.K, cam.D, cam.image_size, np.eye(3), balance=balance
    )
    return cv2.fisheye.undistortImage(img, cam.K, cam.D, Knew=newK)


def run_fisheye_calibration_live(
    source: int | str = 0,
    board_size: tuple[int, int] = (9, 6),
    display_fn: Callable[[np.ndarray, np.ndarray | None], None] | None = None,
) -> FisheyeCamera:
    """Open the camera and calibrate interactively. Stop with KeyboardInterrupt."""
    return calibrate_fisheye_from_frames(
        frames=None,
        source=source,
        board_size=board_size,
        display_fn=display_fn,
    )


# ---------------------------------------------------------------------------
# Calibration save / load
# ---------------------------------------------------------------------------

def save_calibration(path: Path | str, cam: PinholeCamera | FisheyeCamera) -> None:
    """Save calibration to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(cam, FisheyeCamera):
        data = {
            "model": "fisheye",
            "K": cam.K.tolist(),
            "D": cam.D.tolist(),
            "image_size": list(cam.image_size),
        }
    else:
        data = {
            "model": "pinhole",
            "K": cam.K.tolist(),
            "dist": cam.dist_coeffs.tolist(),
        }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_calibration(path: Path | str) -> PinholeCamera | FisheyeCamera:
    """Load calibration from JSON saved by save_calibration."""
    with open(path) as f:
        data = json.load(f)
    model = data.get("model", "pinhole")
    K = np.array(data["K"], dtype=np.float64)
    if model == "fisheye":
        D = np.array(data["D"], dtype=np.float64)
        image_size = tuple(data["image_size"])
        return FisheyeCamera(K=K, D=D, image_size=image_size)
    dist = np.array(data.get("dist", np.zeros(5)), dtype=np.float64)
    return PinholeCamera.from_matrix(K, dist)
