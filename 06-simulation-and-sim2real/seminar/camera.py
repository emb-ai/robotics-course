"""
Fisheye (UVC) and Intel RealSense RGB streamers, ArUco pose estimation,
and CLI: fisheye calibration, or side-by-side stream preview (undistorted fisheye + RealSense).

Import this module to use streamers and pose helpers without opening cameras.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import textwrap
import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_FISHEYE_CALIB_PATH = BASE_DIR / "calibration_data" / "camera_calibration.json"

# Defaults for seminar demos (override per instance)
DEFAULT_TARGET_SIZE = (640, 480)
DEFAULT_MARKER_LENGTH_M = 0.045
DEFAULT_ARUCO_DICTIONARY = cv2.aruco.DICT_6X6_250


def _debug_log(run_id: str, hypothesis_id: str, location: str, message: str, data: dict[str, Any]) -> None:
    payload = {
        "sessionId": "262d20",
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        with Path("/Users/penchekrak/PycharmProjects/ai_in_robotics/.cursor/debug-262d20.log").open("a") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception:
        pass


# --- Calibration file I/O ---


def load_fisheye_calibration(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    path = Path(path)
    with path.open() as f:
        calib = json.load(f)
    K = np.array(calib["intrinsic"], dtype=np.float64)
    D = np.array(calib["distortion"], dtype=np.float64)
    return K, D


def save_calibration(path: str | Path, intrinsic: np.ndarray, distortion: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(
            {"intrinsic": intrinsic.tolist(), "distortion": distortion.tolist()},
            f,
        )


# --- Fisheye streamer ---


class FisheyeRGBStreamer:
    """BGR frames from a UVC / OpenCV camera with optional fisheye undistort + resize."""

    def __init__(
        self,
        source: int | str = 0,
        *,
        calib_path: str | Path | None = None,
        undistort: bool = True,
        target_size: tuple[int, int] | None = DEFAULT_TARGET_SIZE,
        balance: float = 1.0,
    ) -> None:
        self._source = source
        self._calib_path = Path(calib_path) if calib_path is not None else None
        self._undistort = undistort and calib_path is not None
        self._target_size = target_size
        self._balance = balance

        # region agent log
        _debug_log(
            "pre-fix",
            "H1",
            "camera.py:FisheyeRGBStreamer.__init__:before_videocapture",
            "About to create cv2.VideoCapture",
            {"source": source, "target_size": target_size, "undistort": self._undistort},
        )
        # endregion
        self._cap = cv2.VideoCapture(source)
        # region agent log
        _debug_log(
            "pre-fix",
            "H1",
            "camera.py:FisheyeRGBStreamer.__init__:after_videocapture",
            "Created cv2.VideoCapture",
            {"source": source},
        )
        # endregion
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video source {source!r}")

        self._K: np.ndarray | None = None
        self._D: np.ndarray | None = None
        self._map1: np.ndarray | None = None
        self._map2: np.ndarray | None = None
        self._new_K: np.ndarray | None = None
        self._raw_size: tuple[int, int] | None = None

        if self._undistort:
            if self._calib_path is None or not self._calib_path.is_file():
                raise FileNotFoundError(
                    f"Fisheye calibration file not found: {self._calib_path}"
                )
            self._K, self._D = load_fisheye_calibration(self._calib_path)
            # region agent log
            _debug_log(
                "pre-fix",
                "H2",
                "camera.py:FisheyeRGBStreamer.__init__:after_load_calib",
                "Loaded fisheye calibration",
                {
                    "calib_path": str(self._calib_path),
                    "K_shape": list(self._K.shape) if self._K is not None else None,
                    "D_shape": list(self._D.shape) if self._D is not None else None,
                },
            )
            # endregion

    def _ensure_undistort_maps(self, width: int, height: int) -> None:
        if not self._undistort or self._K is None or self._D is None:
            return
        if self._map1 is not None and self._raw_size == (width, height):
            return
        self._raw_size = (width, height)
        self._new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            self._K, self._D, (width, height), np.eye(3), balance=self._balance
        )
        self._map1, self._map2 = cv2.fisheye.initUndistortRectifyMap(
            self._K, self._D, np.eye(3), self._new_K, (width, height), cv2.CV_16SC2
        )

    @property
    def undistorted_camera_matrix_for_size(self) -> np.ndarray | None:
        """Pinhole K after undistort, scaled if target_size is set (matches output frame)."""
        if self._new_K is None:
            return None
        if self._target_size is None or self._raw_size is None:
            return self._new_K.copy()
        w0, h0 = self._raw_size
        tw, th = self._target_size
        K = self._new_K.copy()
        K[0, :] *= tw / w0
        K[1, :] *= th / h0
        return K

    def read(self) -> tuple[bool, np.ndarray | None]:
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return False, None
        h, w = frame.shape[:2]
        if self._undistort:
            self._ensure_undistort_maps(w, h)
            if self._map1 is not None and self._map2 is not None:
                frame = cv2.remap(frame, self._map1, self._map2, cv2.INTER_LINEAR)
        if self._target_size is not None:
            frame = cv2.resize(frame, self._target_size)
        return True, frame

    def release(self) -> None:
        self._cap.release()

    def __enter__(self) -> FisheyeRGBStreamer:
        return self

    def __exit__(self, *args: Any) -> None:
        self.release()


# --- RealSense streamer ---


def _realsense_intrinsics_to_opencv(intr: Any) -> tuple[np.ndarray, np.ndarray]:
    """Map pyrealsense2 intrinsics to OpenCV camera_matrix and dist_coeffs (5x1 brown)."""
    K = np.array(
        [[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    coeffs = list(intr.coeffs[:5])
    while len(coeffs) < 5:
        coeffs.append(0.0)
    D = np.array(coeffs, dtype=np.float64).reshape(5, 1)
    return K, D


class RealSenseRGBStreamer:
    """BGR frames from an Intel RealSense color sensor."""

    def __init__(
        self,
        *,
        serial: str | None = None,
        width: int = 640,
        height: int = 480,
        fps: int = 6,
    ) -> None:
        try:
            import pyrealsense2 as rs  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "RealSense streaming requires pyrealsense2. "
                "Install it in the conda env ai_in_robotics."
            ) from e

        self._rs = rs
        self._pipeline = rs.pipeline()
        config = rs.config()
        if serial:
            config.enable_device(serial)
        config.enable_stream(rs.stream.color, 1920, 1080, rs.format.rgb8, 30)
        profile = self._pipeline.start(config)
        stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        self._intrinsics = stream.get_intrinsics()

    def get_camera_matrix_and_distortion(self) -> tuple[np.ndarray, np.ndarray]:
        return _realsense_intrinsics_to_opencv(self._intrinsics)

    def read(self) -> tuple[bool, np.ndarray | None]:
        frames = self._pipeline.wait_for_frames()
        color = frames.get_color_frame()
        if not color:
            return False, None
        return True, np.asanyarray(color.get_data())

    def release(self) -> None:
        self._pipeline.stop()

    def __enter__(self) -> RealSenseRGBStreamer:
        return self

    def __exit__(self, *args: Any) -> None:
        self.release()


def _probe_realsense_worker(serial: str | None, width: int, height: int, fps: int, q: mp.Queue) -> None:
    try:
        import pyrealsense2 as rs  # type: ignore[import-untyped]

        pipeline = rs.pipeline()
        config = rs.config()
        if serial:
            config.enable_device(serial)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        profile = pipeline.start(config)
        _ = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        pipeline.stop()
        q.put({"ok": True})
    except Exception as e:
        q.put({"ok": False, "error": f"{type(e).__name__}: {e}"})


def probe_realsense_safe(
    serial: str | None, width: int, height: int, fps: int
) -> tuple[bool, str | None]:
    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()
    p = ctx.Process(target=_probe_realsense_worker, args=(serial, width, height, fps, q))
    p.start()
    p.join(timeout=8.0)
    if p.is_alive():
        p.terminate()
        p.join(timeout=1.0)
        return False, "RealSense probe timeout"
    if p.exitcode != 0:
        return False, f"RealSense probe crashed (exitcode={p.exitcode})"
    try:
        msg = q.get_nowait()
    except Exception:
        return False, "RealSense probe returned no result"
    if msg.get("ok"):
        return True, None
    return False, str(msg.get("error", "RealSense probe failed"))


def _open_realsense_streamer_with_retries(
    *,
    serial: str | None,
    width: int,
    height: int,
    fps: int,
    max_attempts: int,
    retry_delay_s: float,
) -> tuple[RealSenseRGBStreamer | None, str | None]:
    """Open RealSense in the current process; retry on transient failures (e.g. macOS power state)."""
    n = max(1, max_attempts)
    delay = max(0.0, retry_delay_s)
    last_err: str | None = None
    for attempt in range(1, n + 1):
        try:
            return (
                RealSenseRGBStreamer(
                    serial=serial,
                    width=width,
                    height=height,
                    fps=fps,
                ),
                None,
            )
        except ImportError as e:
            return None, str(e)
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            if attempt < n:
                time.sleep(delay)
    return None, last_err


# --- ArUco / pose ---


def marker_object_points(marker_length_m: float) -> np.ndarray:
    h = marker_length_m / 2.0
    return np.array(
        [
            [-h, h, 0.0],
            [h, h, 0.0],
            [h, -h, 0.0],
            [-h, -h, 0.0],
        ],
        dtype=np.float32,
    )


def T_cam_marker_from_pnp(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Transform marker frame -> camera frame: P_c = R @ P_m + t."""
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return T


def relative_pose_cam2_wrt_cam1(
    rvec1: np.ndarray,
    tvec1: np.ndarray,
    rvec2: np.ndarray,
    tvec2: np.ndarray,
) -> np.ndarray:
    """
    Rigid transform mapping points from camera1 frame to camera2 frame (4x4).
    Both (rvec_i, tvec_i) come from solvePnP for the same marker in each camera.
    """
    T1 = T_cam_marker_from_pnp(rvec1, tvec1)
    T2 = T_cam_marker_from_pnp(rvec2, tvec2)
    return T2 @ np.linalg.inv(T1)


# Alias for clearer call sites
relative_pose_between_cameras = relative_pose_cam2_wrt_cam1


def camera_position_in_marker_frame(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    rmat, _ = cv2.Rodrigues(rvec)
    return (-rmat.T @ tvec).reshape(3)


def camera_transform_in_marker_frame(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    rmat, _ = cv2.Rodrigues(rvec)
    tvec = tvec.reshape(3, 1)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rmat.T
    transform[:3, 3] = (-rmat.T @ tvec).reshape(3)
    return transform


def make_aruco_detector(
    dictionary_id: int = DEFAULT_ARUCO_DICTIONARY,
    detector_params: cv2.aruco.DetectorParameters | None = None,
) -> cv2.aruco.ArucoDetector:
    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_id)
    params = detector_params or cv2.aruco.DetectorParameters()
    return cv2.aruco.ArucoDetector(aruco_dict, params)


def estimate_marker_poses(
    frame: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    marker_length_m: float,
    *,
    aruco_detector: cv2.aruco.ArucoDetector | None = None,
    solve_flags: int = cv2.SOLVEPNP_ITERATIVE,
) -> list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Detect ArUco markers and solve pose per marker.
    Returns list of (marker_id, rvec, tvec, corners_1x4x2) for drawDetectedMarkers.
    """
    detector = aruco_detector or make_aruco_detector()
    corners, ids, _ = detector.detectMarkers(frame)
    if ids is None or len(ids) == 0:
        return []

    obj_pts = marker_object_points(marker_length_m)
    out: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
    for i in range(len(ids)):
        mid = int(ids[i][0])
        raw = np.asarray(corners[i], dtype=np.float32)
        img_corners = raw.reshape((4, 2)).astype(np.float32)
        ok, rvec, tvec = cv2.solvePnP(
            obj_pts,
            img_corners,
            camera_matrix,
            dist_coeffs,
            flags=solve_flags,
        )
        if ok:
            out.append((mid, rvec, tvec, raw))
    return out


def pose_for_marker_id(
    poses: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]],
    marker_id: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    for mid, rvec, tvec, _ in poses:
        if mid == marker_id:
            return rvec, tvec
    return None


# --- Fisheye calibration (chessboard) ---


def find_chessboard_corners(
    frame: np.ndarray, pattern_size: tuple[int, int]
) -> np.ndarray | None:
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    candidates: list[np.ndarray] = [gray]
    try:
        candidates.append(cv2.equalizeHist(gray))
    except Exception:
        candidates.append(gray)
    th = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    candidates.append(th)
    candidates.append(255 - th)

    for img in candidates:
        try:
            sb_flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_ACCURACY
            found, corners = cv2.findChessboardCornersSB(img, pattern_size, flags=sb_flags)
            if found and corners is not None and len(corners) == pattern_size[0] * pattern_size[1]:
                return corners.reshape(-1, 1, 2).astype(np.float32)
        except Exception:
            continue

    for img in candidates:
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(img, pattern_size, flags)
        if not found or corners is None:
            continue
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3)
        refined = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), term)
        if refined is not None and len(refined) == pattern_size[0] * pattern_size[1]:
            return refined
    return None


def calibrate_fisheye(
    image_size: tuple[int, int],
    object_points_list: list[np.ndarray],
    image_points_list: list[np.ndarray],
) -> tuple[float, np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
    K = np.zeros((3, 3), dtype=np.float64)
    D = np.zeros((4, 1), dtype=np.float64)
    rvecs: list[np.ndarray] = []
    tvecs: list[np.ndarray] = []
    flags = (
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        | cv2.fisheye.CALIB_CHECK_COND
        | cv2.fisheye.CALIB_FIX_SKEW
    )
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
        object_points_list,
        image_points_list,
        image_size,
        K,
        D,
        rvecs,
        tvecs,
        flags,
        criteria,
    )
    return rms, K, D, rvecs, tvecs


def create_object_points(pattern_size: tuple[int, int], square_size: float) -> np.ndarray:
    cols, rows = pattern_size
    object_points = np.zeros((1, rows * cols, 3), np.float32)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    object_points[0, :, :2] = grid * square_size
    return object_points


def main_calibrate_fisheye(args: argparse.Namespace) -> None:
    current_intrinsic = np.eye(3, dtype=np.float64)
    current_distortion = np.ones((4, 1), dtype=np.float64)

    cap = cv2.VideoCapture(args.source)
    ok0, frame0 = cap.read()
    if not ok0 or frame0 is None:
        raise RuntimeError("Failed to read from video source")
    h0, w0 = frame0.shape[:2]
    image_size = (w0, h0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w0)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h0)
    cap.set(cv2.CAP_PROP_FPS, 30)

    scale_factor = 3
    num_samples = 20
    min_samples_for_calib = 8
    pattern = (args.pattern_cols, args.pattern_rows)
    object_points_template = create_object_points(pattern, args.square_size)
    image_points_list: deque[np.ndarray] = deque(maxlen=num_samples)
    frame_idx = 0
    last_rms: float | None = None

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(frame, (image_size[0] // scale_factor, image_size[1] // scale_factor))
        small_corners = find_chessboard_corners(small, pattern)
        if small_corners is not None:
            scaled_corners = (small_corners * scale_factor).astype(np.float32)
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3)
            refined = cv2.cornerSubPix(gray, scaled_corners, (5, 5), (-1, -1), term)
            corners_for_draw = refined if refined is not None else scaled_corners
            cv2.drawChessboardCorners(frame, pattern, corners_for_draw, True)

            if frame_idx % 15 == 0:
                sample = corners_for_draw.reshape(1, -1, 2).astype(np.float64)
                image_points_list.append(sample)

            if len(image_points_list) >= min_samples_for_calib:
                object_points_list = [object_points_template.copy() for _ in range(len(image_points_list))]
                rms, K, D, _, _ = calibrate_fisheye(
                    image_size, object_points_list, list(image_points_list)
                )
                if last_rms is None or float(rms) < last_rms:
                    current_intrinsic = K
                    current_distortion = D
                    last_rms = float(rms)
        else:
            cv2.putText(
                frame,
                "No chessboard detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        if last_rms is not None:
            cv2.putText(
                frame,
                f"RMS: {last_rms:.3f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            current_intrinsic,
            current_distortion,
            image_size,
            np.eye(3),
            balance=0.0,
            fov_scale=1.0,
        )
        undistorted_frame = cv2.fisheye.undistortImage(
            frame, current_intrinsic, current_distortion, Knew=newK
        )
        side_by_side = np.vstack((frame, undistorted_frame))
        cv2.imshow("Frame", side_by_side)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        frame_idx += 1

    save_calibration(args.save, current_intrinsic, current_distortion)
    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved calibration to {args.save}")


def _build_calibrate_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fisheye camera calibration (chessboard).")
    p.add_argument("--source", type=int, default=0, help="OpenCV camera index")
    default_save = BASE_DIR / "calibration_data" / "calibration_params_top_camera.json"
    p.add_argument(
        "--save",
        type=str,
        default=str(default_save),
        help="Output JSON path (intrinsic + distortion)",
    )
    p.add_argument(
        "--pattern-cols",
        type=int,
        default=9,
        help="Inner corners along one side (OpenCV: width of grid)",
    )
    p.add_argument(
        "--pattern-rows",
        type=int,
        default=6,
        help="Inner corners along other side",
    )
    p.add_argument(
        "--square-size",
        type=float,
        default=1.0,
        help="Chessboard square size in arbitrary units (consistent scale)",
    )
    return p


def _build_stream_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Side-by-side preview: fisheye (undistorted) + Intel RealSense RGB.",
        epilog=(
            "RealSense: close Intel RealSense Viewer and other apps using the device; prefer a "
            "direct USB port or powered hub. On macOS, 'failed to set power state' is a known "
            "USB/stack issue—try another port or cable."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--fisheye-source",
        type=int,
        default=0,
        help="OpenCV index for the UVC fisheye camera",
    )
    p.add_argument(
        "--calib",
        type=str,
        default=str(DEFAULT_FISHEYE_CALIB_PATH),
        help="Fisheye calibration JSON (intrinsic + distortion)",
    )
    p.add_argument(
        "--width",
        type=int,
        default=DEFAULT_TARGET_SIZE[0],
        help="Fisheye output width after undistort + resize",
    )
    p.add_argument(
        "--height",
        type=int,
        default=DEFAULT_TARGET_SIZE[1],
        help="Fisheye output height after undistort + resize",
    )
    p.add_argument("--rs-width", type=int, default=424, help="RealSense color width")
    p.add_argument("--rs-height", type=int, default=240, help="RealSense color height")
    p.add_argument("--rs-fps", type=int, default=6, help="RealSense color FPS")
    p.add_argument("--realsense-serial", type=str, default=None, help="Optional device serial")
    p.add_argument(
        "--rs-init-retries",
        type=int,
        default=3,
        metavar="N",
        help="In-process RealSense pipeline start attempts (default: 3)",
    )
    p.add_argument(
        "--rs-retry-delay",
        type=float,
        default=0.75,
        metavar="SEC",
        help="Seconds to wait between RealSense start retries (default: 0.75)",
    )
    p.add_argument(
        "--rs-probe-subprocess",
        action="store_true",
        help=(
            "Debug: run pipeline.start in a spawned subprocess before opening in this process "
            "(default off; can fail on macOS while in-process init works)"
        ),
    )
    return p


def _annotate_preview(img: np.ndarray, label: str) -> None:
    cv2.putText(
        img,
        label,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def _error_panel(width: int, height: int, title: str, message: str) -> np.ndarray:
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(
        panel,
        title,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    max_chars = max(24, width // 9)
    lines = textwrap.wrap(message, width=max_chars) or [message]
    y = 55
    for ln in lines:
        if y > height - 14:
            break
        cv2.putText(
            panel,
            ln,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        y += 18
    return panel


def main_stream_preview(args: argparse.Namespace) -> None:
    # region agent log
    _debug_log(
        "pre-fix",
        "H4",
        "camera.py:main_stream_preview:entry",
        "Entered stream preview",
        {
            "fisheye_source": args.fisheye_source,
            "calib": args.calib,
            "width": args.width,
            "height": args.height,
            "rs_width": args.rs_width,
            "rs_height": args.rs_height,
            "rs_fps": args.rs_fps,
            "opencv_version": cv2.__version__,
            "numpy_version": np.__version__,
        },
    )
    # endregion
    target_size = (args.width, args.height)
    calib_path = Path(args.calib)
    if not calib_path.is_file():
        raise FileNotFoundError(f"Fisheye calibration not found: {calib_path}")

    fisheye = FisheyeRGBStreamer(
        args.fisheye_source,
        calib_path=calib_path,
        undistort=True,
        target_size=target_size,
    )
    # region agent log
    _debug_log(
        "pre-fix",
        "H1",
        "camera.py:main_stream_preview:after_fisheye_init",
        "Fisheye streamer initialized",
        {"fisheye_source": args.fisheye_source},
    )
    # endregion

    rs: RealSenseRGBStreamer | None = None
    rs_error: str | None = None
    # region agent log
    _debug_log(
        "pre-fix",
        "H3",
        "camera.py:main_stream_preview:before_rs_init",
        "About to initialize RealSense streamer",
        {
            "serial": args.realsense_serial,
            "width": args.rs_width,
            "height": args.rs_height,
            "fps": args.rs_fps,
            "rs_probe_subprocess": args.rs_probe_subprocess,
        },
    )
    # endregion
    if args.rs_probe_subprocess:
        rs_ok, rs_probe_error = probe_realsense_safe(
            serial=args.realsense_serial,
            width=args.rs_width,
            height=args.rs_height,
            fps=args.rs_fps,
        )
        if not rs_ok:
            rs_error = rs_probe_error or "RealSense probe failed"
        else:
            rs, rs_error = _open_realsense_streamer_with_retries(
                serial=args.realsense_serial,
                width=args.rs_width,
                height=args.rs_height,
                fps=args.rs_fps,
                max_attempts=args.rs_init_retries,
                retry_delay_s=args.rs_retry_delay,
            )
    else:
        rs, rs_error = _open_realsense_streamer_with_retries(
            serial=args.realsense_serial,
            width=args.rs_width,
            height=args.rs_height,
            fps=args.rs_fps,
            max_attempts=args.rs_init_retries,
            retry_delay_s=args.rs_retry_delay,
        )
    # region agent log
    _debug_log(
        "pre-fix",
        "H3",
        "camera.py:main_stream_preview:after_rs_init",
        "RealSense init finished",
        {"serial": args.realsense_serial, "ok": rs_error is None},
    )
    # endregion

    blank_f = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    blank_r = np.zeros((args.rs_height, args.rs_width, 3), dtype=np.uint8)
    _annotate_preview(blank_f, "Fisheye: no frame")
    _annotate_preview(blank_r, "RealSense: no frame")

    if rs_error:
        print(f"RealSense unavailable ({rs_error}); right panel shows error. Fisheye still streams.")
    else:
        print("Stream preview: undistorted fisheye (left) + RealSense RGB (right). Press 'q' to quit.")

    try:
        while True:
            ok_f, frame_f = fisheye.read()
            if rs is not None:
                ok_r, frame_r = rs.read()
            else:
                ok_r, frame_r = False, None

            left = frame_f if ok_f and frame_f is not None else blank_f.copy()
            if rs_error:
                right = _error_panel(args.rs_width, args.rs_height, "RealSense", rs_error)
            else:
                right = frame_r if ok_r and frame_r is not None else blank_r.copy()

            _annotate_preview(left, "Fisheye (undistorted)")
            if not rs_error:
                _annotate_preview(right, "RealSense")

            # Same height for horizontal stack
            h = max(left.shape[0], right.shape[0])
            if left.shape[0] != h:
                scale = h / left.shape[0]
                left = cv2.resize(
                    left,
                    (int(left.shape[1] * scale + 0.5), h),
                    interpolation=cv2.INTER_LINEAR,
                )
            if right.shape[0] != h:
                scale = h / right.shape[0]
                right = cv2.resize(
                    right,
                    (int(right.shape[1] * scale + 0.5), h),
                    interpolation=cv2.INTER_LINEAR,
                )

            combined = np.hstack((left, right))
            cv2.imshow("Fisheye (undistorted) | RealSense", combined)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        fisheye.release()
        if rs is not None:
            rs.release()
        cv2.destroyAllWindows()


def _split_cli_argv(argv: list[str]) -> tuple[str, list[str]]:
    """Route: stream | calibrate (default if no known subcommand)."""
    if not argv:
        return "calibrate", []
    first = argv[0]
    if first == "stream":
        return "stream", argv[1:]
    if first == "calibrate":
        return "calibrate", argv[1:]
    if first in ("-h", "--help"):
        return "help", argv
    return "calibrate", argv


def _print_main_help() -> None:
    print(
        """Usage:
  python camera.py [calibrate] [OPTIONS]   Chessboard fisheye calibration (default)
  python camera.py stream [OPTIONS]        Undistorted fisheye + RealSense side by side

Examples:
  python camera.py --source 0 --save out.json
  python camera.py calibrate --source 0
  python camera.py stream --fisheye-source 2

For full options: python camera.py calibrate -h   or   python camera.py stream -h
"""
    )


def main_cli() -> None:
    argv = sys.argv[1:]
    mode, rest = _split_cli_argv(argv)
    if mode == "help":
        _print_main_help()
        return

    if mode == "stream":
        main_stream_preview(_build_stream_parser().parse_args(rest))
        return

    main_calibrate_fisheye(_build_calibrate_parser().parse_args(rest))


if __name__ == "__main__":
    main_cli()
