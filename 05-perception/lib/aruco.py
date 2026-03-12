from __future__ import annotations

from typing import Callable

import cv2
import numpy as np


def detect_aruco(
    img: np.ndarray,
    K: np.ndarray | None = None,
    dist: np.ndarray | None = None,
    marker_length: float = 0.05,
    dictionary_id: int = cv2.aruco.DICT_6X6_250,
) -> tuple[list[np.ndarray], np.ndarray | None, list[tuple[np.ndarray, np.ndarray]] | None]:
    """Detect ArUco markers and optionally estimate poses.

    Returns (corners_list, ids, poses) where poses is a list of
    (rvec, tvec) tuples if K is provided.
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_id)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    corners, ids, _ = detector.detectMarkers(img)

    poses = None
    if K is not None and ids is not None and len(ids) > 0:
        poses = []
        obj_pts = np.array([
            [-marker_length / 2, marker_length / 2, 0],
            [marker_length / 2, marker_length / 2, 0],
            [marker_length / 2, -marker_length / 2, 0],
            [-marker_length / 2, -marker_length / 2, 0],
        ], dtype=np.float64)
        if dist is None:
            dist = np.zeros(5)
        for c in corners:
            _, rvec, tvec = cv2.solvePnP(obj_pts, c.reshape(-1, 2), K, dist)
            poses.append((rvec, tvec))

    return list(corners), ids, poses


def estimate_aruco_pose(
    corners: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray | None = None,
    marker_length: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate pose for a single marker's corners. Returns (rvec, tvec)."""
    obj_pts = np.array([
        [-marker_length / 2, marker_length / 2, 0],
        [marker_length / 2, marker_length / 2, 0],
        [marker_length / 2, -marker_length / 2, 0],
        [-marker_length / 2, -marker_length / 2, 0],
    ], dtype=np.float64)
    if dist is None:
        dist = np.zeros(5)
    _, rvec, tvec = cv2.solvePnP(obj_pts, corners.reshape(-1, 2), K, dist)
    return rvec, tvec


def detect_and_overlay_live(
    frame: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray | None = None,
    marker_length: float = 0.05,
    dictionary_id: int = cv2.aruco.DICT_6X6_250,
    axis_length: float | None = None,
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]] | None]:
    """Detect ArUco markers on `frame`, draw IDs + pose axes in-place.

    Returns the annotated frame (same array) and the list of (rvec, tvec) poses.
    """
    dist_safe = np.zeros(5) if dist is None else dist
    ax_len = axis_length if axis_length is not None else marker_length * 0.5

    corners, ids, poses = detect_aruco(frame, K, dist_safe, marker_length, dictionary_id)

    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        if poses:
            for rvec, tvec in poses:
                cv2.drawFrameAxes(frame, K, dist_safe, rvec, tvec, ax_len)

    return frame, poses


def stream_aruco_live(
    source: int | str = 0,
    K: np.ndarray | None = None,
    dist: np.ndarray | None = None,
    marker_length: float = 0.05,
    dictionary_id: int = cv2.aruco.DICT_6X6_250,
    axis_length: float | None = None,
    display_fn: Callable[[np.ndarray], None] | None = None,
) -> None:
    """Open camera, detect ArUco markers each frame, call display_fn with the annotated BGR frame.

    Stops on KeyboardInterrupt. If display_fn is None, shows via cv2.imshow (press q to quit).
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera source: {source!r}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            annotated = frame.copy()
            detect_and_overlay_live(
                annotated, K if K is not None else np.eye(3),
                dist, marker_length, dictionary_id, axis_length,
            )
            if display_fn is not None:
                display_fn(annotated)
            else:
                cv2.imshow("ArUco Live", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if display_fn is None:
            cv2.destroyAllWindows()
