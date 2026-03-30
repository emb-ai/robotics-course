"""
Interactive ArUco demo: undistorted fisheye stream, capture poses, trimesh visualization.

Run from seminar directory:
  conda run -n ai_in_robotics python camera_aruco_demo.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import trimesh

from camera import (
    DEFAULT_FISHEYE_CALIB_PATH,
    DEFAULT_MARKER_LENGTH_M,
    DEFAULT_TARGET_SIZE,
    camera_position_in_marker_frame,
    camera_transform_in_marker_frame,
    estimate_marker_poses,
    make_aruco_detector,
    FisheyeRGBStreamer,
)


def make_camera_frustum(transform: np.ndarray, scale: float = 0.03) -> trimesh.path.Path3D:
    origin = np.array([0.0, 0.0, 0.0, 1.0])
    corners = np.array(
        [
            [-0.5, -0.5, 1.0, 1.0],
            [0.5, -0.5, 1.0, 1.0],
            [0.5, 0.5, 1.0, 1.0],
            [-0.5, 0.5, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    corners[:, :3] *= scale
    o = (transform @ origin)[:3]
    c = (transform @ corners.T).T[:, :3]
    segments = np.array(
        [
            [o, c[0]],
            [o, c[1]],
            [o, c[2]],
            [o, c[3]],
            [c[0], c[1]],
            [c[1], c[2]],
            [c[2], c[3]],
            [c[3], c[0]],
        ],
        dtype=np.float64,
    )
    return trimesh.load_path(segments)


def run_fisheye_demo(
    *,
    source: int,
    calib_path: Path,
    target_size: tuple[int, int],
    marker_length_m: float,
    capture_count: int,
    screenshot_dir: Path,
) -> None:
    detector = make_aruco_detector()
    dist_coeffs = np.zeros((5, 1), dtype=np.float64)

    with FisheyeRGBStreamer(
        source,
        calib_path=calib_path,
        undistort=True,
        target_size=target_size,
    ) as stream:
        ok0, _ = stream.read()
        if not ok0:
            raise RuntimeError(f"Cannot read initial frame from camera {source}")
        K = stream.undistorted_camera_matrix_for_size
        if K is None:
            raise RuntimeError("Undistorted camera matrix not available.")

        screenshot_dir.mkdir(parents=True, exist_ok=True)
        captures: list[dict] = []

        try:
            while len(captures) < capture_count:
                ok, frame = stream.read()
                if not ok or frame is None:
                    break

                poses = estimate_marker_poses(
                    frame, K, dist_coeffs, marker_length_m, aruco_detector=detector
                )
                solved_pose = None
                if poses:
                    mid, rvec, tvec, corners_raw = poses[0]
                    cv2.aruco.drawDetectedMarkers(frame, [corners_raw], np.array([[mid]]))
                    cv2.drawFrameAxes(frame, K, dist_coeffs, rvec, tvec, 0.03)
                    cam_pos = camera_position_in_marker_frame(rvec, tvec)
                    cam_tf = camera_transform_in_marker_frame(rvec, tvec)
                    solved_pose = (mid, cam_pos, cam_tf)
                    cv2.putText(
                        frame,
                        f"cam_pos[m]: {cam_pos[0]:+.3f}, {cam_pos[1]:+.3f}, {cam_pos[2]:+.3f}",
                        (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

                cv2.putText(
                    frame,
                    f"Press 'c' to capture ({len(captures)}/{capture_count}), 'q' to quit",
                    (10, target_size[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("Undistorted webcam + ArUco", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("c"):
                    if solved_pose is None:
                        print("No valid marker pose in current frame; capture skipped.")
                        continue
                    marker_id, cam_pos, cam_tf = solved_pose
                    idx = len(captures)
                    path = screenshot_dir / f"capture_{idx}.png"
                    cv2.imwrite(str(path), frame)
                    captures.append({"position": cam_pos, "transform": cam_tf})
                    print(
                        f"[capture {idx + 1}/{capture_count}] marker={marker_id} "
                        f"cam_pos={cam_pos} saved={path}"
                    )
        finally:
            cv2.destroyAllWindows()

    if len(captures) < capture_count:
        raise RuntimeError(
            f"Captured only {len(captures)} valid pose(s). "
            f"Keep marker visible and rerun to get {capture_count}."
        )

    points = np.array([c["position"] for c in captures], dtype=np.float64)
    print("Captured camera positions (marker frame, meters):")
    for i, p in enumerate(points):
        print(f"  {i}: [{p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f}]")

    scene = trimesh.Scene()
    scene.add_geometry(trimesh.creation.axis(origin_size=0.01, axis_length=0.08))
    for i, p in enumerate(points):
        sphere = trimesh.creation.icosphere(radius=0.006)
        color = np.array(
            [
                [255, 60, 60, 255],
                [60, 255, 60, 255],
                [60, 120, 255, 255],
            ][i],
            dtype=np.uint8,
        )
        sphere.visual.vertex_colors = color
        sphere.apply_translation(p)
        scene.add_geometry(sphere, node_name=f"camera_{i}")
        frustum = make_camera_frustum(captures[i]["transform"], scale=0.03)
        frustum.colors = np.tile(color, (len(frustum.entities), 1))
        scene.add_geometry(frustum, node_name=f"frustum_{i}")
    path_geom = trimesh.load_path(points)
    scene.add_geometry(path_geom, node_name="camera_path")
    print("Opening trimesh scene with inferred camera locations...")
    scene.show()


def main() -> None:
    base = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="ArUco + fisheye demo with trimesh path view.")
    p.add_argument("--source", type=int, default=2, help="OpenCV camera index")
    p.add_argument(
        "--calib",
        type=str,
        default=str(DEFAULT_FISHEYE_CALIB_PATH),
        help="Fisheye calibration JSON",
    )
    p.add_argument("--marker-length", type=float, default=DEFAULT_MARKER_LENGTH_M)
    p.add_argument("--captures", type=int, default=3)
    p.add_argument(
        "--screenshots",
        type=str,
        default=str(base / "screenshots"),
        help="Directory for saved PNGs",
    )
    args = p.parse_args()
    run_fisheye_demo(
        source=args.source,
        calib_path=Path(args.calib),
        target_size=DEFAULT_TARGET_SIZE,
        marker_length_m=args.marker_length,
        capture_count=args.captures,
        screenshot_dir=Path(args.screenshots),
    )


if __name__ == "__main__":
    main()
