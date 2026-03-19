from __future__ import annotations

import json
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from lib.live import (
    CameraPoseSample,
    DepthFrame,
    ImuSample,
    RgbFrame,
    SourceMode,
    SyncBuffer,
)

PHONE_ASSETS = Path(__file__).resolve().parent.parent / "assets" / "phone_recordings"


# ---------------------------------------------------------------------------
# Legacy loaders (unchanged interface, kept for backwards compatibility)
# ---------------------------------------------------------------------------

def load_imu_csv(path: Path | str | None = None) -> dict[str, np.ndarray] | None:
    """Load pre-recorded IMU data (CSV: t, ax, ay, az, gx, gy, gz)."""
    if path is None:
        candidates = sorted(PHONE_ASSETS.glob("imu*.csv"))
        if not candidates:
            return None
        path = candidates[0]
    path = Path(path)
    if not path.exists():
        return None
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    return {"t": data[:, 0], "accel": data[:, 1:4], "gyro": data[:, 4:7]}


def load_depth_ply(path: Path | str | None = None) -> np.ndarray | None:
    """Load a point cloud from PLY via open3d or minimal ASCII parser."""
    if path is None:
        candidates = sorted(PHONE_ASSETS.glob("*.ply"))
        if not candidates:
            return None
        path = candidates[0]
    path = Path(path)
    if not path.exists():
        return None
    try:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(str(path))
        return np.asarray(pcd.points)
    except ImportError:
        pass
    points = []
    in_data = False
    with open(path) as f:
        for line in f:
            if line.startswith("element vertex"):
                pass
            elif line.strip() == "end_header":
                in_data = True
                continue
            elif in_data:
                vals = line.strip().split()
                if len(vals) >= 3:
                    points.append([float(v) for v in vals[:3]])
    return np.array(points) if points else None


def load_camera_frames(
    directory: Path | str | None = None, max_frames: int = 50,
) -> list[np.ndarray]:
    """Load pre-captured camera frames (PNG/JPG) from a directory."""
    import cv2
    if directory is None:
        directory = PHONE_ASSETS / "frames"
    directory = Path(directory)
    if not directory.exists():
        return []
    exts = {".png", ".jpg", ".jpeg"}
    paths = sorted(p for p in directory.iterdir() if p.suffix.lower() in exts)[:max_frames]
    return [img for p in paths if (img := cv2.imread(str(p))) is not None]


def get_sample_data() -> dict[str, Any]:
    return {"imu": load_imu_csv(), "depth": load_depth_ply(), "frames": load_camera_frames()}


# ---------------------------------------------------------------------------
# Depth PNG decoder (RoboGSmart / PerceptionDemo format)
#   depth stored as RGBA PNG where depth_mm = R<<8 | G
# ---------------------------------------------------------------------------

def decode_depth_png(png_path: Path) -> np.ndarray | None:
    """Return float32 depth in metres from a packed RG-depth PNG."""
    import cv2
    img = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3 and img.shape[2] >= 2:
        r = img[:, :, 2].astype(np.uint16)  # OpenCV reads BGR
        g = img[:, :, 1].astype(np.uint16)
        depth_mm = (r << 8) | g
    elif img.ndim == 2:
        depth_mm = img.astype(np.uint16)
    else:
        return None
    return (depth_mm.astype(np.float32) / 1000.0)


# ---------------------------------------------------------------------------
# PhoneReplaySource — plays back a recorded scan folder
# ---------------------------------------------------------------------------

@dataclass
class PhoneReplaySource:
    """Replay a scan folder as a stream of live.py packets.

    Supports two layouts:
    - RoboGSmart: trajectory.jsonl + rgb/*.jpg + depth/*.png
    - Legacy:     imu*.csv + frames/*.{png,jpg} + *.ply
    """

    scan_dir: Path | str = PHONE_ASSETS
    replay_fps: float = 10.0
    loop: bool = False
    sync_buffer: SyncBuffer | None = None
    mode: SourceMode = SourceMode.REPLAY

    def __post_init__(self) -> None:
        self.scan_dir = Path(self.scan_dir)
        if self.sync_buffer is None:
            self.sync_buffer = SyncBuffer()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    # --- RoboGSmart layout ---

    def _is_robogs_layout(self) -> bool:
        return (self.scan_dir / "trajectory.jsonl").exists()

    def _iter_robogs(self) -> Iterator[tuple[str, Any]]:
        import cv2

        traj_path = self.scan_dir / "trajectory.jsonl"
        with open(traj_path) as f:
            entries = [json.loads(line) for line in f if line.strip()]

        for entry in entries:
            ts = float(entry.get("timestamp", 0.0))
            idx = entry["frame_index"]

            # IMU
            imu = entry.get("imu")
            if imu:
                yield "imu", ImuSample(
                    timestamp=ts,
                    accel=np.array(imu["user_acceleration"], dtype=np.float64),
                    gyro=np.array(imu["rotation_rate"], dtype=np.float64),
                    gravity=np.array(imu["gravity"], dtype=np.float64)
                    if "gravity" in imu else None,
                )

            # RGB
            rgb_path = self.scan_dir / "rgb" / f"{idx:06d}.jpg"
            if rgb_path.exists():
                img = cv2.imread(str(rgb_path))
                pose_flat = entry.get("intrinsics", [])
                K = (
                    np.array(pose_flat, dtype=np.float64).reshape(3, 3, order="F")
                    if len(pose_flat) == 9 else np.eye(3)
                )
                if img is not None:
                    yield "rgb", RgbFrame(timestamp=ts, image=img, intrinsics=K)

            # Depth + Pose
            depth_path = self.scan_dir / "depth" / f"{idx:06d}.png"
            if depth_path.exists():
                depth_m = decode_depth_png(depth_path)
                pose_flat = entry.get("pose", [])
                T_cw = (
                    np.array(pose_flat, dtype=np.float64).reshape(4, 4, order="F")
                    if len(pose_flat) == 16 else np.eye(4)
                )
                depth_K_flat = entry.get("depth_intrinsics") or entry.get("intrinsics", [])
                K_d = (
                    np.array(depth_K_flat, dtype=np.float64).reshape(3, 3, order="F")
                    if len(depth_K_flat) == 9 else np.eye(3)
                )
                if depth_m is not None:
                    yield "depth", DepthFrame(
                        timestamp=ts, depth_m=depth_m, intrinsics=K_d, pose_T_cw=T_cw
                    )

    # --- Legacy layout ---

    def _iter_legacy(self) -> Iterator[tuple[str, Any]]:
        imu_data = load_imu_csv(self.scan_dir / "imu.csv")
        frames = load_camera_frames(self.scan_dir / "frames")
        ts = 0.0
        dt = 1.0 / self.replay_fps

        if imu_data is not None:
            n = len(imu_data["t"])
            for i in range(n):
                yield "imu", ImuSample(
                    timestamp=float(imu_data["t"][i]),
                    accel=imu_data["accel"][i],
                    gyro=imu_data["gyro"][i],
                )

        for frame in frames:
            yield "rgb", RgbFrame(timestamp=ts, image=frame, intrinsics=np.eye(3))
            ts += dt

    def _replay_loop(self) -> None:
        interval = 1.0 / self.replay_fps
        while not self._stop_event.is_set():
            it = self._iter_robogs() if self._is_robogs_layout() else self._iter_legacy()
            for kind, packet in it:
                if self._stop_event.is_set():
                    return
                assert self.sync_buffer is not None
                if kind == "imu":
                    self.sync_buffer.push_imu(packet)
                elif kind == "rgb":
                    self.sync_buffer.push_rgb(packet)
                elif kind == "depth":
                    self.sync_buffer.push_depth(packet)
                time.sleep(interval)
            if not self.loop:
                break

    def start(self) -> None:
        if self.is_running():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._replay_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()


# ---------------------------------------------------------------------------
# PhoneUSBSource — receives packets from iPhone over USB via iproxy
# ---------------------------------------------------------------------------

@dataclass
class PhoneUSBSource:
    """Receives live packets from the iOS PerceptionDemo app via USB-C + iproxy.

    Before calling start():
      1. Plug iPhone into Mac via USB-C.
      2. In a terminal: iproxy 7777 7777
      3. In the app tap "Start Listening".
    """

    port: int = 7777
    host: str = "127.0.0.1"
    sync_buffer: SyncBuffer | None = None
    mode: SourceMode = SourceMode.LIVE

    def __post_init__(self) -> None:
        if self.sync_buffer is None:
            self.sync_buffer = SyncBuffer()
        self._server: Any = None
        self._connected = False

    def start(self) -> None:
        if self.is_running():
            return
        from lib.ws_server import PerceptionUSBServer

        def _on_connect():
            self._connected = True
            print("iPhone connected via USB!")

        def _on_disconnect():
            self._connected = False
            print("iPhone disconnected.")

        self._server = PerceptionUSBServer(
            host=self.host,
            port=self.port,
            sync_buffer=self.sync_buffer,
            on_connect=_on_connect,
            on_disconnect=_on_disconnect,
        )
        self._server.start()
        print(f"Connecting to iPhone on localhost:{self.port} (via iproxy)…")

    def stop(self) -> None:
        if self._server:
            self._server.stop()
            self._server = None
        self._connected = False

    def is_running(self) -> bool:
        return self._server is not None and self._server.is_running()

    def is_connected(self) -> bool:
        return self._connected

    def wait_for_connection(self, timeout: float = 30.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._connected:
                return True
            time.sleep(0.25)
        return False

    def get_latest_imu(self) -> ImuSample | None:
        assert self.sync_buffer is not None
        return self.sync_buffer.get_latest_imu()

    def get_latest_rgb(self) -> RgbFrame | None:
        assert self.sync_buffer is not None
        return self.sync_buffer.get_latest_rgb()

    def get_latest_depth(self) -> DepthFrame | None:
        assert self.sync_buffer is not None
        return self.sync_buffer.get_latest_depth()


# ---------------------------------------------------------------------------
# PhoneWebSocketSource — connects to the Python WS server and pulls packets
# ---------------------------------------------------------------------------

@dataclass
class PhoneWebSocketSource:
    """Receives live packets from the iOS PerceptionDemo app via the local WS server.

    Starts a PerceptionWSServer internally on `port` and waits for the app to connect.
    """

    port: int = 8765
    host: str = "0.0.0.0"
    sync_buffer: SyncBuffer | None = None
    mode: SourceMode = SourceMode.LIVE

    def __post_init__(self) -> None:
        if self.sync_buffer is None:
            self.sync_buffer = SyncBuffer()
        self._server = None
        self._connected = False

    def start(self) -> None:
        if self.is_running():
            return
        from lib.ws_server import PerceptionWSServer

        def _on_connect():
            self._connected = True
            print("iPhone connected!")

        def _on_disconnect():
            self._connected = False
            print("iPhone disconnected.")

        self._server = PerceptionWSServer(
            host=self.host,
            port=self.port,
            sync_buffer=self.sync_buffer,
            on_connect=_on_connect,
            on_disconnect=_on_disconnect,
        )
        self._server.start()
        print(
            f"WebSocket server listening on ws://<your-ip>:{self.port}\n"
            f"Open PerceptionDemo on iPhone and enter this machine's IP."
        )

    def stop(self) -> None:
        if self._server:
            self._server.stop()
            self._server = None
        self._connected = False

    def is_running(self) -> bool:
        return self._server is not None and self._server.is_running()

    def is_connected(self) -> bool:
        return self._connected

    def wait_for_connection(self, timeout: float = 30.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._connected:
                return True
            time.sleep(0.25)
        return False

    # Convenience pass-throughs to buffer
    def get_latest_imu(self) -> ImuSample | None:
        assert self.sync_buffer is not None
        return self.sync_buffer.get_latest_imu()

    def get_latest_rgb(self) -> RgbFrame | None:
        assert self.sync_buffer is not None
        return self.sync_buffer.get_latest_rgb()

    def get_latest_depth(self) -> DepthFrame | None:
        assert self.sync_buffer is not None
        return self.sync_buffer.get_latest_depth()
