from __future__ import annotations

import asyncio
import json
import socket
import struct
import threading
from typing import Callable

import cv2
import numpy as np

from lib.live import (
    DepthFrame,
    ImuSample,
    RgbFrame,
    SourceMode,
    SyncBuffer,
)

# Wire protocol type bytes (must match iOS PacketEncoder.swift)
_TYPE_IMU = 0x01
_TYPE_RGB = 0x02
_TYPE_DEPTH = 0x03
_TYPE_CONTROL = 0x10


# ---------------------------------------------------------------------------
# Shared packet decoders
# ---------------------------------------------------------------------------

def _decode_imu(payload: bytes) -> ImuSample | None:
    try:
        data = json.loads(payload.decode())
        return ImuSample(
            timestamp=float(data["t"]),
            accel=np.array(data["accel"], dtype=np.float64),
            gyro=np.array(data["gyro"], dtype=np.float64),
            gravity=np.array(data["gravity"], dtype=np.float64) if "gravity" in data else None,
        )
    except Exception:
        return None


def _decode_rgb(payload: bytes) -> RgbFrame | None:
    try:
        ts = struct.unpack_from("<f", payload, 0)[0]
        img = cv2.imdecode(np.frombuffer(payload[4:], np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None
        return RgbFrame(timestamp=float(ts), image=img, intrinsics=np.eye(3, dtype=np.float64))
    except Exception:
        return None


def _decode_depth(payload: bytes) -> DepthFrame | None:
    try:
        offset = 0
        ts = struct.unpack_from("<f", payload, offset)[0]; offset += 4
        pose_flat = struct.unpack_from("<16f", payload, offset); offset += 64
        k_flat = struct.unpack_from("<9f", payload, offset); offset += 36
        w, h = struct.unpack_from("<HH", payload, offset); offset += 4
        depth_u16 = np.frombuffer(payload, dtype="<u2", offset=offset, count=w * h)
        depth_m = depth_u16.reshape(h, w).astype(np.float32) / 1000.0
        T_cw = np.array(pose_flat, dtype=np.float64).reshape(4, 4, order="F")
        K = np.array(k_flat, dtype=np.float64).reshape(3, 3, order="F")
        return DepthFrame(timestamp=float(ts), depth_m=depth_m, intrinsics=K, pose_T_cw=T_cw)
    except Exception:
        return None


def _dispatch_packet(data: bytes, sync_buffer: SyncBuffer) -> None:
    if len(data) < 1:
        return
    type_byte, payload = data[0], data[1:]
    if type_byte == _TYPE_IMU:
        sample = _decode_imu(payload)
        if sample:
            sync_buffer.push_imu(sample)
    elif type_byte == _TYPE_RGB:
        frame = _decode_rgb(payload)
        if frame:
            sync_buffer.push_rgb(frame)
    elif type_byte == _TYPE_DEPTH:
        frame = _decode_depth(payload)
        if frame:
            sync_buffer.push_depth(frame)


# ---------------------------------------------------------------------------
# PerceptionUSBServer — TCP client connecting to iPhone via iproxy
# ---------------------------------------------------------------------------

class PerceptionUSBServer:
    """TCP client that receives length-framed packets from the iOS app via USB/iproxy.

    Packet framing: [uint32 big-endian length][packet bytes]
    Packet format: same type byte + payload as PerceptionWSServer.

    Usage:
        # On Mac terminal (once, after plugging in iPhone):
        #   iproxy 7777 7777
        server = PerceptionUSBServer(port=7777)
        server.start()
    """

    def __init__(
        self,
        port: int = 7777,
        host: str = "127.0.0.1",
        sync_buffer: SyncBuffer | None = None,
        on_connect: Callable[[], None] | None = None,
        on_disconnect: Callable[[], None] | None = None,
        retry_interval: float = 1.0,
    ):
        self.port = port
        self.host = host
        self.sync_buffer = sync_buffer or SyncBuffer()
        self.on_connect = on_connect
        self.on_disconnect = on_disconnect
        self.retry_interval = retry_interval

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._connected = False
        self._sock: socket.socket | None = None

    def _recv_exactly(self, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            if self._stop_event.is_set():
                raise ConnectionError("stopped")
            chunk = self._sock.recv(n - len(buf))  # type: ignore[union-attr]
            if not chunk:
                raise ConnectionError("connection closed")
            buf += chunk
        return buf

    def _reader_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._sock = socket.create_connection((self.host, self.port), timeout=2.0)
                self._sock.settimeout(None)
                self._connected = True
                if self.on_connect:
                    self.on_connect()
                while not self._stop_event.is_set():
                    header = self._recv_exactly(4)
                    length = struct.unpack(">I", header)[0]
                    data = self._recv_exactly(length)
                    _dispatch_packet(data, self.sync_buffer)
            except Exception:
                pass
            finally:
                self._connected = False
                if self._sock:
                    try:
                        self._sock.close()
                    except Exception:
                        pass
                    self._sock = None
                if not self._stop_event.is_set():
                    if self.on_disconnect:
                        self.on_disconnect()
                    self._stop_event.wait(self.retry_interval)

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._reader_loop, daemon=True, name="USBServer")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=3.0)

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def is_connected(self) -> bool:
        return self._connected

    @property
    def mode(self) -> SourceMode:
        return SourceMode.LIVE


# ---------------------------------------------------------------------------
# PerceptionWSServer — WebSocket server (WiFi, kept for reference)
# ---------------------------------------------------------------------------

class PerceptionWSServer:
    """Asyncio WebSocket server that receives packets from the iOS app (WiFi).

    Binary message format: [type_byte (1)] [payload ...]
      0x01  IMU JSON  {"t":float,"accel":[3],"gyro":[3],"gravity":[3]}
      0x02  RGB       float32_LE(ts) + JPEG_bytes
      0x03  Depth     float32_LE(ts) + float32_LE*16(pose) +
                      float32_LE*9(K) + uint16_LE(w) + uint16_LE(h) +
                      uint16_LE*W*H(depth_mm)
      0x10  Control   JSON {"cmd":"hello"|"start"|"stop"|"heartbeat"}
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        sync_buffer: SyncBuffer | None = None,
        on_connect: Callable[[], None] | None = None,
        on_disconnect: Callable[[], None] | None = None,
    ):
        self.host = host
        self.port = port
        self.sync_buffer = sync_buffer or SyncBuffer()
        self.on_connect = on_connect
        self.on_disconnect = on_disconnect

        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._server = None
        self._stop_event: asyncio.Event | None = None
        self._connected = False

    async def _handler(self, websocket) -> None:
        self._connected = True
        if self.on_connect:
            self.on_connect()
        try:
            async for message in websocket:
                if isinstance(message, str):
                    message = message.encode()
                if len(message) < 1:
                    continue
                type_byte = message[0]
                payload = message[1:]
                if type_byte == _TYPE_CONTROL:
                    try:
                        cmd = json.loads(payload.decode())
                        if cmd.get("cmd") == "hello":
                            await websocket.send(
                                json.dumps({"cmd": "hello_ack", "server": "perception-demo"})
                            )
                    except Exception:
                        pass
                else:
                    _dispatch_packet(message, self.sync_buffer)
        finally:
            self._connected = False
            if self.on_disconnect:
                self.on_disconnect()

    async def _run(self) -> None:
        from websockets.asyncio.server import serve

        self._stop_event = asyncio.Event()
        async with serve(self._handler, self.host, self.port) as server:
            self._server = server
            await self._stop_event.wait()

    def _thread_target(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._run())
        self._loop.close()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._thread_target, daemon=True, name="WSServer")
        self._thread.start()

    def stop(self) -> None:
        if self._loop and self._stop_event:
            self._loop.call_soon_threadsafe(self._stop_event.set)
        if self._thread:
            self._thread.join(timeout=3.0)

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def is_connected(self) -> bool:
        return self._connected

    @property
    def mode(self) -> SourceMode:
        return SourceMode.LIVE
