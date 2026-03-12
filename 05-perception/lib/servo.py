from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass

import numpy as np

from lib.encoder import EncoderModel
from lib.live import ServoSample, SourceMode, SyncBuffer

# STS3215 / SMS_STS: 4096 ticks per revolution
_TICKS_PER_REV = 4096
_TICKS_TO_RAD = 2 * math.pi / _TICKS_PER_REV

# Use vendored Feetech-tuna SCServo SDK (same as tuna tool)
try:
    from lib.feetech_sdk.port_handler import PortHandler
    from lib.feetech_sdk.sms_sts import sms_sts
    from lib.feetech_sdk.scservo_def import COMM_SUCCESS
    _HAS_FEETECH_SDK = True
except ImportError:
    _HAS_FEETECH_SDK = False


def _resolve_port(port: str) -> str | None:
    import glob
    if not port or (("*" not in port and "?" not in port)):
        return port
    candidates = sorted(glob.glob(port))
    if not candidates:
        for fallback in ("/dev/cu.usbserial-*", "/dev/tty.usbserial-*", "/dev/ttyUSB*", "/dev/ttyACM*"):
            if fallback != port:
                candidates = sorted(glob.glob(fallback))
                if candidates:
                    break
    return candidates[0] if candidates else None


# ---------------------------------------------------------------------------
# FeetechServoSource — uses Feetech-tuna SCServo SDK (same as tuna CLI)
# ---------------------------------------------------------------------------

@dataclass
class FeetechServoSource:
    """Live reader for Feetech STS/SMS servos via the same SDK as Feetech-tuna.

    Uses vendored SCServo_Python from Feetech-tuna (PortHandler + sms_sts).
    Register 56 = Present Position, 58 = Present Speed, 60 = Present Load.
    """

    port: str = "/dev/tty.usbserial-*"
    baud: int = 1_000_000
    servo_id: int = 1
    poll_hz: float = 100.0
    read_speed_load: bool = True
    sync_buffer: SyncBuffer | None = None

    mode: SourceMode = SourceMode.LIVE

    def __post_init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._port_handler: PortHandler | None = None
        self._packet_handler: sms_sts | None = None
        self._resolved_port: str | None = None
        if self.sync_buffer is None:
            self.sync_buffer = SyncBuffer()

    def _open_port(self) -> None:
        if not _HAS_FEETECH_SDK:
            raise RuntimeError(
                "Feetech SDK not available. Install 05-perception lib with feetech_sdk."
            )
        resolved = _resolve_port(self.port)
        if not resolved:
            raise RuntimeError(
                f"No serial port matching '{self.port}'. "
                f"Try listing: /dev/tty.usb* or /dev/cu.usb*"
            )
        self._resolved_port = resolved
        self._port_handler = PortHandler(resolved)
        if not self._port_handler.openPort():
            raise RuntimeError(f"Failed to open port {resolved}")
        if not self._port_handler.setBaudRate(self.baud):
            self._port_handler.closePort()
            raise RuntimeError(f"Failed to set baudrate {self.baud}")
        self._packet_handler = sms_sts(self._port_handler)

    def _poll_loop(self) -> None:
        interval = 1.0 / self.poll_hz
        t0 = time.monotonic()
        while not self._stop_event.is_set():
            tick_start = time.monotonic()
            ph = self._packet_handler
            if ph is None:
                break
            if self.read_speed_load:
                pos, speed, comm_result, _ = ph.ReadPosSpeed(self.servo_id)
                speed_val = speed if comm_result == COMM_SUCCESS else None
            else:
                pos, comm_result, _ = ph.ReadPos(self.servo_id)
                speed_val = None
            load_val = None
            if comm_result == COMM_SUCCESS:
                pos_unsigned = pos if pos >= 0 else (pos + 0x10000) & 0xFFFF
                pos_rad = pos_unsigned * _TICKS_TO_RAD
                if self.read_speed_load:
                    load_word, lr, _ = ph.read2ByteTxRx(self.servo_id, 60)
                    if lr == COMM_SUCCESS:
                        load_val = ph.scs_tohost(load_word, 15)
                sample = ServoSample(
                    timestamp=time.monotonic() - t0,
                    position_ticks=pos_unsigned,
                    position_rad=pos_rad,
                    speed=speed_val,
                    load=load_val,
                )
                assert self.sync_buffer is not None
                self.sync_buffer.push_servo(sample)
            elapsed = time.monotonic() - tick_start
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def start(self) -> None:
        self._open_port()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._port_handler is not None:
            self._port_handler.closePort()
            self._port_handler = None
        self._packet_handler = None

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def get_recent(self, n: int = 50) -> list[ServoSample]:
        assert self.sync_buffer is not None
        return self.sync_buffer.get_recent_servo(n)

    @staticmethod
    def scan_bus(port: str, baud: int = 1_000_000) -> list[int]:
        """Return list of servo IDs on the bus (uses same SDK as tuna list)."""
        if not _HAS_FEETECH_SDK:
            return []
        resolved = _resolve_port(port)
        if not resolved:
            return []
        found = []
        try:
            ph = PortHandler(resolved)
            if not ph.openPort() or not ph.setBaudRate(baud):
                return []
            pkt = sms_sts(ph)
            for sid in range(1, 254):
                _, result, _ = pkt.ping(sid)
                if result == COMM_SUCCESS:
                    found.append(sid)
            ph.closePort()
        except Exception:
            pass
        return found


# ---------------------------------------------------------------------------
# SimulatedServoSource — generates synthetic encoder samples
# ---------------------------------------------------------------------------

@dataclass
class SimulatedServoSource:
    """Synthetic servo using EncoderModel(ticks_per_rev=4096)."""

    ticks_per_rev: int = 4096
    omega: float = 2.0
    amplitude: float = math.pi
    poll_hz: float = 100.0
    sync_buffer: SyncBuffer | None = None

    mode: SourceMode = SourceMode.SIMULATED

    def __post_init__(self) -> None:
        self._encoder = EncoderModel(ticks_per_rev=self.ticks_per_rev)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        if self.sync_buffer is None:
            self.sync_buffer = SyncBuffer()

    def _poll_loop(self) -> None:
        interval = 1.0 / self.poll_hz
        t0 = time.monotonic()
        while not self._stop_event.is_set():
            tick_start = time.monotonic()
            t = time.monotonic() - t0
            true_angle = self.amplitude * math.sin(self.omega * t)
            quantized_angle = float(self._encoder.quantize(np.array([true_angle]))[0])
            pos_ticks = int(round(quantized_angle / _TICKS_TO_RAD)) % self.ticks_per_rev
            sample = ServoSample(
                timestamp=t,
                position_ticks=pos_ticks,
                position_rad=quantized_angle,
                speed=None,
                load=None,
            )
            assert self.sync_buffer is not None
            self.sync_buffer.push_servo(sample)
            elapsed = time.monotonic() - tick_start
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def get_recent(self, n: int = 50) -> list[ServoSample]:
        assert self.sync_buffer is not None
        return self.sync_buffer.get_recent_servo(n)


# ---------------------------------------------------------------------------
# Auto-select: try real servo via Feetech-tuna SDK, else simulated
# ---------------------------------------------------------------------------

def make_servo_source(
    port: str = "/dev/tty.usbserial-*",
    servo_id: int = 1,
    poll_hz: float = 100.0,
    sync_buffer: SyncBuffer | None = None,
) -> FeetechServoSource | SimulatedServoSource:
    """Use real Feetech servo if SDK and port are available; else simulated."""
    resolved = _resolve_port(port)
    if not resolved:
        print(f"No port matching '{port}'; using simulated encoder.")
        return SimulatedServoSource(poll_hz=poll_hz, sync_buffer=sync_buffer)
    if not _HAS_FEETECH_SDK:
        print("Feetech SDK not found; using simulated encoder.")
        return SimulatedServoSource(poll_hz=poll_hz, sync_buffer=sync_buffer)
    try:
        src = FeetechServoSource(
            port=resolved,
            servo_id=servo_id,
            poll_hz=poll_hz,
            sync_buffer=sync_buffer,
        )
        src.start()
        time.sleep(0.2)
        n = src.sync_buffer.counts()["servo"] if src.sync_buffer else 0
        src.stop()
        if n > 0:
            print(f"Feetech servo found on {resolved} (ID={servo_id})")
            return FeetechServoSource(
                port=resolved,
                servo_id=servo_id,
                poll_hz=poll_hz,
                sync_buffer=sync_buffer,
            )
    except Exception as exc:
        raise exc
        print(f"Feetech servo not available ({exc}); using simulated encoder.")
    return SimulatedServoSource(poll_hz=poll_hz, sync_buffer=sync_buffer)
