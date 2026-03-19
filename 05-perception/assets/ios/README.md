# PerceptionDemo iOS App

Minimal iPhone app that streams sensor data to the Python perception notebook over USB-C.

## What it streams

| Stream | Rate | Format |
|---|---|---|
| IMU (accel + gyro + gravity) | 100 Hz | JSON, type 0x01 |
| RGB frames | ~5 Hz | JPEG, type 0x02 |
| Depth + Pose + Intrinsics | ~2 Hz | Binary, type 0x03 |

Requires iPhone with LiDAR (iPhone 12 Pro or later).

Packets are length-framed TCP: `[uint32 big-endian length][packet bytes]`.

## Build

### Prerequisites

```bash
brew install xcodegen
```

### Generate Xcode project and open

```bash
cd 05-perception/assets/ios/PerceptionDemo
xcodegen generate
open PerceptionDemo.xcodeproj
```

Connect your iPhone, select it as the build target, choose your own
development team in Signing if Xcode asks for it, and run.

## Usage

1. Install `libimobiledevice` (one-time):
   ```bash
   brew install libimobiledevice
   ```

2. Plug iPhone into Mac via **USB-C**.

3. On Mac, forward the port over USB:
   ```bash
   iproxy 7777 7777
   ```
   Leave this running in a terminal.

4. In the app, tap **Start Listening**. The status line will say `Listening on :7777`.

5. In the notebook, start the USB source:
   ```python
   from lib.phone_stream import PhoneUSBSource
   phone = PhoneUSBSource()
   phone.start()
   phone.wait_for_connection(timeout=15)
   ```

6. The app status will update to **Mac connected ✓** once Python connects.

## Offline fallback

If no iPhone is available, use `PhoneReplaySource` to replay pre-recorded data
from `assets/phone_recordings/`.
