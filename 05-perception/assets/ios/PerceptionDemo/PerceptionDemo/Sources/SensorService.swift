import ARKit
import CoreMotion
import Foundation

@MainActor
class SensorService: NSObject, ObservableObject {

    // Stream enable toggles (bound to UI)
    @Published var streamIMU = true
    @Published var streamRGB = true
    @Published var streamDepth = true

    // FPS targets
    var targetRgbFPS: Double = 5.0
    var targetDepthFPS: Double = 2.0

    // Diagnostics
    @Published var imuSentCount = 0
    @Published var rgbSentCount = 0
    @Published var depthSentCount = 0
    @Published var arStatus = "Not running"

    private let arSession = ARSession()
    private let motionManager = CMMotionManager()
    private var sessionStartTime: CFTimeInterval = 0

    private var lastRgbTime: CFTimeInterval = 0
    private var lastDepthTime: CFTimeInterval = 0

    weak var transport: USBTransport?

    func start(transport: USBTransport) {
        self.transport = transport
        sessionStartTime = CACurrentMediaTime()

        guard ARWorldTrackingConfiguration.isSupported else {
            arStatus = "ARKit not supported on this device"
            return
        }
        guard ARWorldTrackingConfiguration.supportsFrameSemantics([.sceneDepth, .smoothedSceneDepth]) else {
            arStatus = "LiDAR depth not supported on this device"
            return
        }

        let config = ARWorldTrackingConfiguration()
        config.frameSemantics = [.sceneDepth, .smoothedSceneDepth]
        config.isAutoFocusEnabled = false
        arSession.delegate = self
        arSession.run(config, options: [.resetTracking, .removeExistingAnchors])
        arStatus = "Running"

        motionManager.deviceMotionUpdateInterval = 1.0 / 100.0
        motionManager.startDeviceMotionUpdates(to: .main) { [weak self] motion, error in
            guard let self, let motion, error == nil else { return }
            Task { @MainActor in self.handleMotion(motion) }
        }
    }

    func stop() {
        arSession.pause()
        motionManager.stopDeviceMotionUpdates()
        arStatus = "Stopped"
    }

    private func handleMotion(_ motion: CMDeviceMotion) {
        guard streamIMU, let t = transport, t.isConnected else { return }
        let ts = CACurrentMediaTime() - sessionStartTime
        let packet = PacketEncoder.encodeIMU(motion: motion, timestamp: ts)
        t.send(packet)
        imuSentCount += 1
    }
}

// MARK: - ARSessionDelegate

extension SensorService: ARSessionDelegate {

    nonisolated func session(_ session: ARSession, didUpdate frame: ARFrame) {
        Task { @MainActor in processFrame(frame) }
    }

    nonisolated func session(_ session: ARSession, didFailWithError error: Error) {
        Task { @MainActor in arStatus = "Error: \(error.localizedDescription)" }
    }

    @MainActor
    private func processFrame(_ frame: ARFrame) {
        guard let t = transport, t.isConnected else { return }
        let ts = frame.timestamp - sessionStartTime
        let now = CACurrentMediaTime()

        // RGB
        if streamRGB && (now - lastRgbTime) >= 1.0 / targetRgbFPS {
            if let packet = PacketEncoder.encodeRGB(
                pixelBuffer: frame.capturedImage,
                timestamp: ts
            ) {
                t.send(packet)
                rgbSentCount += 1
                lastRgbTime = now
            }
        }

        // Depth + Pose
        if streamDepth && (now - lastDepthTime) >= 1.0 / targetDepthFPS {
            if let depthMap = (frame.smoothedSceneDepth ?? frame.sceneDepth)?.depthMap {
                let scaledK = scaleIntrinsics(
                    frame.camera.intrinsics,
                    from: frame.camera.imageResolution,
                    to: CGSize(
                        width: CVPixelBufferGetWidth(depthMap),
                        height: CVPixelBufferGetHeight(depthMap)
                    )
                )
                if let packet = PacketEncoder.encodeDepth(
                    depthBuffer: depthMap,
                    pose: frame.camera.transform,
                    intrinsics: scaledK,
                    timestamp: ts
                ) {
                    t.send(packet)
                    depthSentCount += 1
                    lastDepthTime = now
                }
            }
        }
    }

    private func scaleIntrinsics(
        _ K: simd_float3x3,
        from colorSize: CGSize,
        to depthSize: CGSize
    ) -> simd_float3x3 {
        let sx = Float(depthSize.width) / Float(colorSize.width)
        let sy = Float(depthSize.height) / Float(colorSize.height)
        var scaled = K
        scaled.columns.0[0] *= sx   // fx
        scaled.columns.1[1] *= sy   // fy
        scaled.columns.2[0] *= sx   // cx
        scaled.columns.2[1] *= sy   // cy
        return scaled
    }
}
