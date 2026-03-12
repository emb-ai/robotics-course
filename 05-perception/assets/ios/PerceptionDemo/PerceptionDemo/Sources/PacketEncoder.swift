import Foundation
import ARKit
import CoreMotion
import Accelerate

// Wire protocol type bytes — must match ws_server.py
enum PacketType: UInt8 {
    case imu     = 0x01
    case rgb     = 0x02
    case depth   = 0x03
    case control = 0x10
}

struct PacketEncoder {

    // MARK: - Control

    static func encodeControl(cmd: String) -> Data {
        let json = #"{"cmd":"\#(cmd)"}"#
        var data = Data([PacketType.control.rawValue])
        data.append(json.data(using: .utf8)!)
        return data
    }

    // MARK: - IMU

    static func encodeIMU(motion: CMDeviceMotion, timestamp: Double) -> Data {
        let payload: [String: Any] = [
            "t": timestamp,
            "accel": [motion.userAcceleration.x * 9.81,
                      motion.userAcceleration.y * 9.81,
                      motion.userAcceleration.z * 9.81],
            "gyro": [motion.rotationRate.x,
                     motion.rotationRate.y,
                     motion.rotationRate.z],
            "gravity": [motion.gravity.x * 9.81,
                        motion.gravity.y * 9.81,
                        motion.gravity.z * 9.81]
        ]
        guard let jsonData = try? JSONSerialization.data(withJSONObject: payload) else {
            return Data()
        }
        var data = Data([PacketType.imu.rawValue])
        data.append(jsonData)
        return data
    }

    // MARK: - RGB

    static func encodeRGB(pixelBuffer: CVPixelBuffer, timestamp: Double, jpegQuality: CGFloat = 0.5) -> Data? {
        guard let uiImage = uiImageFromPixelBuffer(pixelBuffer),
              let jpegData = uiImage.jpegData(compressionQuality: jpegQuality) else {
            return nil
        }
        var data = Data([PacketType.rgb.rawValue])
        var ts = Float(timestamp)
        data.append(Data(bytes: &ts, count: 4))
        data.append(jpegData)
        return data
    }

    // MARK: - Depth + Pose

    static func encodeDepth(
        depthBuffer: CVPixelBuffer,
        pose: simd_float4x4,
        intrinsics: simd_float3x3,
        timestamp: Double
    ) -> Data? {
        guard let depthU16 = depthToU16Millimetres(depthBuffer) else { return nil }

        let width = CVPixelBufferGetWidth(depthBuffer)
        let height = CVPixelBufferGetHeight(depthBuffer)

        var data = Data([PacketType.depth.rawValue])

        // timestamp (4 bytes)
        var ts = Float(timestamp)
        data.append(Data(bytes: &ts, count: 4))

        // pose 4×4 column-major (64 bytes)
        data.append(matrixToData(pose))

        // intrinsics 3×3 column-major (36 bytes)
        data.append(matrix3ToData(intrinsics))

        // width, height (4 bytes)
        var w = UInt16(width), h = UInt16(height)
        data.append(Data(bytes: &w, count: 2))
        data.append(Data(bytes: &h, count: 2))

        // raw uint16 depth in row-major order (width*height*2 bytes)
        data.append(depthU16)
        return data
    }

    // MARK: - Helpers

    private static func depthToU16Millimetres(_ buffer: CVPixelBuffer) -> Data? {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }

        let width = CVPixelBufferGetWidth(buffer)
        let height = CVPixelBufferGetHeight(buffer)
        guard let baseAddress = CVPixelBufferGetBaseAddress(buffer) else { return nil }

        let srcPtr = baseAddress.assumingMemoryBound(to: Float32.self)
        var result = Data(count: width * height * 2)
        result.withUnsafeMutableBytes { dst in
            let dstPtr = dst.bindMemory(to: UInt16.self)
            for i in 0..<(width * height) {
                let metres = srcPtr[i]
                let mm = metres * 1000.0
                dstPtr[i] = UInt16(clamping: Int(mm.rounded()))
            }
        }
        return result
    }

    private static func matrixToData(_ m: simd_float4x4) -> Data {
        // column-major: col0 then col1 then col2 then col3
        var floats: [Float] = [
            m.columns.0.x, m.columns.0.y, m.columns.0.z, m.columns.0.w,
            m.columns.1.x, m.columns.1.y, m.columns.1.z, m.columns.1.w,
            m.columns.2.x, m.columns.2.y, m.columns.2.z, m.columns.2.w,
            m.columns.3.x, m.columns.3.y, m.columns.3.z, m.columns.3.w,
        ]
        return Data(bytes: &floats, count: floats.count * 4)
    }

    private static func matrix3ToData(_ m: simd_float3x3) -> Data {
        var floats: [Float] = [
            m.columns.0.x, m.columns.0.y, m.columns.0.z,
            m.columns.1.x, m.columns.1.y, m.columns.1.z,
            m.columns.2.x, m.columns.2.y, m.columns.2.z,
        ]
        return Data(bytes: &floats, count: floats.count * 4)
    }

    private static func uiImageFromPixelBuffer(_ buffer: CVPixelBuffer) -> UIImage? {
        let ciImage = CIImage(cvPixelBuffer: buffer)
        let context = CIContext()
        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else { return nil }
        return UIImage(cgImage: cgImage)
    }
}
