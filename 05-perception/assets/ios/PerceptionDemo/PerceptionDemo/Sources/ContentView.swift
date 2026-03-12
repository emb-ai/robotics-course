import SwiftUI

struct ContentView: View {

    @StateObject private var transport = USBTransport()
    @StateObject private var sensorService = SensorService()

    @State private var isStreaming = false

    var body: some View {
        NavigationView {
            Form {
                usbSection
                statusSection
                streamTogglesSection
                controlSection
                diagnosticsSection
            }
            .navigationTitle("Perception Demo")
        }
    }

    // MARK: - Sections

    private var usbSection: some View {
        Section("USB Connection") {
            HStack {
                Image(systemName: "cable.connector")
                    .foregroundColor(.secondary)
                Text("Port \(USBTransport.listenPort)")
                    .monospacedDigit()
            }
            Text("Connect iPhone via USB-C, then on Mac:\niproxy \(USBTransport.listenPort) \(USBTransport.listenPort)")
                .font(.caption)
                .foregroundColor(.secondary)
        }
    }

    private var statusSection: some View {
        Section("Status") {
            HStack {
                Circle()
                    .fill(transport.isConnected ? Color.green : Color.red)
                    .frame(width: 10, height: 10)
                Text(transport.statusMessage)
            }
            HStack {
                Circle()
                    .fill(sensorService.arStatus == "Running" ? Color.green : Color.orange)
                    .frame(width: 10, height: 10)
                Text("AR: \(sensorService.arStatus)")
            }
        }
    }

    private var streamTogglesSection: some View {
        Section("Streams") {
            Toggle("IMU (100 Hz)", isOn: $sensorService.streamIMU)
            Toggle("RGB frames (~5 Hz)", isOn: $sensorService.streamRGB)
            Toggle("Depth + Pose (~2 Hz)", isOn: $sensorService.streamDepth)
        }
    }

    private var controlSection: some View {
        Section {
            if !isStreaming {
                Button(action: startStreaming) {
                    Label("Start Listening", systemImage: "play.fill")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
            } else {
                Button(action: stopStreaming) {
                    Label("Stop", systemImage: "stop.fill")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .tint(.red)
            }
        }
    }

    private var diagnosticsSection: some View {
        Section("Packets sent") {
            HStack {
                Text("IMU")
                Spacer()
                Text("\(sensorService.imuSentCount)")
                    .monospacedDigit()
                    .foregroundColor(.secondary)
            }
            HStack {
                Text("RGB")
                Spacer()
                Text("\(sensorService.rgbSentCount)")
                    .monospacedDigit()
                    .foregroundColor(.secondary)
            }
            HStack {
                Text("Depth")
                Spacer()
                Text("\(sensorService.depthSentCount)")
                    .monospacedDigit()
                    .foregroundColor(.secondary)
            }
        }
    }

    // MARK: - Actions

    private func startStreaming() {
        transport.startListening()
        sensorService.start(transport: transport)
        isStreaming = true
    }

    private func stopStreaming() {
        sensorService.stop()
        transport.stopListening()
        isStreaming = false
    }
}

#Preview {
    ContentView()
}
