import Foundation
import Network

@MainActor
class USBTransport: ObservableObject {

    static let listenPort: UInt16 = 7777

    @Published var isConnected = false
    @Published var statusMessage = "Not started"

    private var listener: NWListener?
    private var connection: NWConnection?

    func startListening() {
        guard let port = NWEndpoint.Port(rawValue: Self.listenPort) else { return }
        do {
            listener = try NWListener(using: .tcp, on: port)
        } catch {
            statusMessage = "Listener error: \(error)"
            return
        }

        listener?.stateUpdateHandler = { [weak self] state in
            DispatchQueue.main.async {
                guard let self else { return }
                switch state {
                case .ready:
                    self.statusMessage = "Listening on :\(Self.listenPort) — run iproxy \(Self.listenPort) \(Self.listenPort)"
                case .failed(let err):
                    self.statusMessage = "Listener error: \(err)"
                default:
                    break
                }
            }
        }
        listener?.newConnectionHandler = { [weak self] conn in
            DispatchQueue.main.async { self?.accept(conn) }
        }
        listener?.start(queue: .main)
        statusMessage = "Starting…"
    }

    func stopListening() {
        connection?.cancel()
        listener?.cancel()
        connection = nil
        listener = nil
        isConnected = false
        statusMessage = "Stopped"
    }

    private func accept(_ conn: NWConnection) {
        connection?.cancel()
        connection = conn
        conn.stateUpdateHandler = { [weak self] state in
            DispatchQueue.main.async {
                guard let self else { return }
                switch state {
                case .ready:
                    self.isConnected = true
                    self.statusMessage = "Mac connected ✓"
                case .failed(let err):
                    self.isConnected = false
                    self.statusMessage = "Lost: \(err) — waiting…"
                case .cancelled:
                    self.isConnected = false
                    self.statusMessage = "Disconnected — waiting…"
                default:
                    break
                }
            }
        }
        conn.start(queue: .main)
    }

    func send(_ data: Data) {
        guard isConnected, let conn = connection else { return }
        var length = UInt32(data.count).bigEndian
        var framed = Data(bytes: &length, count: 4)
        framed.append(data)
        conn.send(content: framed, completion: .contentProcessed { [weak self] error in
            if let error {
                DispatchQueue.main.async {
                    self?.isConnected = false
                    self?.statusMessage = "Send error: \(error)"
                }
            }
        })
    }
}
