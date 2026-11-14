import Foundation
import MLX
import MLXToolKit
import RelayVSRMLX

/// Compute precision. `bf16` is the default: generator bf16 (its released dtype and upstream's inference dtype),
/// decoder fp16 (see `RelayVSRDTypes.production` for why not bf16). `fp32` is the parity lane the S1 / E2E gates run;
/// it reads the same files and upcasts them exactly.
public enum RelayVSRPrecision: String, Codable, Sendable, CaseIterable {
    case bf16
    case fp32

    var quant: Quant { self == .bf16 ? .bf16 : .fp32 }
    var dtypes: RelayVSRDTypes { self == .bf16 ? .production : .parity }
}

/// Init-time configuration for `RelayVSRUpscalePackage` (C9).
///
/// Materialization (contract 1.24): ONE Hub repo, in mlx-community's `<UpstreamRepoName>-<precision>` grammar —
/// `mlx-community/RelayVSR-bf16` (10.2 GB): this repo's `oracle/convert_weights.py` output from HF `kopper/RelayVSR`
/// (Apache-2.0), the generator in its released bf16 and the decoder in its released fp32. Both precisions load it (fp32
/// is an exact upcast), so a fresh machine's engine downloads the same lane either way before `load()`.
/// `weightsDirectory` (a local lane directory) always wins and never touches the network.
public struct RelayVSRConfiguration: PackageConfiguration, ModelStorable {
    public static let repo = "mlx-community/RelayVSR-bf16"

    public var precision: RelayVSRPrecision
    /// Noise seed (the reference generator is one-step generative: the seed picks the detail it proposes).
    public var seed: UInt64
    /// Reference interval + 1 (upstream `--window-size`, default 16): a keyframe every `windowSize − 1` frames, and
    /// as many frames of lookahead. Smaller = more generator calls (slower, more detail between keyframes).
    public var windowSize: Int
    /// Explicit directory holding the lane's files; host-specific → excluded from `Codable`.
    public var weightsDirectory: URL?
    /// Stamped by the engine from its `ModelStore`; excluded from `Codable`.
    public var modelsRootDirectory: URL?

    public init(precision: RelayVSRPrecision = .bf16, seed: UInt64 = 42, windowSize: Int = 16,
                weightsDirectory: URL? = nil, modelsRootDirectory: URL? = nil) {
        self.precision = precision
        self.seed = seed
        self.windowSize = windowSize
        self.weightsDirectory = weightsDirectory
        self.modelsRootDirectory = modelsRootDirectory
    }

    private enum CodingKeys: String, CodingKey { case precision, seed, windowSize }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        precision = try c.decodeIfPresent(RelayVSRPrecision.self, forKey: .precision) ?? .bf16
        seed = try c.decodeIfPresent(UInt64.self, forKey: .seed) ?? 42
        windowSize = max(2, try c.decodeIfPresent(Int.self, forKey: .windowSize) ?? 16)
    }

    public var fileNames: [String] { RelayVSRComponent.files }

    /// Where `load()` reads the lane: the explicit directory if set, else the store's flat repo directory
    /// (`<root>/models--mlx-community--RelayVSR-bf16/`, where the engine's materializer lands files).
    public func resolvedWeightsDirectory(storeRoot: URL? = nil) -> URL? {
        if let weightsDirectory { return weightsDirectory }
        return ModelStore(root: storeRoot ?? modelsRootDirectory).directory(for: Self.repo)
    }

    var options: RelayVSROptions { RelayVSROptions(windowSize: max(2, windowSize), seed: seed) }
}

extension RelayVSRConfiguration: WeightSourcing {
    public var weightSources: [WeightSource] {
        [WeightSource(role: "relayvsr-bf16", repo: Self.repo, revision: "main", matching: fileNames)]
    }

    /// Explicit `weightsDirectory` first (complete → nothing missing, incomplete → the lane is missing), then the
    /// default probe over the store.
    public func missingWeightSources(storeRoot: URL?) -> [WeightSource] {
        if let weightsDirectory {
            let present = fileNames.allSatisfy {
                FileManager.default.fileExists(atPath: weightsDirectory.appendingPathComponent($0).path)
            }
            return present ? [] : weightSources
        }
        return defaultMissingWeightSources(storeRoot: storeRoot)
    }
}

extension RelayVSRConfiguration: WeightPrewarming {
    public var prewarmPaths: [URL] {
        guard let dir = resolvedWeightsDirectory() else { return [] }
        return fileNames.filter { $0.hasSuffix(".safetensors") }.map { dir.appendingPathComponent($0) }
    }
}

extension RelayVSRConfiguration: QuantConfigured {
    public var quant: Quant { precision.quant }
}

/// FIT-3: map a request to output pixels per frame (the declared `ActivationScaling` axis), so the engine can refuse a
/// run past the measured ceiling BEFORE loading 10 GB of weights. The canonical `Video` carries no geometry, so the
/// dimensions are read from the container's video `tkhd` box (16.16 fixed-point width/height) — synchronous, a few
/// microseconds, no AVFoundation. The model always computes at ×4 of the source padded up to multiples of 8, whatever
/// scale is asked for (×2 is the ×4 result box-averaged), so the workload is 16 × the padded source pixels.
/// `nil` (unparseable container, other request types) is never refused.
extension RelayVSRConfiguration: WorkloadDeclaring {
    public func workloadUnits(for request: any CapabilityRequest) -> Double? {
        guard let req = request as? VideoUpscaleRequest, let (w, h) = Self.videoDimensions(req.video.data) else {
            return nil
        }
        let pad = { (n: Int) in (n + 7) / 8 * 8 * 4 }
        return Double(pad(w) * pad(h))
    }

    /// Width × height of the first `tkhd` with a non-zero size (the video track) in an ISO-BMFF / QuickTime file.
    static func videoDimensions(_ data: Data) -> (Int, Int)? {
        let bytes = [UInt8](data)
        let tag: [UInt8] = Array("tkhd".utf8)
        var i = 4
        while i + 4 <= bytes.count {
            if bytes[i] == tag[0] && bytes[i + 1] == tag[1] && bytes[i + 2] == tag[2] && bytes[i + 3] == tag[3] {
                let body = i + 4                                   // version(1) flags(3) …
                guard body < bytes.count else { return nil }
                let off = body + (bytes[body] == 1 ? 88 : 76)      // width, height: 16.16 fixed, big-endian
                if off + 8 <= bytes.count {
                    func u32(_ o: Int) -> Int {
                        (Int(bytes[o]) << 24) | (Int(bytes[o + 1]) << 16) | (Int(bytes[o + 2]) << 8) | Int(bytes[o + 3])
                    }
                    let (w, h) = (u32(off) >> 16, u32(off + 4) >> 16)
                    if w > 0 && h > 0 { return (w, h) }
                }
                i = body
            } else {
                i += 1
            }
        }
        return nil
    }
}
