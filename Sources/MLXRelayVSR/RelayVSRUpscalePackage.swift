import CoreMedia
import CoreVideo
import Foundation
import FrameStreamNative
import MLX
import MLXNN
import MLXToolKit
import RelayVSRMLX

public enum RelayVSRPackageError: Error, CustomStringConvertible {
    /// The converted weights are not on this machine.
    case weightsNotProvisioned(expected: String)
    case unsupportedScale(Int)
    case pixelBuffer(String)

    public var description: String {
        switch self {
        case .weightsNotProvisioned(let p):
            return "RelayVSR weights not found at \(p). The engine downloads them from \(RelayVSRConfiguration.repo) on "
                + "first use; to run without the network, point RelayVSRConfiguration.weightsDirectory at a lane "
                + "directory (`oracle/convert_weights.py <kopper/RelayVSR snapshot> <dir>`)."
        case .unsupportedScale(let s): return "RelayVSR supports ×4 (native) and ×2 (got ×\(s))"
        case .pixelBuffer(let d): return "RelayVSR pixel buffer: \(d)"
        }
    }
}

/// An MLXEngine `videoUpscale` package over **RelayVSR** (kopperx, Apache-2.0 code and weights; arXiv 2609.37850):
/// a one-step Wan2.2-TI2V-5B reference generator that runs on one frame in fifteen, and a 32 M-parameter dual-memory
/// transformer that super-resolves EVERY frame from the low-resolution frame and the two endpoint reference latents.
///
/// The faithful generative tier. It stays close to the source — it keeps defocus soft and a drawing a drawing — so,
/// unlike FlashVSR, it is fit for anime and graphics as well as live action. It is also the softest of the fleet's
/// generative tiers: it removes grain, adds little high-frequency energy, and does not repair codec damage.
///
/// The run streams: AVFoundation decode (frame-stream-native, no FFmpeg) → `RelayVSRStream` → HEVC encode. Output
/// lags input by at most `windowSize − 1` frames (an interval is decoded once its far endpoint has arrived); every
/// source frame gets exactly one output frame carrying its own PTS (`timedTransform` / `timedFlush`).
///
/// Scale: ×4 native (the checkpoint is ×4 only). ×2 = the ×4 result averaged over 2×2 boxes — the same cost as ×4.
/// Sizes: the source is edge-padded to multiples of 8 and the output cropped back.
@InferenceActor
public final class RelayVSRUpscalePackage: ModelPackage {
    public typealias Configuration = RelayVSRConfiguration

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            // C7: the RelayVSR checkpoints are Apache-2.0 (HF model card + LICENSE; base Wan2.2-TI2V-5B Apache-2.0).
            // C8: the port is derived from upstream's Apache-2.0 code (refvsr_inference/) → Apache-2.0.
            license: LicenseDeclaration(weightLicense: .apache2, portCodeLicense: .apache2),
            // the lane repo is converted from kopper/RelayVSR @ 751954dc (its config.json pins the source hashes)
            provenance: Provenance(sourceRepo: RelayVSRConfiguration.repo, revision: "main", tier: 2),
            requirements: RequirementsManifest(
                footprints: RelayVSRUpscalePackage.footprints,
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
                chipFloor: nil
            ),
            specialties: [],
            surfaces: [
                VideoUpscaleContract.descriptor(
                    name: "relayvsr-upscale",
                    summary: "RelayVSR streaming video super-resolution (x4 native, x2): sparse generative reference "
                        + "keyframes relayed through a light per-frame network. Faithful to the source — keeps "
                        + "defocus and drawn content intact; every frame kept. Does not repair codec damage."
                )
            ]
        )
    }

    private let configuration: Configuration
    private var pipeline: RelayVSRPipeline?

    public nonisolated init(configuration: Configuration) {
        self.configuration = configuration
    }

    public func load() async throws {
        guard pipeline == nil else { return }
        guard let dir = configuration.resolvedWeightsDirectory(),
              configuration.fileNames.allSatisfy({
                  FileManager.default.fileExists(atPath: dir.appendingPathComponent($0).path)
              }) else {
            throw RelayVSRPackageError.weightsNotProvisioned(
                expected: configuration.resolvedWeightsDirectory()?.path ?? "<no weightsDirectory and no store root>")
        }
        pipeline = try RelayVSRPipeline.load(directory: dir, dtypes: configuration.precision.dtypes)
    }

    public func unload() async {
        pipeline = nil
        MLX.Memory.clearCache()
    }

    /// C14 seam: the loaded pipeline (nil until `load()`).
    var loadedPipeline: RelayVSRPipeline? { pipeline }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        try Task.checkCancellation()                       // CAN-1: first act, before notLoaded validation
        guard let pipeline else { throw PackageError.notLoaded }
        guard request.capability == .videoUpscale, let req = request as? VideoUpscaleRequest else {
            throw PackageError.unsupportedCapability(request.capability)
        }
        let scale = req.scale ?? 4
        guard scale == 2 || scale == 4 else { throw RelayVSRPackageError.unsupportedScale(scale) }

        let tmp = FileManager.default.temporaryDirectory
        let inURL = tmp.appendingPathComponent(UUID().uuidString).appendingPathExtension(req.video.format.rawValue)
        let outURL = tmp.appendingPathComponent(UUID().uuidString).appendingPathExtension("mp4")
        try req.video.data.write(to: inURL)
        defer {
            try? FileManager.default.removeItem(at: inURL)
            try? FileManager.default.removeItem(at: outURL)
        }
        let info = try await NativeFrameStream.probe(url: inURL)
        let expected = max(1, Int((info.duration * info.frameRate).rounded()))

        let driver = RelayVSRDriver(pipeline: pipeline, configuration: configuration, scale: scale)
        let meta = try await NativeFrameStream.run(
            input: inURL, output: outURL, timing: .preserveSource,
            timedTransform: { frame, pts in
                try Task.checkCancellation()               // CAN cadence: once per decoded source frame
                let out = try driver.ingest(frame, pts: pts)
                RunProgress.report(.upsample, step: driver.emitted, totalSteps: expected)
                return out
            },
            timedFlush: {
                let out = try driver.flush()
                RunProgress.report(.upsample, step: driver.emitted, totalSteps: expected)
                return out
            })
        let data = try Data(contentsOf: outURL)
        return VideoUpscaleResponse(
            video: Video(format: .mp4, data: data, durationSeconds: meta.sourceDuration,
                         frameRate: meta.sourceFrameRate),
            appliedScale: scale)
    }
}

/// One clip: BGRA pixel buffers in, (SR pixel buffer, PTS) out — in order, lagging up to `windowSize − 1` frames.
final class RelayVSRDriver {
    let pipeline: RelayVSRPipeline
    let configuration: RelayVSRConfiguration
    let scale: Int
    private var stream: RelayVSRStream?
    private var pendingPTS: [CMTime] = []
    private(set) var ingested = 0, emitted = 0

    init(pipeline: RelayVSRPipeline, configuration: RelayVSRConfiguration, scale: Int) {
        self.pipeline = pipeline
        self.configuration = configuration
        self.scale = scale
    }

    func ingest(_ pb: CVPixelBuffer, pts: CMTime) throws -> [(CVPixelBuffer, CMTime)] {
        let x = try Self.rgb255(from: pb)                                       // (H, W, 3) 0…255
        if stream == nil { stream = RelayVSRStream(pipeline: pipeline, options: configuration.options) }
        pendingPTS.append(pts)
        ingested += 1
        return try emit(stream!.push(x))
    }

    func flush() throws -> [(CVPixelBuffer, CMTime)] {
        guard let stream, ingested > 0 else { return [] }
        return try emit(stream.finish())
    }

    private func emit(_ frames: [MLXArray]) throws -> [(CVPixelBuffer, CMTime)] {
        try frames.map { f in
            var y = (f.asType(.float32) + 1) / 2                                // (4H, 4W, 3) in [0, 1]
            if scale == 2 { y = Self.boxHalve(y) }
            emitted += 1
            return (try Self.pixelBuffer(from: y), pendingPTS.removeFirst())
        }
    }

    /// (2h, 2w, 3) → (h, w, 3): the mean of every 2×2 box (an exact antialiased ×½ of the ×4 result).
    static func boxHalve(_ x: MLXArray) -> MLXArray {
        let (h, w) = (x.dim(0) / 2, x.dim(1) / 2)
        return x[0 ..< (h * 2), 0 ..< (w * 2)].reshaped([h, 2, w, 2, 3]).mean(axes: [1, 3])
    }

    /// 32BGRA → (H, W, 3) float32 RGB in 0…255.
    static func rgb255(from pb: CVPixelBuffer) throws -> MLXArray {
        guard CVPixelBufferGetPixelFormatType(pb) == kCVPixelFormatType_32BGRA else {
            throw RelayVSRPackageError.pixelBuffer("expected 32BGRA")
        }
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        let (w, h, bpr) = (CVPixelBufferGetWidth(pb), CVPixelBufferGetHeight(pb), CVPixelBufferGetBytesPerRow(pb))
        guard let base = CVPixelBufferGetBaseAddress(pb) else { throw RelayVSRPackageError.pixelBuffer("no base") }
        var packed = [UInt8](repeating: 0, count: w * h * 4)
        packed.withUnsafeMutableBytes { dst in
            for row in 0 ..< h { memcpy(dst.baseAddress! + row * w * 4, base + row * bpr, w * 4) }
        }
        let bgra = MLXArray(packed, [h, w, 4])
        return concatenated([bgra[.ellipsis, 2 ..< 3], bgra[.ellipsis, 1 ..< 2], bgra[.ellipsis, 0 ..< 1]], axis: -1)
            .asType(.float32)
    }

    /// (H, W, 3) float RGB in [0, 1] → 32BGRA (clamped, rounded, opaque).
    static func pixelBuffer(from x: MLXArray) throws -> CVPixelBuffer {
        let (h, w) = (x.dim(0), x.dim(1))
        let q = (clip(x.asType(.float32), min: 0, max: 1) * 255).round().asType(.uint8)
        let bgra = concatenated([q[.ellipsis, 2 ..< 3], q[.ellipsis, 1 ..< 2], q[.ellipsis, 0 ..< 1],
                                 MLXArray.full([h, w, 1], values: MLXArray(UInt8(255)))], axis: -1)
        let bytes = bgra.reshaped([h * w * 4]).asArray(UInt8.self)
        var pb: CVPixelBuffer?
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: w, kCVPixelBufferHeightKey as String: h,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:],
        ]
        guard CVPixelBufferCreate(nil, w, h, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb) == kCVReturnSuccess,
              let buffer = pb else { throw RelayVSRPackageError.pixelBuffer("allocate \(w)x\(h)") }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        let dst = CVPixelBufferGetBaseAddress(buffer)!
        let bpr = CVPixelBufferGetBytesPerRow(buffer)
        bytes.withUnsafeBytes { src in
            for row in 0 ..< h { memcpy(dst + row * bpr, src.baseAddress! + row * w * 4, w * 4) }
        }
        return buffer
    }
}

extension RelayVSRUpscalePackage {
    public nonisolated static var registration: PackageRegistration { .of(RelayVSRUpscalePackage.self) }

    public nonisolated static func requiredBytes(_ precision: RelayVSRPrecision) -> UInt64 {
        guard let fp = footprints.first(where: { $0.quant == precision.quant }) else { return .max }
        return fp.residentBytes + fp.peakActivationBytes
    }

    /// Split footprint — MEASURED through the real MLXServeEngine (`relayvsr-smoke engine`, 2026-11-14, M5 Max,
    /// thermal .nominal, GPU shared; one clip per process after a warm-up of the same size; basis = process
    /// lifetime-peak `phys_footprint`; dense masked attention):
    ///
    ///   input → output (×4)         lane   phys peak   MLX peak
    ///   320×192 → 1280×768          bf16   14.33 GB    12.75 GB
    ///   480×270 → 1920×1080 (1088)  bf16   16.80 GB    15.25 GB
    ///   320×192 → 1280×768          fp32   26.09 GB    24.43 GB
    ///
    /// Resident = the weights as loaded: bf16 lane 10.06 GB generator (bf16) + 0.06 GB decoder (fp16) + the cached
    /// prompt K/V ≈ 10.3 GB; fp32 lane 20.1 + 0.13 ≈ 20.4 GB. Streaming makes memory independent of clip length, so
    /// activation scales with OUTPUT PIXELS per frame (×2 costs what ×4 does: the model always computes ×4).
    /// bf16: the line through the two points, 12.13 GB + 2.23 KB/px of phys, less 10.19 GB of weights, ×1.2 headroom
    /// → 2.33 GB + 2.68 KB/px, measured ceiling 1920×1088; the scalar is the representative 1280×768 case,
    /// (14.33 − 10.19) × 1.2 + 0.256 ≈ 5.3 GB. fp32 (the parity lane) was measured at 1280×768 only: a flat line at
    /// its scalar, (26.09 − 20.3) × 1.2 + 0.256 ≈ 7.2 GB, ceiling 1280×768 — the engine refuses fp32 above it.
    nonisolated static var footprints: [QuantFootprint] {
        [
            QuantFootprint(quant: .bf16, residentBytes: 10_300_000_000, peakActivationBytes: 5_300_000_000,
                           activationScaling: ActivationScaling(axis: .outputPixels, baseBytes: 2_330_000_000,
                                                                bytesPerUnit: 2_680, measuredCeiling: 1920 * 1088)),
            QuantFootprint(quant: .fp32, residentBytes: 20_400_000_000, peakActivationBytes: 7_200_000_000,
                           activationScaling: ActivationScaling(axis: .outputPixels, baseBytes: 7_200_000_000,
                                                                bytesPerUnit: 0, measuredCeiling: 1280 * 768)),
        ]
    }
}
