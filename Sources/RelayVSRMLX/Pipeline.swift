// The keyframe / interval driver — port of kopperx/RelayVSR @ e856773 `refvsr_inference/runtime/streaming.py`
// (`CollaborativeFrameStream`) and the one-step sampler in `models/wan/sampling.py`.
//
// Schedule (window W, default 16): keyframes at 0, W − 1, 2(W − 1), … and the last frame. Each interval is decoded
// with BOTH endpoint references, so the stream looks ahead W − 1 frames. The interval's first frame is the previous
// interval's last: it is decoded again to rebuild the video memory, but not emitted twice. Every input frame comes
// out exactly once, in order, at exactly 4× its size.

import Foundation
import MLX
import MLXNN

public struct RelayVSROptions: Sendable {
    /// Reference interval + 1 (upstream `--window-size`): a keyframe every `windowSize − 1` frames.
    public var windowSize = 16
    public var seed: UInt64 = 42
    public init(windowSize: Int = 16, seed: UInt64 = 42) {
        self.windowSize = windowSize
        self.seed = seed
    }
}

/// What the gates can inject and observe; nil members cost nothing.
public struct RelayVSRHooks {
    /// The generator's sampling noise for keyframe ordinal j (0, 1, …), shape (1, h, w, 48). nil → seeded MLX noise.
    public var noise: ((Int, [Int]) -> MLXArray)?
    /// Called with every reference latent (1, h, w, 48) as it is produced.
    public var latent: ((Int, MLXArray) -> Void)?
    public init() {}
}

public struct RelayVSRStageTimes: Sendable {
    public var generator = 0.0, condition = 0.0, decoder = 0.0
    public var keyframes = 0, decodes = 0
}

public final class RelayVSRPipeline {
    public let generator: RelayVSRGenerator
    public let decoder: RelayVSRDecoder
    /// Wall seconds per stage of the last stream when `profile` is on (each stage is evaluated before the next).
    public var profile = false
    public internal(set) var stageTimes = RelayVSRStageTimes()

    public init(generator: RelayVSRGenerator, decoder: RelayVSRDecoder) {
        self.generator = generator
        self.decoder = decoder
    }

    /// The stride every LR frame is padded to (replicate, right and bottom): 16 · patch / 4 = 8.
    public static let stride = 8

    /// Upstream `prepare_lr`: (H, W, 3) with values 0…255 → (1, H', W', 3) in [−1, 1], H', W' multiples of 8.
    public static func prepareLR(_ frame: MLXArray) -> MLXArray {
        let (h, w) = (frame.dim(0), frame.dim(1))
        let x = (frame.asType(.float32) / 127.5 - 1).expandedDimensions(axis: 0)
        let (ph, pw) = ((stride - h % stride) % stride, (stride - w % stride) % stride)
        return ph == 0 && pw == 0 ? x : padded(x, widths: [0, .init((0, ph)), .init((0, pw)), 0], mode: .edge)
    }

    /// Run a clip. `frames`: LR frames (H, W, 3), values 0…255, constant size. `emit(index, frame)` receives every
    /// output frame (4H, 4W, 3) in [−1, 1] (unclamped), in order, as soon as its interval is decoded.
    public func run<S: Sequence>(frames: S, options: RelayVSROptions = .init(), hooks: RelayVSRHooks = .init(),
                                 emit: (Int, MLXArray) throws -> Void) rethrows where S.Element == MLXArray {
        let stream = RelayVSRStream(pipeline: self, options: options, hooks: hooks)
        var index = 0
        for f in frames {
            for out in stream.push(f) { try emit(index, out); index += 1 }
        }
        for out in stream.finish() { try emit(index, out); index += 1 }
    }
}

/// Frame-in / frames-out form of the driver: `push` returns the output frames that became decodable (none until an
/// interval's far endpoint arrives, then the whole interval), `finish` drains the last, partial interval. Outputs
/// are (4H, 4W, 3) in [−1, 1], in input order; the lag is at most `windowSize − 1` frames.
public final class RelayVSRStream {
    let pipeline: RelayVSRPipeline
    let options: RelayVSROptions
    let hooks: RelayVSRHooks
    private let state: RelayVSRGeneratorState
    private var key: MLXArray
    private var ordinal = 0
    private var size: (h: Int, w: Int)?
    /// The open interval: its start frame first (the previous interval's end), then the frames since.
    private var window: [MLXArray] = []
    private var startPrepared: RelayVSRPreparedCondition?
    private var firstWindow = true
    public private(set) var pushed = 0, emitted = 0

    public init(pipeline: RelayVSRPipeline, options: RelayVSROptions = .init(), hooks: RelayVSRHooks = .init()) {
        precondition(options.windowSize >= 2, "windowSize must be at least 2")
        self.pipeline = pipeline
        self.options = options
        self.hooks = hooks
        state = pipeline.generator.makeState()
        key = MLXRandom.key(options.seed)
        pipeline.stageTimes = RelayVSRStageTimes()
    }

    private func timed<R>(_ keyPath: WritableKeyPath<RelayVSRStageTimes, Double>, _ body: () -> R) -> R {
        guard pipeline.profile else { return body() }
        let t0 = DispatchTime.now().uptimeNanoseconds
        let r = body()
        pipeline.stageTimes[keyPath: keyPath] += Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9
        return r
    }

    /// One keyframe: bilinear ×4 → the one-step generator → reference latent → the decoder's persistent K/V.
    private func keyframe(_ frame: MLXArray) -> RelayVSRPreparedCondition {
        let gen = pipeline.generator
        let native = RelayVSRPipeline.prepareLR(frame)
        let latent: MLXArray = timed(\.generator) {
            let up = RelayVSRResample.upsample4(native, cubic: false)
            let shape = [1, up.dim(1) / 16, up.dim(2) / 16, gen.config.inChannels]
            let noise: MLXArray
            if let inject = hooks.noise {
                noise = inject(ordinal, shape)
            } else {
                let (k1, k2) = MLXRandom.split(key: key)
                key = k2
                noise = MLXRandom.normal(shape, key: k1)
            }
            var taps: RelayVSRTaps? = nil
            let pred = gen.step(noise: noise, lq: up, state: state, taps: &taps)
            // one Euler step of the flow from σ = 1 to 0, in fp32 (upstream: scheduler.step(pred.float(), …))
            let z = noise.asType(.float32) - pred.asType(.float32)
            eval(z)
            return z
        }
        hooks.latent?(ordinal, latent)
        ordinal += 1
        pipeline.stageTimes.keyframes += 1
        return timed(\.condition) { pipeline.decoder.prepareCondition(latent) }
    }

    /// (H, W, 3), values 0…255.
    public func push(_ frame: MLXArray) -> [MLXArray] {
        if let size {
            precondition(frame.dim(0) == size.h && frame.dim(1) == size.w, "frames must keep a constant size")
        } else {
            size = (frame.dim(0), frame.dim(1))
        }
        pushed += 1
        if startPrepared == nil {
            // the first reference can be computed while the producer collects lookahead
            startPrepared = keyframe(frame)
            window = [frame]
            return []
        }
        window.append(frame)
        return window.count == options.windowSize ? decodeWindow() : []
    }

    /// Drain the open interval (shorter than a full window, or the clip's only frame).
    public func finish() -> [MLXArray] {
        guard startPrepared != nil, !(window.count == 1 && !firstWindow) else { return [] }
        return decodeWindow()
    }

    private func decodeWindow() -> [MLXArray] {
        guard let start = startPrepared, let size else { return [] }
        let dec = pipeline.decoder
        let end = window.count == 1 ? start : keyframe(window[window.count - 1])
        var st = dec.beginStream(start: start, end: end)
        var outs: [MLXArray] = []
        for (offset, f) in window.enumerated() {
            let out: MLXArray = timed(\.decoder) {
                let o = dec.decodeStep(RelayVSRPipeline.prepareLR(f), state: &st)[0, 0 ..< (size.h * 4), 0 ..< (size.w * 4)]
                eval(o)
                return o
            }
            pipeline.stageTimes.decodes += 1
            // the overlap frame must update the history, but has already been emitted
            if firstWindow || offset > 0 { outs.append(out) }
        }
        firstWindow = false
        window = [window[window.count - 1]]
        startPrepared = end
        emitted += outs.count
        return outs
    }
}
