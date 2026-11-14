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
    public init() {}
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
    /// Wall seconds per stage of the last `run` when `profile` is on (each stage is evaluated before the next).
    public var profile = false
    public private(set) var stageTimes = RelayVSRStageTimes()

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

    private func timed<R>(_ keyPath: WritableKeyPath<RelayVSRStageTimes, Double>, _ body: () -> R) -> R {
        guard profile else { return body() }
        let t0 = DispatchTime.now().uptimeNanoseconds
        let r = body()
        stageTimes[keyPath: keyPath] += Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9
        return r
    }

    /// Run a clip. `frames`: LR frames (H, W, 3), values 0…255, constant size. `emit(index, frame)` receives every
    /// output frame (4H, 4W, 3) in [−1, 1] (unclamped), in order, as soon as its interval is decoded.
    public func run<S: Sequence>(frames: S, options: RelayVSROptions = .init(), hooks: RelayVSRHooks = .init(),
                                 emit: (Int, MLXArray) throws -> Void) rethrows where S.Element == MLXArray {
        precondition(options.windowSize >= 2, "windowSize must be at least 2")
        stageTimes = RelayVSRStageTimes()
        var iterator = frames.makeIterator()
        guard let first = iterator.next() else { return }
        let (h, w) = (first.dim(0), first.dim(1))
        let state = generator.makeState()
        var key = MLXRandom.key(options.seed)
        var ordinal = 0

        func keyframe(_ frame: MLXArray) -> RelayVSRPreparedCondition {
            let native = Self.prepareLR(frame)
            let latent: MLXArray = timed(\.generator) {
                let up = RelayVSRResample.upsample4(native, cubic: false)                  // bilinear ×4
                let shape = [1, up.dim(1) / 16, up.dim(2) / 16, generator.config.inChannels]
                let noise: MLXArray
                if let inject = hooks.noise {
                    noise = inject(ordinal, shape)
                } else {
                    let (k1, k2) = MLXRandom.split(key: key)
                    key = k2
                    noise = MLXRandom.normal(shape, key: k1)
                }
                var taps: RelayVSRTaps? = nil
                let pred = generator.step(noise: noise, lq: up, state: state, taps: &taps)
                // one Euler step of the flow from σ = 1 to 0, in fp32 (upstream: scheduler.step(pred.float(), …))
                let z = noise.asType(.float32) - pred.asType(.float32)
                eval(z)
                return z
            }
            hooks.latent?(ordinal, latent)
            ordinal += 1
            stageTimes.keyframes += 1
            return timed(\.condition) { decoder.prepareCondition(latent) }
        }

        var index = 0
        var start = (index: 0, frame: first)
        var startPrepared = keyframe(first)
        var firstWindow = true
        while true {
            var window = [start]
            while window.count < options.windowSize, let f = iterator.next() {
                index += 1
                precondition(f.dim(0) == h && f.dim(1) == w, "frames must keep a constant size")
                window.append((index, f))
            }
            if window.count == 1 && !firstWindow { return }
            let end = window[window.count - 1]
            let endPrepared = end.index == start.index ? startPrepared : keyframe(end.frame)
            var dec = decoder.beginStream(start: startPrepared, end: endPrepared)
            for (offset, p) in window.enumerated() {
                let out: MLXArray = timed(\.decoder) {
                    let o = decoder.decodeStep(Self.prepareLR(p.frame), state: &dec)[0, 0 ..< (h * 4), 0 ..< (w * 4)]
                    eval(o)
                    return o
                }
                stageTimes.decodes += 1
                // the overlap frame must update the history, but has already been emitted
                if firstWindow || offset > 0 { try emit(p.index, out) }
            }
            firstWindow = false
            start = end
            startPrepared = endPrepared
            if window.count < options.windowSize { return }
        }
    }
}
