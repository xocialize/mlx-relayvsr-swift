// RelayVSR's lightweight VSR network — MLX-Swift port of kopperx/RelayVSR @ e856773
// `refvsr_inference/models/flash/{model,_flashvsr_ops,_local_attention}.py` (Apache-2.0): the paper's Dual-Memory
// Video Transformer (upstream class `FlashVSRConditionNet`, variant S: 10 blocks, dim 512, 8 heads over 2 K/V
// groups, 31.9 M params). Per LR frame it emits the ×4 frame as bicubic(LR) + a PixelShuffle ×16 residual, attending
// to   • the two endpoint reference latents (persistent K/V, primed once per keyframe),
//      • the previous frame (rolling K/V, window 2) and itself.
//
// Isomorphic to upstream: same module names and parameter paths (`lr_encoder.layers.{0,2,4,6}`, `condition_encoder`,
// `backbone.blocks.N.{attention_norm,attention.*,feed_forward_norm,feed_forward.*}`, `spatial_head.0`,
// `residual_projection`, …). Substitutions only:
//   • NCHW → NHWC; tokens are row-major exactly as upstream's `flatten(2).transpose(1, 2)`.
//   • Local attention: upstream's CUDA path is a fused Triton kernel and its portable path SDPA under the window
//     mask, 64 queries at a time; here ONE SDPA call under the same mask (measured identical in fp32 on the oracle).
//   • Grouped-query attention: K/V are repeated over the 4 heads of each group (what `enable_gqa` does).

import Foundation
import MLX
import MLXNN

public struct RelayVSRDecoderConfig: Sendable {
    public var dim = 512, depth = 10, numHeads = 8, numKVGroups = 2
    public var conditionChannels = 48
    public var encoderChannels = [64, 128, 128]
    public var hiddenDim = 1408                       // ceil(dim · 8/3 / 64) · 64
    public var spatialHeadDim = 1024
    public var windowSize = 2                         // video frames a query sees, the current one included
    public var upscale = 4
    public var eps: Float = 1e-5
    public var ropeTheta: Double = 10000
    /// Query-centred half-open windows, in tokens (h, w): the video planes, and the two reference planes.
    public var videoWindow = [24, 24], startWindow = [48, 48], endWindow = [48, 48], conditionSelfWindow = [48, 48]
    public var headDim: Int { dim / numHeads }
    public var pixelShuffle: Int { 4 * upscale }      // encoder downsample 4 × upscale 4
    public init() {}
}

// MARK: - RoPE (3-D: temporal 16 | height 24 | width 24 of the 64-wide head)

/// Upstream `_RotaryEmbedding3D`: inv_freq_j = θ^(−2j/d) per axis in fp32, angle = position · inv_freq, and the
/// rotation applied to adjacent channel pairs. Tables here are per PAIR: (tokens, 32).
struct DecoderRope {
    let invT: [Float], invH: [Float], invW: [Float]
    init(theta: Double, dims: (Int, Int, Int) = (16, 24, 24)) {
        func inv(_ d: Int) -> [Float] { (0 ..< d / 2).map { 1 / powf(Float(theta), Float(2 * $0) / Float(d)) } }
        (invT, invH, invW) = (inv(dims.0), inv(dims.1), inv(dims.2))
    }
    /// (cos, sin) for `temporal.count` planes of an (h, w) grid, plane-major then row-major: (planes·h·w, 32).
    func tables(temporal: [Int], h: Int, w: Int) -> (MLXArray, MLXArray) {
        let n = invT.count + invH.count + invW.count
        var c = [Float](repeating: 0, count: temporal.count * h * w * n), s = c
        var i = 0
        for t in temporal {
            for y in 0 ..< h {
                for x in 0 ..< w {
                    var o = i * n
                    for f in invT { let a = Float(t) * f; c[o] = cos(a); s[o] = sin(a); o += 1 }
                    for f in invH { let a = Float(y) * f; c[o] = cos(a); s[o] = sin(a); o += 1 }
                    for f in invW { let a = Float(x) * f; c[o] = cos(a); s[o] = sin(a); o += 1 }
                    i += 1
                }
            }
        }
        return (MLXArray(c, [temporal.count * h * w, n]), MLXArray(s, [temporal.count * h * w, n]))
    }
    /// Upstream `_RotaryEmbedding3D.apply`: x (B, heads, tokens, 64), in x's own dtype.
    static func apply(_ x: MLXArray, _ cs: (MLXArray, MLXArray)) -> MLXArray {
        let sh = x.shape
        let p = x.reshaped([sh[0], sh[1], sh[2], sh[3] / 2, 2])
        let even = p[.ellipsis, 0], odd = p[.ellipsis, 1]
        let c = cs.0.asType(x.dtype), s = cs.1.asType(x.dtype)
        return stacked([even * c - odd * s, even * s + odd * c], axis: -1).reshaped(sh)
    }
}

// MARK: - modules

final class LRFrameEncoder: Module {
    @ModuleInfo var layers: [UnaryLayer]
    let norm: TorchRMSNorm
    init(_ c: RelayVSRDecoderConfig) {
        let ch = c.encoderChannels
        _layers.wrappedValue = [
            Conv2d(inputChannels: 3, outputChannels: ch[0], kernelSize: 3, stride: 2, padding: 1), SiLU(),
            Conv2d(inputChannels: ch[0], outputChannels: ch[1], kernelSize: 3, stride: 2, padding: 1), SiLU(),
            Conv2d(inputChannels: ch[1], outputChannels: ch[2], kernelSize: 3, stride: 1, padding: 1), SiLU(),
            Conv2d(inputChannels: ch[2], outputChannels: c.dim, kernelSize: 1),
        ]
        norm = TorchRMSNorm(c.dim, eps: c.eps)
    }
    /// (B, H, W, 3) → normalised tokens at 1/4 resolution: (B, H/4 · W/4, dim).
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let f = seq(layers, x)
        return norm(f.reshaped([f.dim(0), f.dim(1) * f.dim(2), f.dim(3)]))
    }
}

final class ConditionEncoder: Module {
    let projection: Conv2d
    let norm: TorchRMSNorm
    init(_ c: RelayVSRDecoderConfig) {
        projection = Conv2d(inputChannels: c.conditionChannels, outputChannels: c.dim, kernelSize: 1)
        norm = TorchRMSNorm(c.dim, eps: c.eps)
    }
    /// (B, h, w, 48) Wan latent → (B, h·w, dim).
    func callAsFunction(_ z: MLXArray) -> MLXArray {
        let f = projection(z)
        return norm(f.reshaped([f.dim(0), f.dim(1) * f.dim(2), f.dim(3)]))
    }
}

final class SwiGLU: Module {
    @ModuleInfo(key: "gate_up_proj") var gateUpProj: Linear
    let down: Linear
    init(dim: Int, hidden: Int) {
        _gateUpProj.wrappedValue = Linear(dim, 2 * hidden, bias: false)
        down = Linear(hidden, dim, bias: false)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gu = gateUpProj(x)
        let h = gu.dim(-1) / 2
        return down(silu(gu[.ellipsis, 0 ..< h]) * gu[.ellipsis, h...])
    }
}

/// Upstream `_ConditionedGroupedQueryAttention`. K and V are both RMS-normalised per head and cached UNROTATED;
/// RoPE is applied at attention time, relative to the current query.
final class DecoderAttention: Module {
    @ModuleInfo(key: "query") var queryProj: Linear
    @ModuleInfo(key: "key") var keyProj: Linear
    @ModuleInfo(key: "value") var valueProj: Linear
    @ModuleInfo(key: "condition_key") var conditionKey: Linear
    @ModuleInfo(key: "condition_value") var conditionValue: Linear
    @ModuleInfo(key: "key_norm") var keyNorm: TorchRMSNorm
    @ModuleInfo(key: "value_norm") var valueNorm: TorchRMSNorm
    @ModuleInfo(key: "output") var outputProj: Linear
    let heads: Int, groups: Int, headDim: Int, dim: Int

    init(_ c: RelayVSRDecoderConfig) {
        (heads, groups, headDim, dim) = (c.numHeads, c.numKVGroups, c.headDim, c.dim)
        let kv = c.numKVGroups * c.headDim
        _queryProj.wrappedValue = Linear(c.dim, c.dim)
        _keyProj.wrappedValue = Linear(c.dim, kv)
        _valueProj.wrappedValue = Linear(c.dim, kv)
        _conditionKey.wrappedValue = Linear(c.dim, kv)
        _conditionValue.wrappedValue = Linear(c.dim, kv)
        _keyNorm.wrappedValue = TorchRMSNorm(c.headDim, eps: c.eps)
        _valueNorm.wrappedValue = TorchRMSNorm(c.headDim, eps: c.eps)
        _outputProj.wrappedValue = Linear(c.dim, c.dim)
    }

    /// Upstream `_project`: q (B, heads, N, hd); k, v (B, groups, N, hd), normalised.
    func project(_ x: MLXArray, condition: Bool) -> (q: MLXArray, k: MLXArray, v: MLXArray) {
        let (b, n) = (x.dim(0), x.dim(1))
        let q = queryProj(x).reshaped([b, n, heads, headDim]).transposed(0, 2, 1, 3)
        let kp = condition ? conditionKey : keyProj, vp = condition ? conditionValue : valueProj
        let k = keyNorm(kp(x).reshaped([b, n, groups, headDim]).transposed(0, 2, 1, 3))
        let v = valueNorm(vp(x).reshaped([b, n, groups, headDim]).transposed(0, 2, 1, 3))
        return (q, k, v)
    }

    /// Upstream `_attend`: rotate q and k, attend under the window mask (nil = everything visible), project out.
    func attend(q: MLXArray, k: MLXArray, v: MLXArray, qRope: (MLXArray, MLXArray), kRope: (MLXArray, MLXArray),
                mask: MLXArray?) -> MLXArray {
        let (b, n) = (q.dim(0), q.dim(2))
        let rep = heads / groups
        let qr = DecoderRope.apply(q, qRope)
        let kr = repeated(DecoderRope.apply(k, kRope), count: rep, axis: 1)
        let vr = repeated(v, count: rep, axis: 1)
        let a = MLXFast.scaledDotProductAttention(queries: qr, keys: kr, values: vr,
                                                  scale: 1 / Float(headDim).squareRoot(), mask: mask)
        return outputProj(a.transposed(0, 2, 1, 3).reshaped([b, n, dim]))
    }
}

final class DecoderBlock: Module {
    @ModuleInfo(key: "attention_norm") var attentionNorm: TorchRMSNorm
    let attention: DecoderAttention
    @ModuleInfo(key: "feed_forward_norm") var feedForwardNorm: TorchRMSNorm
    @ModuleInfo(key: "feed_forward") var feedForward: SwiGLU
    init(_ c: RelayVSRDecoderConfig) {
        _attentionNorm.wrappedValue = TorchRMSNorm(c.dim, eps: c.eps)
        attention = DecoderAttention(c)
        _feedForwardNorm.wrappedValue = TorchRMSNorm(c.dim, eps: c.eps)
        _feedForward.wrappedValue = SwiGLU(dim: c.dim, hidden: c.hiddenDim)
    }
    func feedForwardResidual(_ x: MLXArray) -> MLXArray { x + feedForward(feedForwardNorm(x)) }
}

final class DecoderBackbone: Module {
    let blocks: [DecoderBlock]
    init(_ c: RelayVSRDecoderConfig) { blocks = (0 ..< c.depth).map { _ in DecoderBlock(c) } }
}

// MARK: - state

/// One endpoint's per-block K/V (unrotated, normalised), upstream `FlashVSRPreparedCondition`.
public struct RelayVSRPreparedCondition {
    public let keys: [MLXArray], values: [MLXArray]          // per block: (B, groups, h·w, hd)
    public let height: Int, width: Int                       // token grid
}

/// Upstream `FlashVSRConditionState`: the persistent reference memory and the rolling video memory of one interval.
public struct RelayVSRDecoderState {
    var videoK: [MLXArray?], videoV: [MLXArray?]
    let conditionK: [MLXArray], conditionV: [MLXArray]
    let conditionPlanes: Int
    public internal(set) var frameIndex = 0
}

// MARK: - the network

public final class RelayVSRDecoder: Module {
    public let config: RelayVSRDecoderConfig
    @ModuleInfo(key: "lr_encoder") var lrEncoder: LRFrameEncoder
    @ModuleInfo(key: "condition_encoder") var conditionEncoder: ConditionEncoder
    @ParameterInfo(key: "video_type_embedding") var videoTypeEmbedding: MLXArray
    @ParameterInfo(key: "condition_type_embedding") var conditionTypeEmbedding: MLXArray
    let backbone: DecoderBackbone
    @ModuleInfo(key: "spatial_head") var spatialHead: [UnaryLayer]
    @ModuleInfo(key: "residual_projection") var residualProjection: Linear
    /// Training-only auxiliary head (direct RGB from the reference tokens); in the checkpoint, unused at inference.
    @ModuleInfo(key: "condition_rgb_projection") var conditionRGBProjection: Linear
    let rope: DecoderRope
    private var maskCache: [String: MLXArray?] = [:]
    private var ropeCache: [String: (MLXArray, MLXArray)] = [:]

    public init(_ c: RelayVSRDecoderConfig = .init()) {
        config = c
        _lrEncoder.wrappedValue = LRFrameEncoder(c)
        _conditionEncoder.wrappedValue = ConditionEncoder(c)
        _videoTypeEmbedding.wrappedValue = MLXArray.zeros([1, 1, c.dim])
        _conditionTypeEmbedding.wrappedValue = MLXArray.zeros([1, 1, c.dim])
        backbone = DecoderBackbone(c)
        _spatialHead.wrappedValue = [Linear(c.dim, c.spatialHeadDim), SiLU()]
        let px = 3 * c.pixelShuffle * c.pixelShuffle
        _residualProjection.wrappedValue = Linear(c.spatialHeadDim, px)
        _conditionRGBProjection.wrappedValue = Linear(c.spatialHeadDim, px)
        rope = DecoderRope(theta: c.ropeTheta)
        super.init()
        train(false)
    }

    public var dtype: DType { residualProjection.weight.dtype }

    // MARK: masks and RoPE plans

    /// Upstream `local_attention_mask`: per key plane a query-centred, half-open, boundary-truncated window —
    /// key (ky, kx) is visible to query (qy, qx) iff −wh/2 ≤ ky − qy < wh/2 and −ww/2 ≤ kx − qx < ww/2.
    /// (queries, planes·h·w) bool, or nil when every plane's window covers the whole grid.
    func mask(h: Int, w: Int, windows: [[Int]]) -> MLXArray? {
        let key = "\(h)x\(w):" + windows.map { "\($0[0]),\($0[1])" }.joined(separator: "|")
        if let m = maskCache[key] { return m }
        var m: MLXArray?
        if windows.contains(where: { $0[0] / 2 < h || $0[1] / 2 < w }) {
            let planes = windows.map { win in
                separableMask(h: h, w: w,
                              rowVisible: { q, k in k - q >= -(win[0] / 2) && k - q < win[0] / 2 },
                              colVisible: { q, k in k - q >= -(win[1] / 2) && k - q < win[1] / 2 })
            }
            m = planes.count == 1 ? planes[0] : concatenated(planes, axis: 1)
            eval(m!)
        }
        maskCache[key] = m
        return m
    }

    func ropeTables(temporal: [Int], h: Int, w: Int) -> (MLXArray, MLXArray) {
        let key = "\(h)x\(w):" + temporal.map(String.init).joined(separator: ",")
        if let t = ropeCache[key] { return t }
        let t = rope.tables(temporal: temporal, h: h, w: w)
        eval(t.0, t.1)
        ropeCache[key] = t
        return t
    }

    // MARK: reference priming

    /// Upstream `prepare_condition(latent)`: encode one endpoint's Wan latent once. Each block self-attends the
    /// reference tokens (window `condition_self`) and its `condition_key` / `condition_value` projections are
    /// kept as that block's persistent K/V. latent: (B, h, w, 48).
    public func prepareCondition(_ latent: MLXArray, taps: inout RelayVSRTaps?) -> RelayVSRPreparedCondition {
        let (h, w) = (latent.dim(1), latent.dim(2))
        var x = conditionEncoder(latent.asType(dtype)) + conditionTypeEmbedding
        taps?["tokens"] = x
        let r = ropeTables(temporal: [0], h: h, w: w)
        let m = mask(h: h, w: w, windows: [config.conditionSelfWindow])
        var ks: [MLXArray] = [], vs: [MLXArray] = []
        for (i, blk) in backbone.blocks.enumerated() {
            let (q, k, v) = blk.attention.project(blk.attentionNorm(x), condition: true)
            x = x + blk.attention.attend(q: q, k: k, v: v, qRope: r, kRope: r, mask: m)
            x = blk.feedForwardResidual(x)
            eval(x, k, v)
            ks.append(k); vs.append(v)
            taps?["b\(i).key"] = k; taps?["b\(i).value"] = v
        }
        taps?["hidden"] = x
        return RelayVSRPreparedCondition(keys: ks, values: vs, height: h, width: w)
    }

    public func prepareCondition(_ latent: MLXArray) -> RelayVSRPreparedCondition {
        var t: RelayVSRTaps? = nil
        return prepareCondition(latent, taps: &t)
    }

    /// Upstream `begin_stream(start, end)`: a fresh video history over two independently prepared endpoints.
    public func beginStream(start: RelayVSRPreparedCondition, end: RelayVSRPreparedCondition) -> RelayVSRDecoderState {
        precondition(start.height == end.height && start.width == end.width, "endpoint grids differ")
        let n = backbone.blocks.count
        return RelayVSRDecoderState(
            videoK: Array(repeating: nil, count: n), videoV: Array(repeating: nil, count: n),
            conditionK: zip(start.keys, end.keys).map { concatenated([$0, $1], axis: 2) },
            conditionV: zip(start.values, end.values).map { concatenated([$0, $1], axis: 2) },
            conditionPlanes: 2)
    }

    // MARK: decode

    /// Upstream `decode_step(lr_frame, state)`: one LR frame (B, H, W, 3) in [−1, 1] → (B, 4H, 4W, 3).
    public func decodeStep(_ lrFrame: MLXArray, state: inout RelayVSRDecoderState, taps: inout RelayVSRTaps?) -> MLXArray {
        let c = config
        let (b, height, width) = (lrFrame.dim(0), lrFrame.dim(1), lrFrame.dim(2))
        // preserve the caller's size; pad only the model input (upstream: replicate to a multiple of 4)
        let (padH, padW) = ((4 - height % 4) % 4, (4 - width % 4) % 4)
        var lr = lrFrame.asType(dtype)
        if padH > 0 || padW > 0 { lr = padded(lr, widths: [0, .init((0, padH)), .init((0, padW)), 0], mode: .edge) }
        let (th, tw) = ((height + padH) / 4, (width + padW) / 4), n = th * tw

        var x = lrEncoder(lr) + videoTypeEmbedding
        taps?["tokens"] = x
        // rolling video memory: keep the last (window − 1) frames
        let keep = (c.windowSize - 1) * n
        let cached = state.videoK.first.flatMap { $0 }?.dim(2) ?? 0
        let previous = min(cached, keep) / n
        let planes = state.conditionPlanes
        // unrotated cached keys are re-positioned relative to the current query: references at t = 0 (spatial RoPE
        // only), video frames at −previous … 0
        let qRope = ropeTables(temporal: [0], h: th, w: tw)
        let kRope = ropeTables(temporal: Array(repeating: 0, count: planes) + Array(-previous ... 0), h: th, w: tw)
        let windows = [c.startWindow, c.endWindow].prefix(planes) + Array(repeating: c.videoWindow, count: previous + 1)
        let m = mask(h: th, w: tw, windows: Array(windows))

        for (i, blk) in backbone.blocks.enumerated() {
            let (q, ck, cv) = blk.attention.project(blk.attentionNorm(x), condition: false)
            var vk = ck, vv = cv
            if keep > 0, let pk = state.videoK[i], let pv = state.videoV[i] {
                let have = pk.dim(2)
                let tk = have > keep ? pk[0..., 0..., (have - keep)...] : pk
                let tv = have > keep ? pv[0..., 0..., (have - keep)...] : pv
                vk = concatenated([tk, ck], axis: 2)
                vv = concatenated([tv, cv], axis: 2)
            }
            let k = concatenated([state.conditionK[i], vk], axis: 2)
            let v = concatenated([state.conditionV[i], vv], axis: 2)
            x = x + blk.attention.attend(q: q, k: k, v: v, qRope: qRope, kRope: kRope, mask: m)
            x = blk.feedForwardResidual(x)
            state.videoK[i] = vk
            state.videoV[i] = vv
            eval(x, vk, vv)
            taps?["b\(i)"] = x
        }
        state.frameIndex += 1

        // upstream `_reconstruct`: PixelShuffle ×16 residual + bicubic ×4 of the (padded) LR frame
        let px = residualProjection(seq(spatialHead, x))                                   // (B, n, 3·16·16)
        let r = c.pixelShuffle
        let residual = px.reshaped([b, th, tw, 3, r, r]).transposed(0, 1, 4, 2, 5, 3).reshaped([b, th * r, tw * r, 3])
        let bicubic = RelayVSRResample.upsample4(lr, cubic: true)
        taps?["bicubic"] = bicubic
        let out = bicubic + residual
        return out[0..., 0 ..< (height * c.upscale), 0 ..< (width * c.upscale)]
    }

    public func decodeStep(_ lrFrame: MLXArray, state: inout RelayVSRDecoderState) -> MLXArray {
        var t: RelayVSRTaps? = nil
        return decodeStep(lrFrame, state: &state, taps: &t)
    }
}
