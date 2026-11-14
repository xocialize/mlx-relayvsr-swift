// RelayVSR's reference generator — MLX-Swift port of kopperx/RelayVSR @ e856773
// `refvsr_inference/models/wan/{transformer,lq_proj,sampling,flow}.py` (Apache-2.0; the Wan implementation keeps its
// upstream notice): the Wan2.2-TI2V-5B DiT (30 blocks, dim 3072, 24 heads × 128, ffn 14336, 48 latent channels,
// patch 1×2×2) with the rank-512 LoRA merged, GAN-distilled to ONE step at t = 1000, run as a STREAMING model on
// one keyframe per call. It never sees a VAE: the output latent goes straight to the decoder network.
//
// Isomorphic to upstream (diffusers naming): `patch_embedding`, `condition_embedder.{time_embedder,time_proj,
// text_embedder}`, `blocks.N.{attn1,attn2}.{to_q,to_k,to_v,to_out.0,norm_q,norm_k}`, `blocks.N.{norm2,ffn.net.0.proj,
// ffn.net.2,scale_shift_table}`, `scale_shift_table`, `proj_out`, `lq_proj.{conv1,norm1,conv2,norm2,out_proj}`.
// Substitutions only:
//   • NCDHW → NDHWC; tokens row-major as upstream's `flatten(2).transpose(1, 2)`.
//   • Local spatial attention: upstream's CUDA path is a Triton kernel and its portable path FlexAttention under a
//     block mask; here dense SDPA under the same `mask_mod` (measured ≤ 1/255 level apart in fp32 on the oracle).
//   • RoPE: (cos, sin) tables built in Double (upstream: float64 on CPU/CUDA), applied in fp32.
//   • The K/V ring is a list of per-slot arrays in slot order (upstream: a preallocated tensor written in place).

import Foundation
import MLX
import MLXNN

public struct RelayVSRGeneratorConfig: Sendable {
    public var dim = 3072, inChannels = 48, outChannels = 48, ffnDim = 14336, textDim = 4096, freqDim = 256
    public var numHeads = 24, numLayers = 30
    public var eps: Float = 1e-6
    public var lqHidden1 = 768, lqHidden2 = 3072
    /// Query-centred, edge-clamped spatial window in tokens (h, w); nil = global attention.
    public var spatialWindow: [Int]? = [22, 40]
    /// Keyframes whose K/V a new keyframe attends to, itself included.
    public var temporalContextFrames = 4
    /// `temporal_rope.training_frames`: temporal positions stay inside [0, trainingFrames).
    public var ropeTrainingFrames = 8
    public var ropeMaxSeqLen = 1024
    public var headDim: Int { dim / numHeads }
    public init() {}
}

// MARK: - RoPE (3-D factorised: temporal 44 | height 42 | width 42 of the 128-wide head)

/// Upstream `WanRotaryPosEmbed`: three `get_1d_rotary_pos_embed` tables, each normalised by its own sub-dimension.
struct WanRope {
    let cosT: [Float], sinT: [Float], cosH: [Float], sinH: [Float], cosW: [Float], sinW: [Float]
    let nT: Int, nH: Int, nW: Int   // complex pairs per axis: 22, 21, 21

    init(headDim d: Int, maxSeqLen: Int, theta: Double = 10000) {
        let dHW = 2 * (d / 6), dT = d - 2 * dHW
        func table(_ dim: Int) -> ([Float], [Float], Int) {
            let n = dim / 2
            var c = [Float](repeating: 0, count: maxSeqLen * n), s = c
            for p in 0 ..< maxSeqLen {
                for j in 0 ..< n {
                    let a = Double(p) / pow(theta, Double(2 * j) / Double(dim))
                    c[p * n + j] = Float(cos(a)); s[p * n + j] = Float(sin(a))
                }
            }
            return (c, s, n)
        }
        (cosT, sinT, nT) = table(dT)
        (cosH, sinH, nH) = table(dHW)
        (cosW, sinW, nW) = table(dHW)
    }

    /// Upstream `for_positions(pph, ppw, positions)`: (cos, sin) for one (h, w) grid per temporal position, frame-major
    /// then row-major: (frames·h·w, 1, 64).
    func grid(positions: [Int], h: Int, w: Int) -> (MLXArray, MLXArray) {
        let n = nT + nH + nW
        var c = [Float](repeating: 0, count: positions.count * h * w * n), s = c
        var i = 0
        for t in positions {
            for y in 0 ..< h {
                for x in 0 ..< w {
                    let o = i * n
                    for j in 0 ..< nT { c[o + j] = cosT[t * nT + j]; s[o + j] = sinT[t * nT + j] }
                    for j in 0 ..< nH { c[o + nT + j] = cosH[y * nH + j]; s[o + nT + j] = sinH[y * nH + j] }
                    for j in 0 ..< nW { c[o + nT + nH + j] = cosW[x * nW + j]; s[o + nT + nH + j] = sinW[x * nW + j] }
                    i += 1
                }
            }
        }
        return (MLXArray(c, [positions.count * h * w, 1, n]), MLXArray(s, [positions.count * h * w, 1, n]))
    }

    /// Upstream `_apply_wan_rotary`: adjacent channel pairs of every head rotated by the token's angles.
    /// x: (B, L, dim), computed in fp32 and returned in x's dtype.
    static func apply(_ x: MLXArray, heads: Int, _ cs: (MLXArray, MLXArray)) -> MLXArray {
        let (b, l, d) = (x.dim(0), x.dim(1), x.dim(2))
        let xf = x.asType(.float32).reshaped([b, l, heads, d / heads / 2, 2])
        let x1 = xf[.ellipsis, 0], x2 = xf[.ellipsis, 1]
        return stacked([x1 * cs.0 - x2 * cs.1, x1 * cs.1 + x2 * cs.0], axis: -1).reshaped([b, l, d]).asType(x.dtype)
    }
}

// MARK: - modules

/// Upstream `WanAttention` (+ `WanAttnProcessor`): q/k RMS-normalised ACROSS heads (over the full 3072).
final class WanAttention: Module {
    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "to_out") var toOut: [Linear]
    @ModuleInfo(key: "norm_q") var normQ: TorchRMSNorm
    @ModuleInfo(key: "norm_k") var normK: TorchRMSNorm
    let heads: Int

    init(dim: Int, heads: Int, eps: Float) {
        _toQ.wrappedValue = Linear(dim, dim)
        _toK.wrappedValue = Linear(dim, dim)
        _toV.wrappedValue = Linear(dim, dim)
        _toOut.wrappedValue = [Linear(dim, dim)]
        _normQ.wrappedValue = TorchRMSNorm(dim, eps: eps)
        _normK.wrappedValue = TorchRMSNorm(dim, eps: eps)
        self.heads = heads
    }

    /// (B, L, dim) → (B, heads, L, hd)
    func split(_ x: MLXArray) -> MLXArray {
        x.reshaped([x.dim(0), x.dim(1), heads, x.dim(2) / heads]).transposed(0, 2, 1, 3)
    }

    func attend(q: MLXArray, k: MLXArray, v: MLXArray, mask: MLXArray?) -> MLXArray {
        let (b, l, d) = (q.dim(0), q.dim(1), q.dim(2))
        let a = MLXFast.scaledDotProductAttention(queries: split(q), keys: split(k), values: split(v),
                                                  scale: 1 / Float(d / heads).squareRoot(), mask: mask)
        return toOut[0](a.transposed(0, 2, 1, 3).reshaped([b, l, d]).asType(q.dtype))
    }
}

/// diffusers `GELU(dim_in, dim_out, approximate="tanh")`: the feed-forward's first layer.
final class GELUProj: Module, UnaryLayer {
    let proj: Linear
    init(_ i: Int, _ o: Int) { proj = Linear(i, o) }
    func callAsFunction(_ x: MLXArray) -> MLXArray { geluApproximate(proj(x)) }
}

/// diffusers `FeedForward(dim, inner_dim, activation_fn="gelu-approximate")`: net = [GELU-proj, Dropout, Linear].
final class WanFeedForward: Module {
    @ModuleInfo var net: [UnaryLayer]
    init(dim: Int, inner: Int) { _net.wrappedValue = [GELUProj(dim, inner), Identity(), Linear(inner, dim)] }
    func callAsFunction(_ x: MLXArray) -> MLXArray { seq(net, x) }
}

final class WanBlock: Module {
    let attn1: WanAttention, attn2: WanAttention
    let norm1: LayerNorm, norm2: LayerNorm, norm3: LayerNorm
    let ffn: WanFeedForward
    @ParameterInfo(key: "scale_shift_table") var scaleShiftTable: MLXArray

    init(_ c: RelayVSRGeneratorConfig) {
        attn1 = WanAttention(dim: c.dim, heads: c.numHeads, eps: c.eps)
        attn2 = WanAttention(dim: c.dim, heads: c.numHeads, eps: c.eps)
        norm1 = LayerNorm(dimensions: c.dim, eps: c.eps, affine: false)
        norm2 = LayerNorm(dimensions: c.dim, eps: c.eps, affine: true)        // cross_attn_norm = true
        norm3 = LayerNorm(dimensions: c.dim, eps: c.eps, affine: false)
        ffn = WanFeedForward(dim: c.dim, inner: c.ffnDim)
        _scaleShiftTable.wrappedValue = MLXArray.zeros([1, 6, c.dim])
    }
}

final class TwoLinear: Module {
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear
    init(_ i: Int, _ o: Int) {
        _linear1.wrappedValue = Linear(i, o)
        _linear2.wrappedValue = Linear(o, o)
    }
}

/// Upstream `WanTimeTextImageEmbedding` (no image branch).
final class ConditionEmbedder: Module {
    @ModuleInfo(key: "time_embedder") var timeEmbedder: TwoLinear       // Linear, SiLU, Linear
    @ModuleInfo(key: "time_proj") var timeProj: Linear
    @ModuleInfo(key: "text_embedder") var textEmbedder: TwoLinear       // Linear, GELU(tanh), Linear
    init(_ c: RelayVSRGeneratorConfig) {
        _timeEmbedder.wrappedValue = TwoLinear(c.freqDim, c.dim)
        _timeProj.wrappedValue = Linear(c.dim, c.dim * 6)
        _textEmbedder.wrappedValue = TwoLinear(c.textDim, c.dim)
    }
}

/// Upstream `RMSNorm2d`: over channels, in fp32, affine, then back to the input dtype.
final class RMSNorm2d: Module, UnaryLayer {
    let weight: MLXArray
    let eps: Float
    init(_ c: Int, eps: Float = 1e-6) {
        weight = MLXArray.ones([c])
        self.eps = eps
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let xf = x.asType(.float32)
        return (xf * rsqrt((xf * xf).mean(axis: -1, keepDims: true) + eps) * weight.asType(.float32)).asType(x.dtype)
    }
}

/// Upstream `LQ4xProj`: the ×4-bilinear keyframe, pixel-unshuffled 16×16 (3 → 768 ch), through a 1×1 conv and a
/// stride-2 3×3 conv (RMS norm + SiLU each) and a Linear into the DiT width — one token per 32×32 output pixels,
/// ADDED to the patch tokens before block 0.
final class LQ4xProj: Module {
    let conv1: Conv2d, conv2: Conv2d
    let norm1: RMSNorm2d, norm2: RMSNorm2d
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(_ c: RelayVSRGeneratorConfig) {
        conv1 = Conv2d(inputChannels: 3 * 256, outputChannels: c.lqHidden1, kernelSize: 1)
        norm1 = RMSNorm2d(c.lqHidden1)
        conv2 = Conv2d(inputChannels: c.lqHidden1, outputChannels: c.lqHidden2, kernelSize: 3, stride: 2, padding: 1)
        norm2 = RMSNorm2d(c.lqHidden2)
        _outProj.wrappedValue = Linear(c.lqHidden2, c.dim)
    }

    /// 'b c (h 16) (w 16) -> b (c 16 16) h w' in channels-last: channel index c·256 + ph·16 + pw.
    static func unshuffle16(_ x: MLXArray) -> MLXArray {
        let (b, h, w, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        return x.reshaped([b, h / 16, 16, w / 16, 16, c]).transposed(0, 1, 3, 5, 2, 4)
            .reshaped([b, h / 16, w / 16, c * 256])
    }

    /// (B, H, W, 3) → (B, H/32 · W/32, dim)
    func callAsFunction(_ frame: MLXArray) -> MLXArray {
        var x = Self.unshuffle16(frame)
        x = silu(norm1(conv1(x)))
        x = silu(norm2(conv2(x)))
        return outProj(x.reshaped([x.dim(0), x.dim(1) * x.dim(2), x.dim(3)]))
    }
}

// MARK: - streaming state

/// Upstream `WanStreamingState`: per block a fixed-capacity ring of normalised, UNROTATED K and V, one slot per
/// keyframe (slot = keyframe index mod window), plus the keyframe counter.
public final class RelayVSRGeneratorState {
    var ringK: [[MLXArray]], ringV: [[MLXArray]]
    let window: Int
    public internal(set) var frameIndex = 0
    init(blocks: Int, window: Int) {
        ringK = Array(repeating: [], count: blocks)
        ringV = Array(repeating: [], count: blocks)
        self.window = window
    }
    /// Upstream `WanSelfAttentionKVCache.update`: write this keyframe's K/V into its slot, return the valid slots
    /// concatenated in SLOT order.
    func update(block i: Int, k: MLXArray, v: MLXArray) -> (MLXArray, MLXArray) {
        let slot = frameIndex % window
        if slot < ringK[i].count { ringK[i][slot] = k; ringV[i][slot] = v } else { ringK[i].append(k); ringV[i].append(v) }
        return (ringK[i].count == 1 ? k : concatenated(ringK[i], axis: 1),
                ringV[i].count == 1 ? v : concatenated(ringV[i], axis: 1))
    }
}

// MARK: - the network

public final class RelayVSRGenerator: Module {
    public let config: RelayVSRGeneratorConfig
    @ModuleInfo(key: "patch_embedding") var patchEmbedding: Conv3d
    @ModuleInfo(key: "condition_embedder") var conditionEmbedder: ConditionEmbedder
    let blocks: [WanBlock]
    @ParameterInfo(key: "scale_shift_table") var scaleShiftTable: MLXArray
    @ModuleInfo(key: "proj_out") var projOut: Linear
    @ModuleInfo(key: "lq_proj") var lqProj: LQ4xProj
    let normOut: LayerNorm
    let rope: WanRope

    /// Constants of a one-step model with a fixed prompt, set by `prepare(prompt:)` (upstream `prepare_streaming`):
    /// temb (1, 1, dim), timestep_proj (1, 6, dim), and every block's cross-attention K/V.
    var temb: MLXArray?
    var timestepProj: MLXArray?
    var context: MLXArray?
    var crossK: [MLXArray] = [], crossV: [MLXArray] = []
    private var maskCache: [String: MLXArray] = [:]

    public init(_ c: RelayVSRGeneratorConfig = .init()) {
        config = c
        _patchEmbedding.wrappedValue = Conv3d(inputChannels: c.inChannels, outputChannels: c.dim,
                                              kernelSize: [1, 2, 2], stride: [1, 2, 2])
        _conditionEmbedder.wrappedValue = ConditionEmbedder(c)
        blocks = (0 ..< c.numLayers).map { _ in WanBlock(c) }
        _scaleShiftTable.wrappedValue = MLXArray.zeros([1, 2, c.dim])
        _projOut.wrappedValue = Linear(c.dim, c.outChannels * 4)
        _lqProj.wrappedValue = LQ4xProj(c)
        normOut = LayerNorm(dimensions: c.dim, eps: c.eps, affine: false)
        rope = WanRope(headDim: c.headDim, maxSeqLen: c.ropeMaxSeqLen)
        super.init()
        train(false)
    }

    public var dtype: DType { projOut.weight.dtype }

    /// diffusers `Timesteps(num_channels, flip_sin_to_cos=True, downscale_freq_shift=0)` in Double:
    /// [cos(t·ω) ‖ sin(t·ω)], ω_j = 10000^(−j/half).
    static func sinusoidal(dim: Int, timestep: Double) -> MLXArray {
        let half = dim / 2
        var v = [Float](repeating: 0, count: dim)
        for j in 0 ..< half {
            let a = timestep * exp(-log(10000.0) * Double(j) / Double(half))
            v[j] = Float(cos(a)); v[half + j] = Float(sin(a))
        }
        return MLXArray(v, [1, dim])
    }

    /// Upstream `WanVSRPipeline.prepare_streaming`: the t = 1000 time constants, the projected prompt, and every
    /// block's cross-attention K/V, computed once. `prompt`: (1, 512, 4096) — the tensor shipped in the checkpoint.
    public func prepare(prompt: MLXArray) {
        let dt = dtype, e = conditionEmbedder
        let ts = Self.sinusoidal(dim: config.freqDim, timestep: 1000).asType(dt)
        let t = e.timeEmbedder.linear2(silu(e.timeEmbedder.linear1(ts)))                    // (1, dim)
        temb = t.reshaped([1, 1, config.dim])
        timestepProj = e.timeProj(silu(t)).reshaped([1, 6, config.dim])
        let ctx = e.textEmbedder.linear2(geluApproximate(e.textEmbedder.linear1(prompt.asType(dt))))
        context = ctx
        crossK = blocks.map { $0.attn2.normK($0.attn2.toK(ctx)) }
        crossV = blocks.map { $0.attn2.toV(ctx) }
        eval([temb!, timestepProj!, ctx] + crossK + crossV)
    }

    public func makeState() -> RelayVSRGeneratorState {
        RelayVSRGeneratorState(blocks: blocks.count, window: config.temporalContextFrames)
    }

    /// Upstream `SpatialAttentionSpec.mask_mod` for one query frame over `slots` cached key frames: key (ky, kx) is
    /// visible iff it lies in the wh × ww window centred on the query and CLAMPED inside the grid — top =
    /// clamp(qy − wh/2, 0, h − wh). Every cached slot is inside the temporal window. (h·w, slots·h·w) bool, or nil
    /// when the window covers the grid.
    func windowMask(h: Int, w: Int, slots: Int) -> MLXArray? {
        guard let win = config.spatialWindow, win[0] < h || win[1] < w else { return nil }
        let key = "\(h)x\(w)x\(slots)"
        if let m = maskCache[key] { return m }
        let (wh, ww) = (min(win[0], h), min(win[1], w))
        let plane = separableMask(
            h: h, w: w,
            rowVisible: { q, k in let top = max(0, min(q - wh / 2, h - wh)); return k >= top && k < top + wh },
            colVisible: { q, k in let left = max(0, min(q - ww / 2, w - ww)); return k >= left && k < left + ww })
        let m = slots == 1 ? plane : concatenated(Array(repeating: plane, count: slots), axis: 1)
        eval(m)
        maskCache[key] = m
        return m
    }

    /// One keyframe — upstream `WanTransformer3DModel.forward(..., streaming_state=…)` for a one-frame latent.
    ///   noise: (B, h, w, 48) — the sampling noise at t = 1000 (the model input)
    ///   lq:    (B, 16h, 16w, 3) in [−1, 1] — the keyframe, bilinear ×4
    /// Returns the model prediction (B, h, w, 48); the reference latent is `noise − pred` (one Euler step, σ 1 → 0).
    public func step(noise: MLXArray, lq: MLXArray, state: RelayVSRGeneratorState,
                     taps: inout RelayVSRTaps?, tapBlocks: Set<Int> = []) -> MLXArray {
        let c = config, dt = dtype
        precondition(temb != nil, "call prepare(prompt:) first")
        let pe = patchEmbedding(noise.asType(dt).expandedDimensions(axis: 1))             // (B, 1, ph, pw, dim)
        let (b, ph, pw) = (pe.dim(0), pe.dim(2), pe.dim(3)), n = ph * pw
        var x = pe.reshaped([b, n, c.dim])
        let lqTokens = lqProj(lq.asType(dt))
        taps?["lq"] = lqTokens
        x = x + lqTokens

        // physical slots are stable; RoPE follows the true keyframe occupying each slot (`fixed_end` policy)
        let frame = state.frameIndex, window = state.window
        let valid = min(frame + 1, window)
        let origin = max(0, frame - c.ropeTrainingFrames + 1)
        let frameIDs = (0 ..< valid).map { slot in frame - ((frame - slot) % window + window) % window }
        let qRope = rope.grid(positions: [frame - origin], h: ph, w: pw)
        let kRope = rope.grid(positions: frameIDs.map { $0 - origin }, h: ph, w: pw)
        let mask = windowMask(h: ph, w: pw, slots: valid)

        let tproj = timestepProj!.asType(.float32)                                           // (1, 6, dim)
        for (i, blk) in blocks.enumerated() {
            let mod = blk.scaleShiftTable.asType(.float32) + tproj
            func row(_ r: Int) -> MLXArray { mod[0..., r ..< (r + 1)] }                       // (1, 1, dim)
            // 1. self-attention over the ring of keyframes
            let nh = (blk.norm1(x.asType(.float32)) * (1 + row(1)) + row(0)).asType(dt)
            let a1 = blk.attn1
            let q = a1.normQ(a1.toQ(nh))
            let (k, v) = state.update(block: i, k: a1.normK(a1.toK(nh)), v: a1.toV(nh))
            let sa = a1.attend(q: WanRope.apply(q, heads: c.numHeads, qRope),
                               k: WanRope.apply(k, heads: c.numHeads, kRope), v: v, mask: mask)
            x = (x.asType(.float32) + sa * row(2)).asType(dt)
            // 2. cross-attention to the cached prompt
            let nc = blk.norm2(x.asType(.float32)).asType(dt)
            let a2 = blk.attn2
            x = x + a2.attend(q: a2.normQ(a2.toQ(nc)), k: crossK[i], v: crossV[i], mask: nil)
            // 3. feed-forward
            let nf = (blk.norm3(x.asType(.float32)) * (1 + row(4)) + row(3)).asType(dt)
            x = (x.asType(.float32) + blk.ffn(nf).asType(.float32) * row(5)).asType(dt)
            eval(x, state.ringK[i][frame % window], state.ringV[i][frame % window])       // one block per graph
            if tapBlocks.contains(i) { taps?["b\(i)"] = x }
        }

        // output norm, projection, unpatchify '(h w) (ph pw c) -> (h ph) (w pw) c'
        let om = scaleShiftTable + temb!                                                     // (1, 2, dim)
        let shift = om[0..., 0 ..< 1], scale = om[0..., 1 ..< 2]
        let y = projOut((normOut(x.asType(.float32)) * (1 + scale) + shift).asType(dt))
        let pred = y.reshaped([b, ph, pw, 2, 2, c.outChannels]).transposed(0, 1, 3, 2, 4, 5)
            .reshaped([b, ph * 2, pw * 2, c.outChannels])
        state.frameIndex += 1
        taps?["pred"] = pred
        return pred
    }
}

public extension RelayVSRGenerator {
    /// The constants `prepare(prompt:)` cached, for the parity gates.
    var gateConstants: (temb: MLXArray, timestepProj: MLXArray, context: MLXArray, crossK0: MLXArray, crossV0: MLXArray) {
        (temb!, timestepProj!, context!, crossK[0], crossV[0])
    }
}
