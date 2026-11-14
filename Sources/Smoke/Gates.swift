// Parity gates against `oracle/dump_goldens.py` — upstream RelayVSR executed VERBATIM on the CPU in fp32.
// The gates pin the CPU stream unless `--gpu`, so the comparison is fp32-vs-fp32 arithmetic, not a Metal-vs-CPU
// drift measurement. Layout: goldens are upstream's NCHW / NCDHW; the port is channels-last.

import Foundation
import MLX
import MLXNN
import RelayVSRMLX

struct Cmp {
    let name: String
    let maxAbs: Float, refMax: Float, rel: Float, psnr: Double
    var line: String {
        String(format: "  %-22@ maxAbs %.3e  ref|max| %.3e  rel %.3e  psnr %6.1f dB", name as NSString,
               maxAbs, refMax, rel, psnr)
    }
}

func compare(_ name: String, _ got: MLXArray, _ ref: MLXArray) -> Cmp {
    precondition(got.shape == ref.shape, "\(name): shape \(got.shape) vs golden \(ref.shape)")
    let g = got.asType(.float32), r = ref.asType(.float32)
    let d = abs(g - r)
    let maxAbs = d.max().item(Float.self)
    let refMax = abs(r).max().item(Float.self)
    let mse = (d * d).mean().item(Float.self)
    let peak = max(refMax, 1e-12)
    let psnr = mse == 0 ? Double.infinity : 10 * log10(Double(peak * peak) / Double(mse))
    return Cmp(name: name, maxAbs: maxAbs, refMax: refMax, rel: maxAbs / peak, psnr: psnr)
}

/// NCHW → NHWC
func cl(_ x: MLXArray) -> MLXArray { x.transposed(0, 2, 3, 1) }
/// one-frame NCDHW (1, C, 1, H, W) → NHWC
func cl1(_ x: MLXArray) -> MLXArray { cl(x[0..., 0..., 0]) }

struct GateReport {
    var failures: [String] = []
    mutating func add(_ c: Cmp, relMax: Float) {
        note(c.line + (c.rel > relMax ? "   ✗ (> \(relMax))" : ""))
        if c.rel > relMax { failures.append("\(c.name) rel \(c.rel) > \(relMax)") }
    }
}

/// S0 — key contract: both networks load with `verify: [.all]` (0 missing / 0 unused) and the tensor / parameter
/// counts equal the released checkpoints' (wan.safetensors minus `prompt_embeds`; flash.safetensors).
func gateS0(weights: URL) throws {
    let p = try RelayVSRPipeline.load(directory: weights, dtypes: .production)
    func count(_ m: Module) -> Int { m.parameters().flattened().reduce(0) { $0 + $1.1.size } }
    func keys(_ m: Module) -> Int { m.parameters().flattened().count }
    note("S0 key contract (verify .all): PASS")
    note("  generator  \(keys(p.generator)) tensors  \(count(p.generator)) params   (released 833 / 5,031,059,136)")
    note("  decoder    \(keys(p.decoder)) tensors  \(count(p.decoder)) params   (released 200 / 31,934,208)")
    guard keys(p.generator) == 833, count(p.generator) == 5_031_059_136,
          keys(p.decoder) == 200, count(p.decoder) == 31_934_208 else { throw SmokeError("S0 count mismatch") }
}

/// S1 — sub-op taps per component, each fed the GOLDEN inputs so a break localises to one component:
///   resamplers · generator constants (temb, timestep_proj, prompt projection, cross K/V) · LQ projector ·
///   five generator keyframes with INJECTED noise (blocks 0 / 1 / 29 and the prediction at keyframes 0 and 4 — the
///   fifth wraps the 4-slot K/V ring — and every reference latent) · decoder reference priming · decoder frames of
///   intervals 0 and 1 from the golden latents · and, with `--e2e`, the whole stream from (lr, noise).
/// `relMax` gates isolated components; `chainRelMax` gates anything behind the 30-block generator.
func gateS1(weights: URL, golden: URL, relMax: Float, chainRelMax: Float, e2e: Bool, gpu: Bool,
            dtypes: RelayVSRDTypes = .parity) throws {
    try Device.withDefaultDevice(gpu ? Device(.gpu) : Device(.cpu)) {
        let (g, meta) = try loadArraysAndMetadata(url: golden)
        func G(_ k: String) throws -> MLXArray {
            guard let a = g[k] else { throw SmokeError("golden lacks \(k)") }
            return a
        }
        note("golden: \(meta["upstream"] ?? "?")  \(meta["lr"] ?? "?") × \(meta["frames"] ?? "?") frames, "
             + "window \(meta["window_size"] ?? "?"), keyframes \(meta["keyframes"] ?? "?"), torch \(meta["torch"] ?? "?")")
        let window = Int(meta["window_size"] ?? "") ?? 3
        let pipe = try RelayVSRPipeline.load(directory: weights, dtypes: dtypes)
        note("compute dtypes: generator \(dtypes.generator), decoder \(dtypes.decoder)")
        let gen = pipe.generator, dec = pipe.decoder
        var rep = GateReport()
        let lr = try G("lr")
        let frames = (0 ..< lr.dim(0)).map { lr[$0] }
        let keyframeIndices = Array(stride(from: 0, to: frames.count, by: window - 1))

        note("resamplers")
        let lr0 = RelayVSRPipeline.prepareLR(frames[0])
        rep.add(compare("bicubic ×4", RelayVSRResample.upsample4(lr0, cubic: true), cl(try G("dec.w0f0.bicubic"))),
                relMax: relMax)

        note("generator constants")
        let c = gen.gateConstants
        rep.add(compare("temb", c.temb, try G("gen.temb")), relMax: relMax)
        rep.add(compare("timestep_proj", c.timestepProj, try G("gen.tproj").reshaped(c.timestepProj.shape)), relMax: relMax)
        rep.add(compare("prompt projection", c.context, try G("gen.ctx")), relMax: relMax)
        rep.add(compare("cross K (block 0)", c.crossK0, try G("gen.cross_k.0").reshaped(c.crossK0.shape)), relMax: relMax)
        rep.add(compare("cross V (block 0)", c.crossV0, try G("gen.cross_v.0").reshaped(c.crossV0.shape)), relMax: relMax)

        note("generator: \(keyframeIndices.count) keyframes, injected noise")
        let state = gen.makeState()
        var latents: [MLXArray] = []
        for (j, fi) in keyframeIndices.enumerated() {
            let up = RelayVSRResample.upsample4(RelayVSRPipeline.prepareLR(frames[fi]), cubic: false)
            let noise = cl1(try G("noise.\(j)"))
            var taps: RelayVSRTaps? = [:]
            let pred = gen.step(noise: noise, lq: up, state: state, taps: &taps, tapBlocks: [0, 1, 29])
            let z = noise - pred
            eval(z)
            latents.append(z)
            if g["gen.kf\(j).lq"] != nil {
                rep.add(compare("kf\(j) lq_proj", taps!["lq"]!, try G("gen.kf\(j).lq")), relMax: relMax)
                for b in [0, 1, 29] {
                    rep.add(compare("kf\(j) block \(b)", taps!["b\(b)"]!, try G("gen.kf\(j).b\(b)")),
                            relMax: b == 0 && j == 0 ? relMax : chainRelMax)
                }
                rep.add(compare("kf\(j) pred", pred, cl1(try G("gen.kf\(j).pred"))), relMax: chainRelMax)
            }
            rep.add(compare("latent \(j)", z, cl(try G("latent.\(j)"))), relMax: chainRelMax)
        }

        note("decoder: reference priming (golden latent 0)")
        var ct: RelayVSRTaps? = [:]
        _ = dec.prepareCondition(cl(try G("latent.0")), taps: &ct)
        rep.add(compare("cond tokens", ct!["tokens"]!, try G("dec.cond0.tokens")), relMax: relMax)
        for b in [0, 9] {
            rep.add(compare("cond b\(b) key", ct!["b\(b).key"]!, try G("dec.cond0.b\(b).key")), relMax: relMax)
            rep.add(compare("cond b\(b) value", ct!["b\(b).value"]!, try G("dec.cond0.b\(b).value")), relMax: relMax)
        }
        rep.add(compare("cond hidden", ct!["hidden"]!, try G("dec.cond0.hidden")), relMax: relMax)

        note("decoder: intervals 0 and 1 (golden latents)")
        for wi in 0 ..< 2 {
            let a = dec.prepareCondition(cl(try G("latent.\(wi)"))), b = dec.prepareCondition(cl(try G("latent.\(wi + 1)")))
            var st = dec.beginStream(start: a, end: b)
            for n in 0 ..< window {
                let fi = wi * (window - 1) + n
                var t: RelayVSRTaps? = [:]
                let out = dec.decodeStep(RelayVSRPipeline.prepareLR(frames[fi]), state: &st, taps: &t)
                let tag = "dec.w\(wi)f\(n)"
                rep.add(compare("w\(wi)f\(n) tokens", t!["tokens"]!, try G("\(tag).tokens")), relMax: relMax)
                rep.add(compare("w\(wi)f\(n) block 0", t!["b0"]!, try G("\(tag).b0")), relMax: relMax)
                rep.add(compare("w\(wi)f\(n) block 9", t!["b9"]!, try G("\(tag).b9")), relMax: relMax)
                // an interval's first frame after the first interval is the OVERLAP: decoded to rebuild the video
                // memory, never emitted — upstream's emitted frame `fi` came from the previous interval
                if wi == 0 || n > 0 {
                    rep.add(compare("w\(wi)f\(n) out", out[0], try G("out.\(fi)").transposed(1, 2, 0)), relMax: relMax)
                }
            }
        }

        if e2e {
            note("E2E: the whole stream from (lr, injected noise)")
            var opts = RelayVSROptions()
            opts.windowSize = window
            var hooks = RelayVSRHooks()
            hooks.noise = { j, _ in cl1(g["noise.\(j)"]!) }
            var emitted = 0
            var worst = Double.infinity
            try pipe.run(frames: frames, options: opts, hooks: hooks) { i, out in
                precondition(i == emitted, "frames out of order")
                emitted += 1
                let cmp = compare("e2e frame \(i)", out, try G("out.\(i)").transposed(1, 2, 0))
                worst = min(worst, cmp.psnr)
                rep.add(cmp, relMax: chainRelMax)
            }
            guard emitted == frames.count else { throw SmokeError("E2E emitted \(emitted) of \(frames.count) frames") }
            note(String(format: "  E2E worst frame PSNR %.1f dB (peak = the frame's own max)", worst))
        }
        note(rep.failures.isEmpty ? "S1: PASS" : "S1: FAIL\n  " + rep.failures.joined(separator: "\n  "))
        if !rep.failures.isEmpty { throw SmokeError("\(rep.failures.count) gate(s) failed") }
    }
}
