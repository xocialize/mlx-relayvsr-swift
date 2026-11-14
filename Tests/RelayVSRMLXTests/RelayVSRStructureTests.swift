// RelayVSRStructureTests — weight-free checks: the key contract (parameter paths == upstream's), the resampler taps,
// the two window masks, pixel (un)shuffle orders, the K/V ring's slot bookkeeping, and the interval schedule through
// a miniature random-weight pipeline (plumbing, not pixels).

import Foundation
import MLX
import MLXNN
import XCTest

@testable import RelayVSRMLX

final class RelayVSRStructureTests: XCTestCase {

    static func tinyPipeline() -> RelayVSRPipeline {
        var gc = RelayVSRGeneratorConfig()
        gc.dim = 64; gc.numHeads = 2; gc.numLayers = 2; gc.ffnDim = 128; gc.textDim = 16; gc.freqDim = 32
        gc.inChannels = 8; gc.outChannels = 8; gc.lqHidden1 = 16; gc.lqHidden2 = 32; gc.spatialWindow = [2, 2]
        var dc = RelayVSRDecoderConfig()
        dc.dim = 64; dc.numHeads = 1; dc.numKVGroups = 1; dc.depth = 2; dc.encoderChannels = [8, 8, 8]
        dc.hiddenDim = 64; dc.spatialHeadDim = 32; dc.conditionChannels = 8
        dc.videoWindow = [2, 2]; dc.startWindow = [4, 4]; dc.endWindow = [4, 4]; dc.conditionSelfWindow = [4, 4]
        let gen = RelayVSRGenerator(gc), dec = RelayVSRDecoder(dc)
        // random weights so nothing degenerates to exact zeros/NaN; norm scales stay 1
        for m in [gen as Module, dec as Module] {
            m.update(parameters: ModuleParameters.unflattened(m.parameters().flattened().map {
                ($0.0, $0.0.contains("norm") ? $0.1 : MLXRandom.normal($0.1.shape) * 0.05) }))
        }
        gen.prepare(prompt: MLXRandom.normal([1, 4, gc.textDim]))
        return RelayVSRPipeline(generator: gen, decoder: dec)
    }

    func testKeyContractMatchesTheReleasedCheckpoints() {
        // wan.safetensors: 834 tensors incl. `prompt_embeds`; flash.safetensors: 200
        let g = Set(RelayVSRGenerator().parameters().flattened().map(\.0))
        XCTAssertEqual(g.count, 833)
        for k in ["patch_embedding.weight", "condition_embedder.time_embedder.linear_1.weight",
                  "condition_embedder.time_proj.bias", "condition_embedder.text_embedder.linear_2.bias",
                  "blocks.0.attn1.to_out.0.weight", "blocks.29.attn2.norm_k.weight", "blocks.7.norm2.bias",
                  "blocks.3.ffn.net.0.proj.weight", "blocks.3.ffn.net.2.bias", "blocks.0.scale_shift_table",
                  "scale_shift_table", "proj_out.weight", "lq_proj.conv2.weight", "lq_proj.norm1.weight",
                  "lq_proj.out_proj.bias"] {
            XCTAssertTrue(g.contains(k), k)
        }
        let d = Set(RelayVSRDecoder().parameters().flattened().map(\.0))
        XCTAssertEqual(d.count, 200)
        for k in ["lr_encoder.layers.0.weight", "lr_encoder.layers.6.bias", "lr_encoder.norm.weight",
                  "condition_encoder.projection.weight", "video_type_embedding", "condition_type_embedding",
                  "backbone.blocks.9.attention.condition_value.weight", "backbone.blocks.0.attention.key_norm.weight",
                  "backbone.blocks.0.attention_norm.weight", "backbone.blocks.4.feed_forward.gate_up_proj.weight",
                  "backbone.blocks.4.feed_forward.down.weight", "spatial_head.0.bias", "residual_projection.weight",
                  "condition_rgb_projection.bias"] {
            XCTAssertTrue(d.contains(k), k)
        }
        XCTAssertFalse(d.contains("backbone.blocks.0.feed_forward.down.bias"), "SwiGLU has no biases")
    }

    func testResamplerTapsAreTorchs() {
        for cubic in [false, true] {
            for ph in RelayVSRResample.taps(cubic: cubic) {
                XCTAssertEqual(ph.map(\.1).reduce(0, +), 1, accuracy: 1e-6)
            }
        }
        // bilinear phase 0: source i − 0.375 → 0.375·x[i−1] + 0.625·x[i]
        let b0 = RelayVSRResample.taps(cubic: false)[0]
        XCTAssertEqual(b0.map(\.0), [-1, 0]); XCTAssertEqual(b0[0].1, 0.375, accuracy: 1e-7)
        // bicubic (A = −0.75) phase 2: t = 0.125 → taps i−1 … i+2
        let c2 = RelayVSRResample.taps(cubic: true)[2]
        XCTAssertEqual(c2.map(\.0), [-1, 0, 1, 2])
        for (got, want) in zip(c2.map(\.1), [-0.0717773, 0.9672852, 0.1147461, -0.0102539] as [Float]) {
            XCTAssertEqual(got, want, accuracy: 1e-6)
        }
    }

    func testUpsampleOfAConstantAndARamp() {
        Device.withDefaultDevice(Device(.cpu)) {
            let k = MLXArray.full([1, 5, 7, 3], values: MLXArray(Float(0.25)))
            for cubic in [false, true] {
                let u = RelayVSRResample.upsample4(k, cubic: cubic)
                XCTAssertEqual(u.shape, [1, 20, 28, 3])
                XCTAssertLessThan(abs(u - 0.25).max().item(Float.self), 1e-6)
            }
            // interior of a horizontal ramp: output x = 4i + p ↦ i + (p + 0.5)/4 − 0.5
            let ramp = broadcast(MLXArray(0 ..< Int32(8)).asType(.float32).reshaped([1, 1, 8, 1]), to: [1, 2, 8, 1])
            let lin = RelayVSRResample.upsample4(ramp, cubic: false)
            XCTAssertEqual(lin[0, 0, 4 * 3 + 2, 0].item(Float.self), 3.125, accuracy: 1e-5)
            // torch's bicubic kernel (A = −0.75) is not linear-precise: Σ wₖ·k = 0.1660 at t = 0.125
            let cub = RelayVSRResample.upsample4(ramp, cubic: true)
            XCTAssertEqual(cub[0, 0, 4 * 3 + 2, 0].item(Float.self), 3.1660156, accuracy: 1e-5)
        }
    }

    func testGeneratorWindowMaskIsCentredAndClamped() {
        Device.withDefaultDevice(Device(.cpu)) {
            var c = RelayVSRGeneratorConfig()
            c.dim = 64; c.numHeads = 2; c.numLayers = 1; c.ffnDim = 64; c.spatialWindow = [3, 2]
            let gen = RelayVSRGenerator(c)
            XCTAssertNil(gen.windowMask(h: 3, w: 2, slots: 2), "a window that covers the grid needs no mask")
            let m = gen.windowMask(h: 5, w: 2, slots: 2)!.asType(.int32)                 // rows window 3 of 5
            XCTAssertEqual(m.shape, [10, 20])
            func row(_ qy: Int) -> [Int32] { (0 ..< 5).map { m[qy * 2, $0 * 2].item(Int32.self) } }
            XCTAssertEqual(row(0), [1, 1, 1, 0, 0], "top = clamp(0 − 1, 0, 2) = 0")
            XCTAssertEqual(row(2), [0, 1, 1, 1, 0], "top = 1")
            XCTAssertEqual(row(4), [0, 0, 1, 1, 1], "top = clamp(3, 0, 2) = 2")
            XCTAssertEqual(m[0..., 0 ..< 10].asArray(Int32.self), m[0..., 10 ..< 20].asArray(Int32.self),
                           "every cached slot sees the same spatial window")
        }
    }

    func testDecoderMaskIsHalfOpenAndTruncated() {
        Device.withDefaultDevice(Device(.cpu)) {
            let dec = RelayVSRDecoder(Self.tinyPipeline().decoder.config)
            XCTAssertNil(dec.mask(h: 2, w: 2, windows: [[4, 4]]), "±2 covers a 2×2 grid")
            let m = dec.mask(h: 6, w: 1, windows: [[4, 4], [2, 2]])!.asType(.int32)       // planes: ±2, ±1
            XCTAssertEqual(m.shape, [6, 12])
            // −2 ≤ ky − qy < 2 for the first plane, −1 ≤ ky − qy < 1 for the second; no clamping at the borders
            XCTAssertEqual((0 ..< 6).map { m[3, $0].item(Int32.self) }, [0, 1, 1, 1, 1, 0])
            XCTAssertEqual((0 ..< 6).map { m[3, 6 + $0].item(Int32.self) }, [0, 0, 1, 1, 0, 0])
            XCTAssertEqual((0 ..< 6).map { m[0, $0].item(Int32.self) }, [1, 1, 0, 0, 0, 0], "truncated at the top")
        }
    }

    func testPixelUnshuffleOrder() {
        Device.withDefaultDevice(Device(.cpu)) {
            // LQ projector: channel = c·256 + ph·16 + pw
            let x = MLXArray(0 ..< Int32(32 * 32 * 3)).asType(.float32).reshaped([1, 32, 32, 3])
            let u = LQ4xProj.unshuffle16(x)
            XCTAssertEqual(u.shape, [1, 2, 2, 768])
            XCTAssertEqual(u[0, 1, 0, 2 * 256 + 5 * 16 + 7].item(Float.self), x[0, 16 + 5, 7, 2].item(Float.self))
        }
    }

    func testKVRingWritesSlotsInOrderAndWraps() {
        Device.withDefaultDevice(Device(.cpu)) {
            let st = RelayVSRGeneratorState(blocks: 1, window: 3)
            var lengths: [Int] = [], firsts: [Float] = []
            for f in 0 ..< 5 {
                let k = MLXArray.full([1, 2, 4], values: MLXArray(Float(f)))
                let (all, _) = st.update(block: 0, k: k, v: k)
                lengths.append(all.dim(1)); firsts.append(all[0, 0, 0].item(Float.self))
                st.frameIndex += 1
            }
            XCTAssertEqual(lengths, [2, 4, 6, 6, 6])
            XCTAssertEqual(firsts, [0, 0, 0, 3, 3], "keyframe 3 overwrites slot 0, keyframe 4 slot 1")
        }
    }

    func testScheduleEmitsEveryFrameOnceInOrder() {
        Device.withDefaultDevice(Device(.cpu)) {
            let pipe = Self.tinyPipeline()
            let frames = (0 ..< 11).map { _ in MLXRandom.uniform(0 ..< 255, [16, 24, 3]) }
            let stream = RelayVSRStream(pipeline: pipe, options: RelayVSROptions(windowSize: 4, seed: 1))
            var counts: [Int] = []
            var outs: [MLXArray] = []
            for f in frames { outs += stream.push(f); counts.append(outs.count) }
            // window 4: intervals [0…3], [3…6], [6…9], then [9, 10] at the flush
            XCTAssertEqual(counts, [0, 0, 0, 4, 4, 4, 7, 7, 7, 10, 10])
            outs += stream.finish()
            XCTAssertEqual(outs.count, 11)
            XCTAssertEqual(outs[0].shape, [64, 96, 3])
            XCTAssertEqual(pipe.stageTimes.keyframes, 5, "keyframes 0, 3, 6, 9 and the last frame")
            XCTAssertEqual(pipe.stageTimes.decodes, 11 + 3, "three overlap frames are decoded twice")
            XCTAssertTrue(stream.finish().isEmpty, "nothing left after the flush")
            // the sequence form is the same stream
            var again: [MLXArray] = []
            pipe.run(frames: frames, options: RelayVSROptions(windowSize: 4, seed: 1)) { i, o in
                XCTAssertEqual(i, again.count); again.append(o)
            }
            XCTAssertEqual(abs(stacked(again) - stacked(outs)).max().item(Float.self), 0)
        }
    }

    func testAOneFrameClipAndOddSizes() {
        Device.withDefaultDevice(Device(.cpu)) {
            let pipe = Self.tinyPipeline()
            var n = 0
            pipe.run(frames: [MLXRandom.uniform(0 ..< 255, [13, 21, 3])]) { _, o in
                n += 1
                XCTAssertEqual(o.shape, [52, 84, 3], "padded to 16×24 inside, cropped back to 4× the source")
            }
            XCTAssertEqual(n, 1)
        }
    }
}
