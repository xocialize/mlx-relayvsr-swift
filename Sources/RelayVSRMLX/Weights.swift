// Weight loading — the lane directory `oracle/convert_weights.py` writes: `generator_bf16.safetensors`,
// `decoder_fp32.safetensors`, `prompt_bf16.safetensors`, `config.json`. Keys are upstream's verbatim, conv weights
// channels-last. Each network loads with `verify: [.all]` (no missing, no unused key — the S0 contract) and is cast
// to the requested compute dtype.

import Foundation
import MLX
import MLXNN

public enum RelayVSRComponent: String, CaseIterable, Sendable {
    case generator, decoder, prompt
    /// The file each component ships as: its RELEASED precision (generator and prompt bf16, decoder fp32).
    public var fileName: String {
        switch self {
        case .generator: "generator_bf16.safetensors"
        case .decoder: "decoder_fp32.safetensors"
        case .prompt: "prompt_bf16.safetensors"
        }
    }
    public static var files: [String] { allCases.map(\.fileName) + ["config.json"] }
}

/// Per-network compute dtypes.
///   • `parity` — fp32 everywhere: the gate lane against upstream's CPU-fp32 goldens. (The generator is RELEASED in
///     bf16, so fp32 here is an exact upcast of the same numbers.)
///   • `production` — bf16 everywhere, upstream's own inference dtype (`--dtype bf16`).
public struct RelayVSRDTypes: Sendable {
    public var generator: DType, decoder: DType
    public init(generator: DType = .float32, decoder: DType = .float32) {
        self.generator = generator; self.decoder = decoder
    }
    public static let parity = RelayVSRDTypes()
    public static let production = RelayVSRDTypes(generator: .bfloat16, decoder: .bfloat16)
}

public enum RelayVSRLoadError: Error, CustomStringConvertible {
    case missing(String)
    case config(String)
    public var description: String {
        switch self {
        case .missing(let p): "RelayVSR weights missing: \(p)"
        case .config(let m): "RelayVSR config.json: \(m)"
        }
    }
}

public extension RelayVSRPipeline {
    /// The architecture numbers the lane's `config.json` carries (upstream's safetensors metadata, resolved).
    static func configs(directory: URL) throws -> (RelayVSRGeneratorConfig, RelayVSRDecoderConfig) {
        let url = directory.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: url),
              let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let g = (root["generator"] as? [String: Any])?["config"] as? [String: Any],
              let d = (root["decoder"] as? [String: Any])?["config"] as? [String: Any]
        else { throw RelayVSRLoadError.missing(url.path) }
        var gc = RelayVSRGeneratorConfig()
        func int(_ m: [String: Any], _ k: String) throws -> Int {
            guard let v = m[k] as? Int else { throw RelayVSRLoadError.config("no integer \(k)") }
            return v
        }
        gc.numHeads = try int(g, "num_attention_heads")
        gc.dim = try gc.numHeads * int(g, "attention_head_dim")
        gc.inChannels = try int(g, "in_channels"); gc.outChannels = try int(g, "out_channels")
        gc.ffnDim = try int(g, "ffn_dim"); gc.numLayers = try int(g, "num_layers")
        gc.textDim = try int(g, "text_dim"); gc.freqDim = try int(g, "freq_dim")
        gc.lqHidden1 = try int(g, "lq_proj_hidden_dim1"); gc.lqHidden2 = try int(g, "lq_proj_hidden_dim2")
        gc.ropeMaxSeqLen = try int(g, "rope_max_seq_len")
        gc.temporalContextFrames = try int(g, "temporal_context_frames")
        gc.spatialWindow = (g["spatial_attention_mode"] as? String) == "global" ? nil : g["spatial_window_size"] as? [Int]
        if let r = g["temporal_rope"] as? [String: Any], let tf = r["training_frames"] as? Int { gc.ropeTrainingFrames = tf }
        guard (g["patch_size"] as? [Int]) == [1, 2, 2], (g["qk_norm"] as? String) == "rms_norm_across_heads",
              (g["cross_attn_norm"] as? Bool) == true, (g["num_frames_per_block"] as? Int) == 1
        else { throw RelayVSRLoadError.config("generator: an architecture this port does not implement") }

        var dc = RelayVSRDecoderConfig()
        guard (d["variant"] as? String) == "S", (d["condition_mode"] as? String) == "endpoints",
              (d["temporal_rope_policy"] as? String) == "current_relative", (d["upscale_factor"] as? Int ?? 4) == 4
        else { throw RelayVSRLoadError.config("decoder: only variant S / endpoints / current_relative / ×4") }
        dc.conditionChannels = try int(d, "condition_channels")
        dc.windowSize = try int(d, "window_size")
        dc.spatialHeadDim = try int(d, "spatial_head_dim")
        if let e = d["encoder_channels"] as? [Int] { dc.encoderChannels = e }
        if let s = d["spatial_attention"] as? [String: Any] {
            guard (s["type"] as? String ?? "query_centered") == "query_centered",
                  (s["interval"] as? String ?? "half_open") == "half_open",
                  (s["boundary"] as? String ?? "truncated") == "truncated"
            else { throw RelayVSRLoadError.config("decoder: unknown spatial_attention flavour") }
            if let v = s["video_window_size"] as? [Int] { dc.videoWindow = v }
            if let v = s["start_condition_window_size"] as? [Int] { dc.startWindow = v }
            if let v = s["end_condition_window_size"] as? [Int] { dc.endWindow = v }
            if let v = s["condition_self_window_size"] as? [Int] { dc.conditionSelfWindow = v }
        }
        return (gc, dc)
    }

    /// Load a lane directory. Weights are read and cast on the CPU stream (a multi-GB read must never sit inside a
    /// Metal command buffer), then the prompt constants are prepared on the default device.
    static func load(directory: URL, dtypes: RelayVSRDTypes = .production) throws -> RelayVSRPipeline {
        let (gc, dc) = try configs(directory: directory)
        func arrays(_ c: RelayVSRComponent, _ dt: DType) throws -> [String: MLXArray] {
            let url = directory.appendingPathComponent(c.fileName)
            guard FileManager.default.fileExists(atPath: url.path) else { throw RelayVSRLoadError.missing(url.path) }
            return try Device.withDefaultDevice(Device(.cpu)) {
                let a = try loadArrays(url: url).mapValues { $0.asType(dt) }
                eval(Array(a.values))
                return a
            }
        }
        let gen = RelayVSRGenerator(gc)
        try gen.update(parameters: ModuleParameters.unflattened(arrays(.generator, dtypes.generator)), verify: [.all])
        let dec = RelayVSRDecoder(dc)
        try dec.update(parameters: ModuleParameters.unflattened(arrays(.decoder, dtypes.decoder)), verify: [.all])
        guard let prompt = try arrays(.prompt, dtypes.generator)["prompt_embeds"] else {
            throw RelayVSRLoadError.missing("prompt_embeds")
        }
        gen.prepare(prompt: prompt)
        return RelayVSRPipeline(generator: gen, decoder: dec)
    }
}
