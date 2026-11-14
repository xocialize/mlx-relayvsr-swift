// relayvsr-smoke — parity gates and runners for the RelayVSR MLX port.
//   relayvsr-smoke s0 <weightsDir>                                   key contract (verify .all) + parameter counts
//   relayvsr-smoke s1 <weightsDir> <golden.safetensors> [--rel 1e-4] [--chain-rel 1e-3] [--e2e] [--gpu [--tf32]]
//   relayvsr-smoke run <lrFramesDir> <outDir> <weightsDir> [--dtype bf16|fp32] [--dec-dtype bf16|fp32]
//                      [--window 16] [--limit N] [--seed S] [--profile] [--cache-mb N]
import Foundation
import MLX
import RelayVSRMLX

struct SmokeError: Error, CustomStringConvertible {
    let description: String
    init(_ d: String) { description = d }
}

func note(_ s: String) { FileHandle.standardError.write((s + "\n").data(using: .utf8)!) }

func flag(_ name: String, _ args: [String]) -> String? {
    guard let i = args.firstIndex(of: name), i + 1 < args.count else { return nil }
    return args[i + 1]
}

let args = Array(CommandLine.arguments.dropFirst())
// gate modes compare fp32 to fp32: keep M5's TF32 matmul off unless asked (must precede the first MLX op)
if args.first == "s1" && !args.contains("--tf32") { setenv("MLX_ENABLE_TF32", "0", 1) }
do {
    switch args.first {
    case "s0":
        try gateS0(weights: URL(fileURLWithPath: args[1]))
    case "s1":
        try gateS1(weights: URL(fileURLWithPath: args[1]), golden: URL(fileURLWithPath: args[2]),
                   relMax: flag("--rel", args).flatMap(Float.init) ?? 1e-4,
                   chainRelMax: flag("--chain-rel", args).flatMap(Float.init) ?? 1e-3,
                   e2e: args.contains("--e2e"), gpu: args.contains("--gpu"))
    case "run":
        func dt(_ f: String, _ d: DType) -> DType {
            switch flag(f, args) { case "bf16": .bfloat16; case "fp16": .float16; case "fp32": .float32; default: d }
        }
        let gen = dt("--dtype", .bfloat16)
        var opts = RelayVSROptions()
        if let w = flag("--window", args).flatMap(Int.init) { opts.windowSize = w }
        if let s = flag("--seed", args).flatMap(UInt64.init) { opts.seed = s }
        try runFrames(inDir: args[1], outDir: args[2], weights: URL(fileURLWithPath: args[3]),
                      dtypes: RelayVSRDTypes(generator: gen, decoder: dt("--dec-dtype", gen)),
                      limit: flag("--limit", args).flatMap(Int.init), options: opts,
                      profile: args.contains("--profile"), cacheMB: flag("--cache-mb", args).flatMap(Int.init),
                      repeats: flag("--repeat", args).flatMap(Int.init) ?? 1)
    default:
        note("usage: relayvsr-smoke s0|s1|run … (see Sources/Smoke/main.swift)")
        exit(2)
    }
} catch {
    note("FAILED: \(error)")
    exit(1)
}
