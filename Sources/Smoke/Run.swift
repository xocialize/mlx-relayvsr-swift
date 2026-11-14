// `relayvsr-smoke run` — a frame directory through the port on the GPU: every frame in, every frame out at ×4.
import CoreGraphics
import Foundation
import ImageIO
import MLX
import RelayVSRMLX
import UniformTypeIdentifiers

// MARK: - images

/// A PNG/JPEG as (H, W, 3) uint8-valued float32 (0…255).
func loadRGB255(_ path: String) throws -> MLXArray {
    guard let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
          let cg = CGImageSourceCreateImageAtIndex(src, 0, nil) else { throw SmokeError("cannot read \(path)") }
    let (w, h) = (cg.width, cg.height)
    var px = [UInt8](repeating: 0, count: w * h * 4)
    guard let ctx = CGContext(data: &px, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                              space: CGColorSpace(name: CGColorSpace.sRGB)!,
                              bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { throw SmokeError("ctx") }
    ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
    return MLXArray(px, [h, w, 4])[0..., 0..., 0 ..< 3].asType(.float32)
}

/// (H, W, 3) float in [0, 1] → 8-bit sRGB PNG (round, clamp).
func writePNG(_ x: MLXArray, _ path: String) throws {
    let (h, w) = (x.dim(0), x.dim(1))
    let rgb = (clip(x.asType(.float32), min: 0, max: 1) * 255).round().asType(.uint8).reshaped([h * w * 3])
        .asArray(UInt8.self)
    var px = [UInt8](repeating: 255, count: w * h * 4)
    for i in 0 ..< w * h { px[4 * i] = rgb[3 * i]; px[4 * i + 1] = rgb[3 * i + 1]; px[4 * i + 2] = rgb[3 * i + 2] }
    let cg: CGImage? = px.withUnsafeMutableBytes { p in
        CGContext(data: p.baseAddress, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                  space: CGColorSpace(name: CGColorSpace.sRGB)!,
                  bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)?.makeImage()
    }
    guard let cg, let dest = CGImageDestinationCreateWithURL(URL(fileURLWithPath: path) as CFURL,
                                                            UTType.png.identifier as CFString, 1, nil)
    else { throw SmokeError("png write \(path)") }
    CGImageDestinationAddImage(dest, cg, nil)
    guard CGImageDestinationFinalize(dest) else { throw SmokeError("png finalize \(path)") }
}

// MARK: - runner

func runFrames(inDir: String, outDir: String, weights: URL, dtypes: RelayVSRDTypes, limit: Int?,
               options: RelayVSROptions, profile: Bool, cacheMB: Int?, repeats: Int) throws {
    if let cacheMB, cacheMB >= 0 { Memory.cacheLimit = cacheMB << 20 }
    var paths = try FileManager.default.contentsOfDirectory(atPath: inDir)
        .filter { $0.hasSuffix(".png") || $0.hasSuffix(".jpg") }.sorted().map { inDir + "/" + $0 }
    if let limit { paths = Array(paths.prefix(limit)) }
    guard !paths.isEmpty else { throw SmokeError("no frames in \(inDir)") }
    try FileManager.default.createDirectory(atPath: outDir, withIntermediateDirectories: true)
    let frames = try paths.map(loadRGB255)
    let (h, w) = (frames[0].dim(0), frames[0].dim(1))
    note("in \(frames.count) frames \(w)×\(h) → \(w * 4)×\(h * 4); window \(options.windowSize), "
         + "generator \(dtypes.generator), decoder \(dtypes.decoder)")
    let (pipe, loadS) = try timed { try RelayVSRPipeline.load(directory: weights, dtypes: dtypes) }
    pipe.profile = profile
    eval(frames)
    let phys0 = physFootprint()
    let gpuBefore = agxUtilization(), thermalBefore = ProcessInfo.processInfo.thermalState
    var produced: [MLXArray] = []
    var runS = 0.0, firstS = 0.0
    // The timed region is the pipeline alone (every frame is evaluated inside `run`); PNG encoding happens after it,
    // like the upstream torch runner's run_s.
    for rep in 0 ..< max(1, repeats) {
        produced.removeAll(keepingCapacity: true)
        let t0 = DispatchTime.now().uptimeNanoseconds
        (_, runS) = timed {
            pipe.run(frames: frames, options: options) { i, out in
                if i == 0 { firstS = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9 }
                produced.append(out)
            }
        }
        note(String(format: "rep %d: %d frames in %.2f s (%.0f ms/frame), first frame %.2f s, keyframes %d  "
                    + "(phys %.2f GB, MLX peak %.2f GB)", rep, produced.count, runS,
                    runS / Double(produced.count) * 1000, firstS, pipe.stageTimes.keyframes,
                    Double(physFootprint().current) / 1e9, Double(Memory.peakMemory) / 1e9))
    }
    var nanCount = 0
    for (i, fr) in produced.enumerated() {
        nanCount += isNaN(fr).asType(.int32).sum().item(Int.self)
        try writePNG((fr + 1) / 2, String(format: "%@/%@", outDir, (paths[i] as NSString).lastPathComponent))
    }
    let phys = physFootprint(), st = pipe.stageTimes
    let rec: [String: Any] = [
        "frames_in": paths.count, "frames_out": produced.count, "lr": "\(w)x\(h)", "target": "\(w * 4)x\(h * 4)",
        "dtypes": "generator \(dtypes.generator) decoder \(dtypes.decoder)", "window_size": options.windowSize,
        "load_s": loadS, "run_s": runS, "s_per_frame": runS / Double(produced.count), "first_frame_s": firstS,
        "keyframes": st.keyframes, "repeat": repeats,
        "stage_s": profile ? ["generator": st.generator, "condition": st.condition, "decoder": st.decoder] : [:],
        "peak_phys_gb": Double(phys.peak) / 1e9, "phys_before_run_gb": Double(phys0.current) / 1e9,
        "mlx_peak_gb": Double(Memory.peakMemory) / 1e9, "seed": options.seed, "cache_limit_mb": cacheMB ?? -1,
        "nan_values": nanCount,
        // timing validity (shared machine): the GPU must be idle before the run and the machine .nominal
        "gpu_util_before_pct": gpuBefore ?? -1, "thermal_before": "\(thermalBefore.rawValue)",
        "thermal_after": "\(ProcessInfo.processInfo.thermalState.rawValue)",
    ]
    let json = try JSONSerialization.data(withJSONObject: rec, options: [.prettyPrinted, .sortedKeys])
    try json.write(to: URL(fileURLWithPath: outDir + "/run.json"))
    note(String(data: json, encoding: .utf8)!)
}

/// (current, lifetime-peak) phys_footprint of this process, bytes — proc_pid_rusage(RUSAGE_INFO_V4).
func physFootprint() -> (current: UInt64, peak: UInt64) {
    var info = rusage_info_v4()
    let rc = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) { proc_pid_rusage(getpid(), RUSAGE_INFO_V4, $0) }
    }
    return rc == 0 ? (info.ri_phys_footprint, info.ri_lifetime_max_phys_footprint) : (0, 0)
}

func timed<R>(_ body: () throws -> R) rethrows -> (R, Double) {
    let t0 = DispatchTime.now().uptimeNanoseconds
    let r = try body()
    return (r, Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9)
}

/// AGX "Device Utilization %" (the unfiltered `ioreg -r -d 1 -c AGXAccelerator` — the `-k PerformanceStatistics`
/// form reads 0 under load, AB-L-0196).
func agxUtilization() -> Int? {
    let p = Process()
    p.executableURL = URL(fileURLWithPath: "/usr/sbin/ioreg")
    p.arguments = ["-r", "-d", "1", "-c", "AGXAccelerator"]
    let pipe = Pipe()
    p.standardOutput = pipe
    guard (try? p.run()) != nil else { return nil }
    let out = String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
    p.waitUntilExit()
    guard let r = out.range(of: #""Device Utilization %"=(\d+)"#, options: .regularExpression) else { return nil }
    return Int(out[r].split(separator: "=").last ?? "")
}
