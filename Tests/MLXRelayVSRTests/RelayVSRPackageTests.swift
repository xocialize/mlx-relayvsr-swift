// RelayVSRPackageTests — the package gates, offline: MAT-1..5, the licence posture (C7/C8 Apache-2.0, admitted by a
// `.blocking` permissive-only host), FIT, CAN-1..3, the honest load refusal, and the driver's frame/PTS plumbing
// through the interval schedule (a miniature random-weight pipeline: plumbing, not pixels).

import AVFoundation
import CoreMedia
import CoreVideo
import Foundation
import MLX
import MLXNN
import MLXServeConformance
import MLXServeCore
import MLXToolKit
import XCTest

@testable import MLXRelayVSR
@testable import RelayVSRMLX

final class RelayVSRPackageTests: XCTestCase {

    // MARK: - MAT

    func testMATGatePerPrecision() throws {
        for p in RelayVSRPrecision.allCases {
            let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            for f in RelayVSRConfiguration(precision: p).fileNames { try Data([0]).write(to: dir.appendingPathComponent(f)) }
            defer { try? FileManager.default.removeItem(at: dir) }
            let report = MaterializationConformance.check(
                freshConfiguration: RelayVSRConfiguration(precision: p),
                satisfiedConfiguration: RelayVSRConfiguration(precision: p, weightsDirectory: dir))
            XCTAssertTrue(report.passed, "\(p):\n\(report.summary)")
        }
    }

    func testOneLaneServesBothPrecisions() {
        let bf16 = RelayVSRConfiguration(), fp32 = RelayVSRConfiguration(precision: .fp32)
        XCTAssertEqual(bf16.weightSources.map(\.repo), ["mlx-community/RelayVSR-bf16"])
        XCTAssertEqual(fp32.weightSources.map(\.repo), ["mlx-community/RelayVSR-bf16"])
        XCTAssertEqual(bf16.weightSources.first?.matching?.sorted(),
                       ["config.json", "decoder_fp32.safetensors", "generator_bf16.safetensors",
                        "prompt_bf16.safetensors"])
        XCTAssertFalse((bf16 as Any) is SelfMaterializing, "the engine materializes the lane from the Hub")
        XCTAssertEqual(bf16.missingWeightSources(storeRoot: nil).count, 1, "fresh machine: missing")
    }

    func testPartialWeightsDirectoryIsMissing() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        try Data([0]).write(to: dir.appendingPathComponent(RelayVSRComponent.decoder.fileName))
        XCTAssertEqual(RelayVSRConfiguration(weightsDirectory: dir).missingWeightSources(storeRoot: nil).count, 1)
    }

    func testConfigurationRoundTripsWithoutHostPaths() throws {
        let cfg = RelayVSRConfiguration(precision: .fp32, seed: 7, windowSize: 9,
                                        weightsDirectory: URL(fileURLWithPath: "/tmp/x"))
        let back = try JSONDecoder().decode(RelayVSRConfiguration.self, from: JSONEncoder().encode(cfg))
        XCTAssertEqual(back.precision, .fp32); XCTAssertEqual(back.seed, 7); XCTAssertEqual(back.windowSize, 9)
        XCTAssertNil(back.weightsDirectory)
        let dflt = try JSONDecoder().decode(RelayVSRConfiguration.self, from: Data("{}".utf8))
        XCTAssertEqual(dflt.windowSize, 16); XCTAssertEqual(dflt.precision, .bf16)
    }

    // MARK: - licence

    func testLicenceIsApacheOnBothLayers() {
        let lic = RelayVSRUpscalePackage.manifest.license
        XCTAssertEqual(lic.weightLicense, .apache2)
        XCTAssertEqual(lic.portCodeLicense, .apache2)
        XCTAssertTrue(LicensePolicy.permissiveOnly.admits(.apache2))
    }

    func testBlockingPermissiveHostRegisters() async throws {
        let engine = MLXServeEngine(policy: .permissiveOnly, licenseEnforcement: .blocking)
        _ = try await engine.register(RelayVSRUpscalePackage.registration, configuration: RelayVSRConfiguration())
    }

    func testDeclaredFootprintsCoverBothPrecisions() {
        let fp = RelayVSRUpscalePackage.manifest.requirements.footprints
        XCTAssertEqual(fp.first { $0.quant == .bf16 }?.activationScaling?.measuredCeiling, 1920 * 1088)
        XCTAssertEqual(fp.first { $0.quant == .fp32 }?.activationScaling?.measuredCeiling, 1280 * 768)
        XCTAssertEqual(Set(fp.map(\.quant)), [.bf16, .fp32])
        XCTAssertLessThan(RelayVSRUpscalePackage.requiredBytes(.bf16), RelayVSRUpscalePackage.requiredBytes(.fp32))
        // the declared line covers both measured points (14.33 GB at 1280×768, 16.80 GB at 1920×1088)
        let b = fp.first { $0.quant == .bf16 }!, s = b.activationScaling!
        func declared(_ px: Int) -> Double { Double(b.residentBytes) + Double(s.baseBytes) + Double(s.bytesPerUnit) * Double(px) }
        XCTAssertGreaterThan(declared(1280 * 768), 14.33e9)
        XCTAssertGreaterThan(declared(1920 * 1088), 16.80e9)
    }

    // MARK: - FIT

    func testFITGate() {
        let report = FootprintConformance.check(manifest: RelayVSRUpscalePackage.manifest,
                                                configuration: RelayVSRConfiguration())
        XCTAssertTrue(report.passed, report.summary)
    }

    func testWorkloadIsOutputPixelsFromTheContainer() throws {
        // a real QuickTime file written by AVFoundation (the fixture is generated, 2 frames 200×120)
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".mov")
        defer { try? FileManager.default.removeItem(at: url) }
        try Self.writeTinyMovie(to: url, width: 200, height: 120)
        let data = try Data(contentsOf: url)
        XCTAssertEqual(RelayVSRConfiguration.videoDimensions(data)?.0, 200)
        XCTAssertEqual(RelayVSRConfiguration.videoDimensions(data)?.1, 120)
        let cfg = RelayVSRConfiguration()
        // the model computes ×4 of the source padded to multiples of 8, at either requested scale
        XCTAssertEqual(cfg.workloadUnits(for: VideoUpscaleRequest(video: Video(format: .mov, data: data))), 800 * 480)
        XCTAssertEqual(cfg.workloadUnits(for: VideoUpscaleRequest(video: Video(format: .mov, data: data), scale: 2)),
                       800 * 480)
        XCTAssertNil(cfg.workloadUnits(for: VideoUpscaleRequest(video: Video(format: .mp4, data: Data([1, 2, 3])))))
    }

    static func writeTinyMovie(to url: URL, width: Int, height: Int) throws {
        let w = try AVAssetWriter(outputURL: url, fileType: .mov)
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264, AVVideoWidthKey: width, AVVideoHeightKey: height,
        ])
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: nil)
        w.add(input)
        guard w.startWriting() else { throw w.error ?? NSError(domain: "writer", code: 1) }
        w.startSession(atSourceTime: .zero)
        for i in 0 ..< 2 {
            let pb = try RelayVSRDriver.pixelBuffer(from: MLXArray.zeros([height, width, 3]) + 0.5)
            while !input.isReadyForMoreMediaData { usleep(1000) }
            adaptor.append(pb, withPresentationTime: CMTime(value: CMTimeValue(i), timescale: 24))
        }
        input.markAsFinished()
        let sem = DispatchSemaphore(value: 0)
        w.finishWriting { sem.signal() }
        sem.wait()
        guard w.status == .completed else { throw w.error ?? NSError(domain: "writer", code: 2) }
    }

    // MARK: - CAN

    func testCANPreCancelledRun() async {
        let report = await CancellationConformance.checkRun(
            package: RelayVSRUpscalePackage(configuration: RelayVSRConfiguration()),
            request: VideoUpscaleRequest(video: Video(format: .mp4, data: Data())))
        XCTAssertTrue(report.passed, report.summary)
    }

    func testCANCadenceDeclaration() {
        XCTAssertTrue(CancellationConformance.longRunImplied(by: RelayVSRUpscalePackage.manifest))
        let report = CancellationConformance.checkCadence(
            manifest: RelayVSRUpscalePackage.manifest,
            // once per decoded source frame in the NativeFrameStream transform closure; RunProgress(.upsample) at
            // the frame seam
            posture: .cadence([.init(phase: .upsample, unit: .frame, reportsRunProgress: true)]))
        XCTAssertTrue(report.passed, report.summary)
    }

    // MARK: - load refusal + driver plumbing

    @InferenceActor
    func testLoadRefusesInsteadOfDownloading() async {
        let pkg = RelayVSRUpscalePackage(configuration: RelayVSRConfiguration(
            weightsDirectory: FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)))
        do {
            try await pkg.load()
            XCTFail("load must refuse when the converted weights are absent")
        } catch let e as RelayVSRPackageError {
            if case .weightsNotProvisioned = e {} else { XCTFail("\(e)") }
        } catch { XCTFail("\(error)") }
    }

    static func tinyPipeline() -> RelayVSRPipeline {
        var gc = RelayVSRGeneratorConfig()
        gc.dim = 64; gc.numHeads = 2; gc.numLayers = 1; gc.ffnDim = 128; gc.textDim = 16; gc.freqDim = 32
        gc.inChannels = 8; gc.outChannels = 8; gc.lqHidden1 = 16; gc.lqHidden2 = 32
        var dc = RelayVSRDecoderConfig()
        dc.dim = 64; dc.numHeads = 1; dc.numKVGroups = 1; dc.depth = 1; dc.encoderChannels = [8, 8, 8]
        dc.hiddenDim = 64; dc.spatialHeadDim = 32; dc.conditionChannels = 8
        let gen = RelayVSRGenerator(gc), dec = RelayVSRDecoder(dc)
        for m in [gen as Module, dec as Module] {
            m.update(parameters: ModuleParameters.unflattened(m.parameters().flattened().map {
                ($0.0, $0.0.contains("norm") ? $0.1 : MLXRandom.normal($0.1.shape) * 0.05) }))
        }
        gen.prepare(prompt: MLXRandom.normal([1, 4, gc.textDim]))
        return RelayVSRPipeline(generator: gen, decoder: dec)
    }

    func testDriverEmitsEveryFrameInOrderWithItsOwnPTS() throws {
        try Device.withDefaultDevice(Device(.cpu)) {
            for scale in [4, 2] {
                let driver = RelayVSRDriver(pipeline: Self.tinyPipeline(),
                                            configuration: RelayVSRConfiguration(windowSize: 4), scale: scale)
                var out: [(CVPixelBuffer, CMTime)] = []
                var counts: [Int] = []
                for i in 0 ..< 9 {
                    let px = MLXRandom.uniform(0 ..< 1, [10, 14, 3])
                    out += try driver.ingest(try RelayVSRDriver.pixelBuffer(from: px),
                                             pts: CMTime(value: CMTimeValue(i * 1001), timescale: 24000))
                    counts.append(out.count)
                }
                XCTAssertEqual(counts, [0, 0, 0, 4, 4, 4, 7, 7, 7], "an interval is emitted when its far endpoint arrives")
                out += try driver.flush()
                XCTAssertEqual(out.count, 9, "every source frame gets one output")
                XCTAssertEqual(out.map { $0.1.value }, (0 ..< 9).map { CMTimeValue($0 * 1001) })
                XCTAssertEqual(CVPixelBufferGetWidth(out[0].0), 14 * scale)
                XCTAssertEqual(CVPixelBufferGetHeight(out[0].0), 10 * scale)
            }
        }
    }

    func testPixelBufferRoundTripIsExactOn8BitLevels() throws {
        try Device.withDefaultDevice(Device(.cpu)) {
            let levels = MLXRandom.randInt(0 ..< 256, [6, 9, 3]).asType(.float32)
            let back = try RelayVSRDriver.rgb255(from: RelayVSRDriver.pixelBuffer(from: levels / 255))
            XCTAssertEqual(abs(back - levels).max().item(Float.self), 0)
        }
    }

    func testBoxHalveAveragesTwoByTwo() {
        Device.withDefaultDevice(Device(.cpu)) {
            let x = MLXArray([Float]([0, 2, 4, 6, 8, 10, 12, 14]), [2, 4, 1])
            let x3 = concatenated([x, x, x], axis: -1)
            XCTAssertEqual(RelayVSRDriver.boxHalve(x3)[0..., 0..., 0].asArray(Float.self), [5, 9])
        }
    }
}
