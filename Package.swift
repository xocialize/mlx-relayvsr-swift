// swift-tools-version: 6.2
import PackageDescription

// mlx-relayvsr-swift — RelayVSR (kopperx/RelayVSR, Apache-2.0 code + weights; arXiv 2609.37850) streaming ×4 video
// super-resolution on MLX: a one-step Wan2.2-TI2V-5B reference generator that runs on one frame in fifteen, and a
// 32 M-parameter dual-memory transformer that super-resolves every frame from the LR frame and the two endpoint
// reference latents. The faithful generative tier: it stays close to the source (fit for drawn content too).
//   • RelayVSRMLX — engine-agnostic core: both networks (isomorphic to upstream, upstream's parameter paths), the
//     keyframe / interval driver (`RelayVSRPipeline`), and the ×4 bilinear / bicubic resamplers torch uses
//   • MLXRelayVSR — the MLXEngine `videoUpscale` ModelPackage (frame-stream-native decode → stream → HEVC encode)
//   • relayvsr-smoke — S0 / S1 / E2E parity gates against upstream's CPU-fp32 goldens, the clip runner, the real-engine
//     drive, memory and time
let package = Package(
    name: "mlx-relayvsr-swift",
    platforms: [
        .macOS(.v26)
    ],
    products: [
        .library(name: "RelayVSRMLX", targets: ["RelayVSRMLX"]),
        .library(name: "MLXRelayVSR", targets: ["MLXRelayVSR"]),
        .executable(name: "relayvsr-smoke", targets: ["RelayVSRSmoke"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.31.2"),
        .package(url: "https://github.com/xocialize/mlx-engine-swift", from: "0.63.0"),
        .package(url: "https://github.com/xocialize/frame-stream-native.git", from: "0.4.0"),
    ],
    targets: [
        .target(
            name: "RelayVSRMLX",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
            ],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .target(
            name: "MLXRelayVSR",
            dependencies: [
                "RelayVSRMLX",
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "FrameStreamNative", package: "frame-stream-native"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .executableTarget(
            name: "RelayVSRSmoke",
            dependencies: [
                "RelayVSRMLX",
                "MLXRelayVSR",
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "MLXServeCore", package: "mlx-engine-swift"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ],
            path: "Sources/Smoke",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .testTarget(
            name: "RelayVSRMLXTests",
            dependencies: [
                "RelayVSRMLX",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ]
        ),
        .testTarget(
            name: "MLXRelayVSRTests",
            dependencies: [
                "MLXRelayVSR",
                "RelayVSRMLX",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "MLXServeCore", package: "mlx-engine-swift"),
                .product(name: "MLXServeConformance", package: "mlx-engine-swift"),
            ]
        ),
    ]
)
