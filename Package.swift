// swift-tools-version: 6.2
import PackageDescription

// mlx-realesrgan-swift — the MLXEngine `imageUpscale` package over Real-ESRGAN (SRVGGNetCompact).
// A transform capability of the visual optimization tier: 4× super-resolution, chaining after
// imageRestore. Thin conformance layer over the realesrgan-mlx-swift core (tile-based, vendored
// weights — no download). Module/product is `MLXRealESRGAN`.
let package = Package(
    name: "mlx-realesrgan-swift",
    platforms: [
        .macOS(.v26)
    ],
    products: [
        .library(name: "MLXRealESRGAN", targets: ["MLXRealESRGAN"]),
    ],
    dependencies: [
        .package(path: "../mlx-engine-swift"),
        .package(url: "https://github.com/xocialize/realesrgan-mlx-swift.git", from: "0.1.0"),
    ],
    targets: [
        .target(
            name: "MLXRealESRGAN",
            dependencies: [
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "RealESRGANMLX", package: "realesrgan-mlx-swift"),
            ],
            // The core's playback tier (MLX) isn't Sendable-audited; the engine serializes
            // lifecycle on InferenceActor, so v5 mode keeps region-isolation a warning.
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .testTarget(
            name: "MLXRealESRGANTests",
            dependencies: [
                "MLXRealESRGAN",
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "MLXServeCore", package: "mlx-engine-swift"),
            ]
        ),
    ]
)
