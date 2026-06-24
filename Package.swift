// swift-tools-version: 6.2
import PackageDescription

// mlx-realesrgan-swift — Real-ESRGAN 4× super-resolution for MLXEngine. ONE repo, TWO products:
//   • RealESRGANMLX — engine-agnostic Swift/MLX core (no MLXToolKit dep; usable standalone)
//   • MLXRealESRGAN — the MLXEngine `imageUpscale` ModelPackage over that core
// Consolidated 2026-06-18: the former standalone `realesrgan-mlx-swift` core was folded in here (and
// archived). Vendored weights — no download. Python ref: xocialize/realesrgan-mlx.
let package = Package(
    name: "mlx-realesrgan-swift",
    platforms: [
        .macOS(.v26)
    ],
    products: [
        .library(name: "RealESRGANMLX", targets: ["RealESRGANMLX"]),
        .library(name: "MLXRealESRGAN", targets: ["MLXRealESRGAN"]),
    ],
    dependencies: [
        .package(url: "https://github.com/xocialize/mlx-engine-swift", from: "0.10.0"),
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.30.0"),
    ],
    targets: [
        // Engine-agnostic core (folded in from realesrgan-mlx-swift) — NO MLXToolKit dep.
        .target(
            name: "RealESRGANMLX",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ],
            // Per-file .copy so the bundle layout is flat (forge ADR-0011).
            resources: [
                .copy("Resources/realesr_general_x4.safetensors"),
                .copy("Resources/realesr_general_wdn_x4.safetensors"),
                .copy("Resources/realesr_anime_x4.safetensors"),
            ]
        ),
        // MLXEngine `imageUpscale` wrapper over the local core.
        .target(
            name: "MLXRealESRGAN",
            dependencies: [
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                "RealESRGANMLX",
            ],
            // The core's playback tier (MLX) isn't Sendable-audited; the engine serializes
            // lifecycle on InferenceActor, so v5 mode keeps region-isolation a warning.
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .testTarget(
            name: "RealESRGANMLXTests",
            dependencies: [
                "RealESRGANMLX",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ]
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
