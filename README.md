# mlx-realesrgan-swift

The MLXEngine **`imageUpscale`** package over [Real-ESRGAN](https://github.com/xocialize/realesrgan-mlx-swift) (SRVGGNetCompact) — 4× image super-resolution on Apple Silicon.

A transform capability of the visual optimization tier, chaining after `imageRestore` (and onto
T2I/decode output) before encode. All model logic — SRVGGNet, 64² tiling with feathered seam
blending — lives in the core; this package maps the canonical
`ImageUpscaleRequest → ImageUpscaleResponse` contract onto it. **No download:** all variants are
vendored in the core bundle (1–2 MB each, BSD-3).

## Variants

`.general` (default — the forge ADR-0008 shipped winner) · `.generalDenoise` (WDN) · `.anime`.
Native scale 4× (the response's `appliedScale` reports what ran).

## Usage

```swift
import MLXServeCore
import MLXRealESRGAN

let engine = MLXServeEngine()
try await engine.register(RealESRGANUpscalePackage.registration, configuration: RealESRGANConfiguration())

let resp = try await engine.run(ImageUpscaleRequest(image: smallImage)) as! ImageUpscaleResponse
// resp.image — 4× .png; resp.appliedScale == 4
```

Requirements: macOS 26+ (Apple Silicon, Metal GPU). Port MIT; weights BSD-3 (xinntao/Real-ESRGAN).
