# mlx-realesrgan-swift

The MLXEngine **`imageUpscale`** package over [Real-ESRGAN](https://github.com/xocialize/realesrgan-mlx-swift) (SRVGGNetCompact) — 4× image super-resolution on Apple Silicon.

A transform capability of the visual optimization tier, chaining after `imageRestore` (and onto
T2I/decode output) before encode. All model logic — SRVGGNet, 128² tiling with feathered seam
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

## The shared tile driver at 16 bits

`RealESRGANMLX.MLXTileProcessor` also drives RealPLKSR, NERVE and HEART. Since 0.8.0 it reads and writes 16 bits
when it's handed a `kCVPixelFormatType_64RGBALE` buffer (`MLXTileProcessor.deepPixelFormat`). That's MLXToolKit's
`.rawRGBA16` (contract 1.52.0, AB-A-0158): RGBA, 16 bits per component, little-endian, 0…65535. The output has the
input's depth, and the seam composite is float32 either way. Every other input format takes the 8-bit BGRA path
exactly as before. This package's own `imageUpscale` converter is still 8-bit; a package adopts the 16-bit boundary
by decoding `.rawRGBA16` (and deep PNG) into a `deepPixelFormat` buffer and declaring
`ToolDescriptor.rawImageFormats`.

## Consuming it

Public + version-tagged on github.com/xocialize. Add by tagged URL:
`.package(url: "https://github.com/xocialize/mlx-realesrgan-swift", from: "0.1.0")`, then import `MLXRealESRGAN` (the conformant `imageUpscale` package). Builds standalone — its engine contract (`MLXToolKit`) and model-core dependencies are tagged-URL net deps, no local checkouts.

Requirements: macOS 26+ (Apple Silicon, Metal GPU). Port MIT; weights BSD-3 (xinntao/Real-ESRGAN).
