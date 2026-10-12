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
exactly as before. A package adopts the 16-bit boundary by decoding `.rawRGBA16` (and deep PNG) into a
`deepPixelFormat` buffer and declaring `ToolDescriptor.rawImageFormats`.

## 16 bits end to end (0.9.0)

This package's own `imageUpscale` converter adopts the boundary too (AB-T-0229). The surface declares
`ToolDescriptor.rawImageFormats = [.rawBGRA8, .rawRGBA16]` (MLXToolKit contract 1.52.0, mlx-engine-swift ≥ 0.67.0).
- A `.rawRGBA16` image comes back `.rawRGBA16`, and a PNG or TIFF deeper than 8 bits comes back as a 16-bit PNG.
- The network is fp32, so the output carries the model's full precision. The sub-native `scale` resample runs at 16
  bits too.
- 8-bit inputs take the 8-bit path unchanged.

Gates (2026-10-11, M5 Max):
- **8-bit byte identity:** Release builds on the GPU, against v0.8.1, 4 images × {whole-frame, tiled} × {png ×4,
  png ×2, rawBGRA8 ×4}: 24/24 outputs byte-identical.
- **16 bits agree with 8 bits:** an 8-bit image widened exactly to 16 bits rounds back to the 8-bit run's output
  except at rounding boundaries (≤ 0.002% of values, max 0.502 of a step).
- **G-6 direct gate** (the AB-R-0489 set): the 16-bit output's smooth-area flat-neighbour share is 0.000–0.128,
  at or below the 16-bit interpolation's 0.015–0.165 on every image, so no 8-bit staircase.

## Consuming it

Public + version-tagged on github.com/xocialize. Add by tagged URL:
`.package(url: "https://github.com/xocialize/mlx-realesrgan-swift", from: "0.1.0")`, then import `MLXRealESRGAN` (the conformant `imageUpscale` package). Builds standalone — its engine contract (`MLXToolKit`) and model-core dependencies are tagged-URL net deps, no local checkouts.

Requirements: macOS 26+ (Apple Silicon, Metal GPU). Port MIT; weights BSD-3 (xinntao/Real-ESRGAN).
