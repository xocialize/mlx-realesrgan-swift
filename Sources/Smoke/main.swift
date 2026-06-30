import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import MLX
import MLXToolKit
import MLXServeCore
import MLXRealESRGAN

// realesrgan-smoke <image> <out.png> [general|generalDenoise|anime]
// Drives RealESRGANUpscalePackage through the REAL MLXServeEngine (register → run) — the full Stage-2
// path: license-gate admission (BSD3/MIT), C10 eligibility, engine-constructs-the-package (C13), the
// tiled 4× upscale forward. Reports timing + the split footprint (resident floor / tiled activation
// peak) per the memory harness, plus output luminance stats so a uniform (silent) output is caught.

enum SmokeError: Error { case usage, badImage, badResponse }

func encodePNG(_ cg: CGImage) throws -> Data {
    let data = NSMutableData()
    guard let dest = CGImageDestinationCreateWithData(data, UTType.png.identifier as CFString, 1, nil)
    else { throw SmokeError.badImage }
    CGImageDestinationAddImage(dest, cg, nil)
    guard CGImageDestinationFinalize(dest) else { throw SmokeError.badImage }
    return data as Data
}

/// Decode a PNG and report luminance mean/min/max — a near-uniform result is the classic
/// "looks fine, is wrong" silent failure.
func rgbStats(_ png: Data) -> (mean: Double, min: Double, max: Double) {
    guard let src = CGImageSourceCreateWithData(png as CFData, nil),
          let cg = CGImageSourceCreateImageAtIndex(src, 0, nil) else { return (0, 0, 0) }
    let w = cg.width, h = cg.height
    var buf = [UInt8](repeating: 0, count: w * h)
    let cs = CGColorSpaceCreateDeviceGray()
    guard let ctx = CGContext(data: &buf, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w,
                              space: cs, bitmapInfo: CGImageAlphaInfo.none.rawValue) else { return (0, 0, 0) }
    ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
    var sum = 0.0, lo = 255, hi = 0
    for v in buf { sum += Double(v); lo = min(lo, Int(v)); hi = max(hi, Int(v)) }
    return (sum / Double(buf.count) / 255.0, Double(lo) / 255.0, Double(hi) / 255.0)
}

@main
struct Smoke {
    static func main() async {
        let a = CommandLine.arguments
        guard a.count >= 3 else {
            FileHandle.standardError.write(Data(
                "usage: realesrgan-smoke <image> <out.png> [general|generalDenoise|anime]\n".utf8))
            exit(2)
        }
        let imagePath = a[1], outPath = a[2]
        let variantArg = a.count > 3 ? a[3] : "general"
        let variant = RealESRGANVariant(rawValue: variantArg) ?? .general
        do {
            guard let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: imagePath) as CFURL, nil),
                  let cg = CGImageSourceCreateImageAtIndex(src, 0, nil) else { throw SmokeError.badImage }
            let image = Image(format: .png, data: try encodePNG(cg), width: cg.width, height: cg.height)

            let config = RealESRGANConfiguration(variant: variant)

            // Full engine path: register (license gate + C10 eligibility) → run (engine constructs/loads
            // the package, C13; vendored weights load lazily inside the first run).
            let engine = MLXServeEngine()
            await engine.useModelStore(ModelStore(root: nil))
            let t0 = Date()
            let id = try await engine.register(RealESRGANUpscalePackage.registration, configuration: config)
            let tReg = Date().timeIntervalSince(t0)

            let t1 = Date()
            let resp = try await engine.run(ImageUpscaleRequest(image: image), package: id)
            let tRun = Date().timeIntervalSince(t1)

            guard let up = resp as? ImageUpscaleResponse else { throw SmokeError.badResponse }
            let outImage = up.image
            try outImage.data.write(to: URL(fileURLWithPath: outPath))

            let s = rgbStats(outImage.data)
            let peakMB = Double(MLX.GPU.snapshot().peakMemory) / 1_048_576
            print(String(format: "OK %@ ×%d → %@ | %dx%d %@ | reg %.2fs run %.2fs | peakGPU %.0f MB | "
                + "luma mean %.3f [%.3f…%.3f]", variantArg, up.appliedScale, outPath, outImage.width ?? -1,
                outImage.height ?? -1, outImage.format.rawValue, tReg, tRun, peakMB, s.mean, s.min, s.max))

            // --- Memory report (this variant; one process = a clean peak). Methodology per the
            // memory-harness: peak active during the real forward + resident floor (active after freeing
            // activations) + the device's recommendedWorkingSet ceiling. Recommend = peak×1.2 + 256 MB.
            MLX.GPU.clearCache()
            let floorMB = Double(MLX.GPU.snapshot().activeMemory) / 1_048_576
            let workingSetMB = Double(MLX.GPU.deviceInfo().maxRecommendedWorkingSetSize) / 1_048_576
            let recommendMB = peakMB * 1.2 + 256
            print(String(format: "MEM %@ | floor %.0f MB · peak %.0f MB · activation %.0f MB · "
                + "recommend %.0f MB (×1.2+256) | workingSet %.0f MB · exceeds %@", variantArg, floorMB,
                peakMB, max(0, peakMB - floorMB), recommendMB, workingSetMB,
                recommendMB > workingSetMB ? "YES" : "no"))
            if s.max - s.min < 0.02 {
                FileHandle.standardError.write(Data(
                    "WARN: output near-uniform (mean \(s.mean)) — possible silent failure\n".utf8))
                exit(3)
            }
        } catch {
            FileHandle.standardError.write(Data("FAILED: \(error)\n".utf8))
            exit(1)
        }
    }
}
