import Foundation
import CoreGraphics
import CoreImage
import CoreVideo
import ImageIO
import UniformTypeIdentifiers
import MLX
import MLXToolKit
import RealESRGANMLX

/// Errors at the Real-ESRGAN package boundary.
public enum RealESRGANPackageError: Error, Equatable {
    case imageDecodeFailed(String)
    case imageEncodeFailed
}

/// An MLXEngine `imageUpscale` package over **Real-ESRGAN (SRVGGNetCompact)** — 4× image
/// super-resolution. A transform capability of the visual optimization tier, chaining after
/// `imageRestore` (and onto T2I/decode output) before encode.
///
/// A thin conformance wrapper over the standalone `RealESRGANMLX` core (realesrgan-mlx-swift);
/// all model logic (SRVGGNet, 64² tiling with feathered seams, NHWC) lives there. All variants
/// are vendored in the core bundle — `load()` involves **no download**.
///
/// Native scale is **4×**. A request's `scale` of `nil` or `≥ 4` runs at native 4×; a sub-native
/// `scale` (e.g. `2`) is honored by post-downsampling the native-4× result to `inputDim * scale`.
/// The response's `appliedScale` always reports what actually ran (per the contract, callers verify it).
@InferenceActor
public final class RealESRGANUpscalePackage: ModelPackage {
    public typealias Configuration = RealESRGANConfiguration

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            // Real-ESRGAN (xinntao) weights + architecture are BSD-3-Clause; port code MIT.
            license: LicenseDeclaration(weightLicense: .bsd3, portCodeLicense: .mit),
            provenance: Provenance(sourceRepo: "mlx-community/Real-ESRGAN-general-x4v3",
                                   revision: "main", tier: 1),
            requirements: RequirementsManifest(
                // Split footprint (engine 1.14). Weights are ~1-2 MB; the working set is the **tiled**
                // activations (64² tiles, feathered seams) + the 4× output buffer — so it's almost all
                // transient, and tile-bounded (it does NOT scale with full input area). Re-measured via
                // `realesrgan-smoke` through the real MLXServeEngine (see EFFICIENCY-ADOPTION.md):
                //   512²→2048² peak 2170 MB · 1024²→4096² peak 1807 MB (lower — tile-bounded, output-driven),
                //   floor ~3 MB in both → resident 32 MB / activation 2.2 GB (the 512² envelope worst case).
                // residentBytes = weights floor (+ overhead); peakActivationBytes = tiled peak − floor. The
                // engine reserves ONE shared transient across residents — the co-residency win for the
                // optimizer chain. `QuantConfigured` (RealESRGANConfiguration) charges the fp32 footprint.
                footprints: [QuantFootprint(quant: .fp32, residentBytes: 32_000_000, peakActivationBytes: 2_200_000_000)],
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
                chipFloor: nil
            ),
            specialties: [],
            surfaces: [
                ImageUpscaleContract.descriptor(
                    name: "realesrgan-upscale",
                    summary: "Real-ESRGAN (SRVGGNetCompact) 4x image super-resolution, tile-based."
                )
            ]
        )
    }

    private let configuration: Configuration
    private var upscaler: SRVGGNetCompact_Playback?

    public nonisolated init(configuration: Configuration) {
        self.configuration = configuration
    }

    public func load() async throws {
        guard upscaler == nil else { return }
        // Weights are vendored in the core bundle; construction validates their presence.
        // (The core lazy-loads tensors on first upscale.)
        upscaler = try SRVGGNetCompact_Playback(variant: configuration.variant.coreVariant)
    }

    public func unload() async {
        upscaler = nil
        MLX.Memory.clearCache()   // release the retained MLX pool so eviction frees RSS (not just drop refs)
    }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        guard let upscaler else { throw PackageError.notLoaded }
        guard request.capability == .imageUpscale,
              let req = request as? ImageUpscaleRequest else {
            throw PackageError.unsupportedCapability(request.capability)
        }
        try Task.checkCancellation()

        let inPB = try Self.decodeToPixelBuffer(req.image)
        let inW = CVPixelBufferGetWidth(inPB), inH = CVPixelBufferGetHeight(inPB)
        let native = upscaler.scaleFactor
        let nativePB = try await upscaler.upscale(inPB)

        // Honor a requested `scale` below the model's native factor by post-downsampling the
        // native-Nx result to `inputDim * scale` (BRIDGE-029). `nil`, the native factor, or any
        // request ≥ native pass through at native scale (the model can't exceed its native factor);
        // `appliedScale` always reports what actually ran, so callers can verify.
        let outPB: CVPixelBuffer
        let appliedScale: Int
        if let s = req.scale, s > 0, s < native {
            outPB = try Self.resizePixelBuffer(nativePB, toWidth: inW * s, height: inH * s)
            appliedScale = s
        } else {
            outPB = nativePB
            appliedScale = native
        }

        let w = CVPixelBufferGetWidth(outPB), h = CVPixelBufferGetHeight(outPB)
        // Output mirrors the input format: rawBGRA8 in ⇒ rawBGRA8 out (no re-encode); else .png.
        let outImage: Image
        if req.image.format == .rawBGRA8 {
            guard let raw = Self.encodeRawBGRA8(outPB) else { throw RealESRGANPackageError.imageEncodeFailed }
            outImage = raw
        } else {
            guard let png = Self.encodePNG(outPB) else { throw RealESRGANPackageError.imageEncodeFailed }
            outImage = Image(format: .png, data: png, width: w, height: h)
        }
        return ImageUpscaleResponse(image: outImage, appliedScale: appliedScale)
    }

    // MARK: - Image codec

    /// Decode a canonical `Image` (.png/.jpeg/.rawBGRA8) to a BGRA `CVPixelBuffer`.
    nonisolated static func decodeToPixelBuffer(_ image: Image) throws -> CVPixelBuffer {
        if image.format == .rawBGRA8 { return try rawBGRA8ToPixelBuffer(image) }
        guard let source = CGImageSourceCreateWithData(image.data as CFData, nil),
              let cg = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw RealESRGANPackageError.imageDecodeFailed("unreadable \(image.format.rawValue) data")
        }
        let w = cg.width, h = cg.height
        var pb: CVPixelBuffer?
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: w,
            kCVPixelBufferHeightKey as String: h,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:],
        ]
        guard CVPixelBufferCreate(nil, w, h, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb) == kCVReturnSuccess,
              let buffer = pb else {
            throw RealESRGANPackageError.imageDecodeFailed("pixel buffer allocation (\(w)x\(h))")
        }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let base = CVPixelBufferGetBaseAddress(buffer),
              let ctx = CGContext(
                data: base, width: w, height: h, bitsPerComponent: 8,
                bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                space: CGColorSpace(name: CGColorSpace.sRGB)!,
                bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
                    | CGBitmapInfo.byteOrder32Little.rawValue) else {
            throw RealESRGANPackageError.imageDecodeFailed("CGContext for BGRA draw")
        }
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
        return buffer
    }

    /// Encode a BGRA `CVPixelBuffer` as PNG bytes.
    nonisolated static func encodePNG(_ pb: CVPixelBuffer) -> Data? {
        let ci = CIImage(cvPixelBuffer: pb)
        let ctx = CIContext(options: [.cacheIntermediates: false])
        guard let cg = ctx.createCGImage(ci, from: ci.extent) else { return nil }
        let out = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(out, UTType.png.identifier as CFString, 1, nil) else { return nil }
        CGImageDestinationAddImage(dest, cg, nil)
        return CGImageDestinationFinalize(dest) ? out as Data : nil
    }

    /// Wrap raw interleaved BGRA8 bytes straight into a 32BGRA `CVPixelBuffer` — no `CGImageSource`,
    /// no decode. `width`/`height` are required for `.rawBGRA8`; `bytesPerRow` is the source row stride
    /// (defaults to tightly packed `width * 4`).
    nonisolated static func rawBGRA8ToPixelBuffer(_ image: Image) throws -> CVPixelBuffer {
        guard let w = image.width, let h = image.height, w > 0, h > 0 else {
            throw RealESRGANPackageError.imageDecodeFailed("rawBGRA8 requires width/height")
        }
        let srcStride = image.bytesPerRow ?? (w * 4)
        guard srcStride >= w * 4, image.data.count >= srcStride * h else {
            throw RealESRGANPackageError.imageDecodeFailed(
                "rawBGRA8 data too small (\(image.data.count) < \(srcStride * h))")
        }
        var pb: CVPixelBuffer?
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: w,
            kCVPixelBufferHeightKey as String: h,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:],
        ]
        guard CVPixelBufferCreate(nil, w, h, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb) == kCVReturnSuccess,
              let buffer = pb else {
            throw RealESRGANPackageError.imageDecodeFailed("pixel buffer allocation (\(w)x\(h))")
        }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else {
            throw RealESRGANPackageError.imageDecodeFailed("pixel buffer base address")
        }
        let dstStride = CVPixelBufferGetBytesPerRow(buffer)
        let rowBytes = min(srcStride, dstStride)
        image.data.withUnsafeBytes { (src: UnsafeRawBufferPointer) in
            guard let srcBase = src.baseAddress else { return }
            for row in 0..<h {
                memcpy(base.advanced(by: row * dstStride), srcBase.advanced(by: row * srcStride), rowBytes)
            }
        }
        return buffer
    }

    /// High-quality downsample of a 32BGRA `CVPixelBuffer` to `w`×`h` (a new 32BGRA buffer).
    /// Used to honor a requested `scale` below the model's native factor (BRIDGE-029).
    nonisolated static func resizePixelBuffer(_ src: CVPixelBuffer, toWidth w: Int, height h: Int) throws -> CVPixelBuffer {
        guard w > 0, h > 0 else {
            throw RealESRGANPackageError.imageEncodeFailed
        }
        let ci = CIImage(cvPixelBuffer: src)
        let ctx = CIContext(options: [.cacheIntermediates: false])
        guard let cg = ctx.createCGImage(ci, from: ci.extent) else {
            throw RealESRGANPackageError.imageEncodeFailed
        }
        var pb: CVPixelBuffer?
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: w,
            kCVPixelBufferHeightKey as String: h,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:],
        ]
        guard CVPixelBufferCreate(nil, w, h, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb) == kCVReturnSuccess,
              let buffer = pb else {
            throw RealESRGANPackageError.imageEncodeFailed
        }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let base = CVPixelBufferGetBaseAddress(buffer),
              let outCtx = CGContext(
                data: base, width: w, height: h, bitsPerComponent: 8,
                bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                space: CGColorSpace(name: CGColorSpace.sRGB)!,
                bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
                    | CGBitmapInfo.byteOrder32Little.rawValue) else {
            throw RealESRGANPackageError.imageEncodeFailed
        }
        outCtx.interpolationQuality = .high
        outCtx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
        return buffer
    }

    /// Emit a 32BGRA `CVPixelBuffer` as tightly-packed raw BGRA8 `Image` bytes (no compression/clamp).
    nonisolated static func encodeRawBGRA8(_ pb: CVPixelBuffer) -> Image? {
        let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb)
        guard w > 0, h > 0 else { return nil }
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(pb) else { return nil }
        let srcStride = CVPixelBufferGetBytesPerRow(pb)
        let dstStride = w * 4
        var out = Data(count: dstStride * h)
        out.withUnsafeMutableBytes { (dst: UnsafeMutableRawBufferPointer) in
            guard let dstBase = dst.baseAddress else { return }
            for row in 0..<h {
                memcpy(dstBase.advanced(by: row * dstStride), base.advanced(by: row * srcStride), dstStride)
            }
        }
        return Image.rawBGRA8(data: out, width: w, height: h)
    }
}

extension RealESRGANUpscalePackage {
    /// The author one-liner the engine registers.
    public nonisolated static var registration: PackageRegistration {
        .of(RealESRGANUpscalePackage.self)
    }
}
