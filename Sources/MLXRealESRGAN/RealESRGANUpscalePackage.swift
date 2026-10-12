import Foundation
import Accelerate
import CoreGraphics
import CoreImage
import CoreVideo
import ImageIO
import UniformTypeIdentifiers
import MLX
import MLXProfiling
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
///
/// **16 bits end to end** (contract 1.52.0, AB-A-0158): the surface declares `.rawRGBA16`. A `.rawRGBA16` image, or a
/// PNG/TIFF deeper than 8 bits, runs through the core's 16-bit tile path and comes back at 16 bits: `.rawRGBA16` for
/// raw, a 16-bit PNG otherwise. 8-bit inputs take the 8-bit path byte for byte.
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
                // RE-BASELINED to in-app phys_footprint (MLXEngineImage, isolate+clearCache) — the
                // admission-relevant basis the engine/R-MEM-1 compares against. The `realesrgan-smoke`
                // MLX-peak (floor ~3 MB / act 2.2 GB) UNDER-read by ~2.7× (MLX buffer cache + process
                // overhead it omits). Clean in-app: floor 1.02 GB · peak 6.26 GB · activation 5.24 GB
                // (4× envelope). residentBytes = phys floor; peakActivationBytes = peak − floor. The engine
                // reserves ONE shared transient across residents — the co-residency win for the optimizer
                // chain. `QuantConfigured` (RealESRGANConfiguration) charges the fp32 footprint.
                footprints: [QuantFootprint(quant: .fp32, residentBytes: 1_020_000_000, peakActivationBytes: 5_240_000_000)],
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
                // Mirrors rawBGRA8 and keeps 16 bits for rawRGBA16 and deep PNG (0.9.0).
                .declaring(rawImageFormats: [.rawBGRA8, .rawRGBA16])
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
        // Forward the host's ceiling; `nil` keeps the core's permissive whole-frame-preferring default.
        upscaler = try SRVGGNetCompact_Playback(
            variant: configuration.variant.coreVariant,
            wholeFrameMaxPixels: configuration.wholeFrameMaxPixels ?? 1920 * 1080,
            inputTileSize: configuration.inputTileSize ?? 64,
            tileOverlap: configuration.tileOverlap ?? 8)
    }

    public func unload() async {
        upscaler = nil
        MLX.Memory.clearCache()   // release the retained MLX pool so eviction frees RSS (not just drop refs)
    }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        // CAN-1: the entry checkpoint is the FIRST act of run() — before notLoaded validation
        // (engine ≥ 0.27.0). Mid-run cadence: the core's tile driver checkpoints once per 64²
        // tile (MLXTileProcessor.process), rethrown unchanged through upscale().
        try Task.checkCancellation()
        guard let upscaler else { throw PackageError.notLoaded }
        guard request.capability == .imageUpscale,
              let req = request as? ImageUpscaleRequest else {
            throw PackageError.unsupportedCapability(request.capability)
        }

        let inPB = try Self.decodeToPixelBuffer(req.image)
        let inW = CVPixelBufferGetWidth(inPB), inH = CVPixelBufferGetHeight(inPB)
        let native = upscaler.scaleFactor

        // Inference is profiled (MLX_PROFILE=1). The spans live in the core's tile driver — one
        // per 64² tile (>1080p inputs) or one whole-frame span (≤1080p fast path); this layer just
        // brackets them with the run header + summary. No-op when MLX_PROFILE is unset.
        let prof = MLXProfiler.shared
        prof.beginRun("realesrgan imageUpscale scale=\(req.scale ?? native) \(inW)x\(inH)")
        let nativePB = try await upscaler.upscale(inPB)
        // ms/tile only when the core actually tiled; the whole-frame path records zero "tile" rows.
        let tileCount = prof.recorded.count(where: { $0.group == "upscale" && $0.label == "tile" })
        prof.endRun(denominators: ["tile": Double(tileCount)])

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
        // Output mirrors the input format: rawBGRA8 in ⇒ rawBGRA8 out, rawRGBA16 in ⇒ rawRGBA16 out (no re-encode);
        // else .png — 16-bit when the input was deep, because the buffer then is.
        let outImage: Image
        if req.image.format == .rawBGRA8 {
            guard let raw = Self.encodeRawBGRA8(outPB) else { throw RealESRGANPackageError.imageEncodeFailed }
            outImage = raw
        } else if req.image.format == .rawRGBA16 {
            guard let raw = Self.encodeRawRGBA16(outPB) else { throw RealESRGANPackageError.imageEncodeFailed }
            outImage = raw
        } else {
            guard let png = Self.encodePNG(outPB) else { throw RealESRGANPackageError.imageEncodeFailed }
            outImage = Image(format: .png, data: png, width: w, height: h)
        }
        return ImageUpscaleResponse(image: outImage, appliedScale: appliedScale)
    }

    // MARK: - Image codec

    /// Decode a canonical `Image` (.png/.jpeg/.rawBGRA8/.rawRGBA16) to a `CVPixelBuffer`: BGRA, or the tile driver's
    /// 16-bit `deepPixelFormat` for `.rawRGBA16` and for an encoded image deeper than 8 bits per component.
    nonisolated static func decodeToPixelBuffer(_ image: Image) throws -> CVPixelBuffer {
        if image.format == .rawBGRA8 { return try rawBGRA8ToPixelBuffer(image) }
        if image.format == .rawRGBA16 { return try rawRGBA16ToPixelBuffer(image) }
        guard let source = CGImageSourceCreateWithData(image.data as CFData, nil),
              let cg = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw RealESRGANPackageError.imageDecodeFailed("unreadable \(image.format.rawValue) data")
        }
        if cg.bitsPerComponent > 8 { return try deepPixelBuffer(drawing: cg) }
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

    /// Encode a BGRA `CVPixelBuffer` as PNG bytes (a 16-bit PNG for a `deepPixelFormat` buffer).
    nonisolated static func encodePNG(_ pb: CVPixelBuffer) -> Data? {
        if MLXTileProcessor.isDeep(pb) {
            guard let cg = deepCGImage(pb) else { return nil }
            let out = NSMutableData()
            guard let dest = CGImageDestinationCreateWithData(out, UTType.png.identifier as CFString, 1, nil) else { return nil }
            CGImageDestinationAddImage(dest, cg, nil)
            return CGImageDestinationFinalize(dest) ? out as Data : nil
        }
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

    /// Lanczos downsample of a `CVPixelBuffer` to `w`×`h`: a new buffer in the source's format, 32BGRA or the 16-bit
    /// `deepPixelFormat` (vImage; AB-T-0232, the AB-T-0225 fix). The CoreGraphics `.high` draw this replaces sampled on
    /// a corner-aligned grid, so a reduction drifted ±0.26 px from the centre to the edges, which cost RealPLKSR and
    /// HEART 1.7–2.8 dB at ×2 (AB-R-0502). vImage's Lanczos is centre-aligned and matches PIL's to 0.02 dB.
    nonisolated static func resizePixelBuffer(_ src: CVPixelBuffer, toWidth w: Int, height h: Int) throws -> CVPixelBuffer {
        guard w > 0, h > 0 else { throw RealESRGANPackageError.imageEncodeFailed }
        let deep = MLXTileProcessor.isDeep(src)
        let buffer: CVPixelBuffer
        if deep {
            buffer = try makeDeepPixelBuffer(w, h)
        } else {
            var pb: CVPixelBuffer?
            let attrs: [String: Any] = [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey as String: w,
                kCVPixelBufferHeightKey as String: h,
                kCVPixelBufferIOSurfacePropertiesKey as String: [:],
            ]
            guard CVPixelBufferCreate(nil, w, h, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb) == kCVReturnSuccess,
                  let made = pb else {
                throw RealESRGANPackageError.imageEncodeFailed
            }
            buffer = made
        }
        CVPixelBufferLockBaseAddress(src, .readOnly)
        CVPixelBufferLockBaseAddress(buffer, [])
        defer {
            CVPixelBufferUnlockBaseAddress(buffer, [])
            CVPixelBufferUnlockBaseAddress(src, .readOnly)
        }
        guard let srcBase = CVPixelBufferGetBaseAddress(src), let dstBase = CVPixelBufferGetBaseAddress(buffer) else {
            throw RealESRGANPackageError.imageEncodeFailed
        }
        var s = vImage_Buffer(data: srcBase, height: vImagePixelCount(CVPixelBufferGetHeight(src)),
                              width: vImagePixelCount(CVPixelBufferGetWidth(src)), rowBytes: CVPixelBufferGetBytesPerRow(src))
        var d = vImage_Buffer(data: dstBase, height: vImagePixelCount(h), width: vImagePixelCount(w),
                              rowBytes: CVPixelBufferGetBytesPerRow(buffer))
        // Four interleaved channels either way, each filtered alike, so BGRA and RGBA order both hold; premultiplied
        // alpha resamples correctly as is.
        let err = deep ? vImageScale_ARGB16U(&s, &d, nil, vImage_Flags(kvImageNoFlags))
                       : vImageScale_ARGB8888(&s, &d, nil, vImage_Flags(kvImageNoFlags))
        guard err == kvImageNoError else { throw RealESRGANPackageError.imageEncodeFailed }
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

    // MARK: - 16 bits (contract 1.52.0, AB-A-0158; the mlx-realplksr-swift 0.2.0 codec)

    /// RGBA, 16 bits per component, little-endian, premultiplied — `.rawRGBA16`'s layout as CoreGraphics draws it.
    nonisolated static let rgba16BitmapInfo =
        CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder16Little.rawValue

    nonisolated static func makeDeepPixelBuffer(_ w: Int, _ h: Int) throws -> CVPixelBuffer {
        var pb: CVPixelBuffer?
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: MLXTileProcessor.deepPixelFormat,
            kCVPixelBufferWidthKey as String: w,
            kCVPixelBufferHeightKey as String: h,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:],
        ]
        guard CVPixelBufferCreate(nil, w, h, MLXTileProcessor.deepPixelFormat, attrs as CFDictionary, &pb) == kCVReturnSuccess,
              let buffer = pb else {
            throw RealESRGANPackageError.imageDecodeFailed("16-bit pixel buffer allocation (\(w)x\(h))")
        }
        return buffer
    }

    /// Draw `cg` into a new `deepPixelFormat` buffer, in sRGB at 16 bits (its own size unless given).
    nonisolated static func deepPixelBuffer(drawing cg: CGImage, width: Int? = nil, height: Int? = nil,
                                            interpolation: CGInterpolationQuality = .default) throws -> CVPixelBuffer {
        let w = width ?? cg.width, h = height ?? cg.height
        let buffer = try makeDeepPixelBuffer(w, h)
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let base = CVPixelBufferGetBaseAddress(buffer),
              let ctx = CGContext(
                data: base, width: w, height: h, bitsPerComponent: 16,
                bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: rgba16BitmapInfo) else {
            throw RealESRGANPackageError.imageDecodeFailed("CGContext for RGBA16 draw")
        }
        ctx.interpolationQuality = interpolation
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
        return buffer
    }

    /// Wrap raw interleaved RGBA16 bytes straight into a `deepPixelFormat` buffer — no decode.
    nonisolated static func rawRGBA16ToPixelBuffer(_ image: Image) throws -> CVPixelBuffer {
        guard let w = image.width, let h = image.height, w > 0, h > 0 else {
            throw RealESRGANPackageError.imageDecodeFailed("rawRGBA16 requires width/height")
        }
        let srcStride = image.bytesPerRow ?? (w * 8)
        guard srcStride >= w * 8, image.data.count >= srcStride * h else {
            throw RealESRGANPackageError.imageDecodeFailed(
                "rawRGBA16 data too small (\(image.data.count) < \(srcStride * h))")
        }
        let buffer = try makeDeepPixelBuffer(w, h)
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else {
            throw RealESRGANPackageError.imageDecodeFailed("pixel buffer base address")
        }
        let dstStride = CVPixelBufferGetBytesPerRow(buffer)
        image.data.withUnsafeBytes { (src: UnsafeRawBufferPointer) in
            guard let srcBase = src.baseAddress else { return }
            for row in 0..<h {
                memcpy(base.advanced(by: row * dstStride), srcBase.advanced(by: row * srcStride), w * 8)
            }
        }
        return buffer
    }

    /// A `deepPixelFormat` buffer as a 16-bit sRGB `CGImage` over a copy of its pixels.
    nonisolated static func deepCGImage(_ pb: CVPixelBuffer) -> CGImage? {
        guard let raw = encodeRawRGBA16(pb), let provider = CGDataProvider(data: raw.data as CFData) else { return nil }
        return CGImage(width: CVPixelBufferGetWidth(pb), height: CVPixelBufferGetHeight(pb), bitsPerComponent: 16,
                       bitsPerPixel: 64, bytesPerRow: CVPixelBufferGetWidth(pb) * 8,
                       space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGBitmapInfo(rawValue: rgba16BitmapInfo),
                       provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)
    }

    /// Emit a `deepPixelFormat` buffer as tightly-packed `.rawRGBA16` bytes.
    nonisolated static func encodeRawRGBA16(_ pb: CVPixelBuffer) -> Image? {
        let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb)
        guard w > 0, h > 0, MLXTileProcessor.isDeep(pb) else { return nil }
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(pb) else { return nil }
        let srcStride = CVPixelBufferGetBytesPerRow(pb)
        let dstStride = w * 8
        var out = Data(count: dstStride * h)
        out.withUnsafeMutableBytes { (dst: UnsafeMutableRawBufferPointer) in
            guard let dstBase = dst.baseAddress else { return }
            for row in 0..<h {
                memcpy(dstBase.advanced(by: row * dstStride), base.advanced(by: row * srcStride), dstStride)
            }
        }
        return Image.rawRGBA16(data: out, width: w, height: h)
    }
}

extension RealESRGANUpscalePackage {
    /// The author one-liner the engine registers.
    public nonisolated static var registration: PackageRegistration {
        .of(RealESRGANUpscalePackage.self)
    }
}
