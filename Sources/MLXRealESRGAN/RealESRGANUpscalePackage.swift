import Foundation
import CoreGraphics
import CoreImage
import CoreVideo
import ImageIO
import UniformTypeIdentifiers
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
/// Native scale is **4×**: a request's `scale` of `nil` or `4` is honored; the response's
/// `appliedScale` always reports what ran (per the contract, callers verify it).
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
                // ~1-2 MB weights; the working set is the tiled activations + the 4× output.
                footprints: [QuantFootprint(quant: .fp32, residentBytes: 1_000_000_000)],
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
    }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        guard let upscaler else { throw PackageError.notLoaded }
        guard request.capability == .imageUpscale,
              let req = request as? ImageUpscaleRequest else {
            throw PackageError.unsupportedCapability(request.capability)
        }
        try Task.checkCancellation()

        let inPB = try Self.decodeToPixelBuffer(req.image)
        let outPB = try await upscaler.upscale(inPB)
        guard let png = Self.encodePNG(outPB) else {
            throw RealESRGANPackageError.imageEncodeFailed
        }
        return ImageUpscaleResponse(
            image: Image(format: .png, data: png,
                         width: CVPixelBufferGetWidth(outPB),
                         height: CVPixelBufferGetHeight(outPB)),
            appliedScale: upscaler.scaleFactor)
    }

    // MARK: - Image codec

    /// Decode a canonical `Image` (.png/.jpeg bytes) to a BGRA `CVPixelBuffer`.
    nonisolated static func decodeToPixelBuffer(_ image: Image) throws -> CVPixelBuffer {
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
}

extension RealESRGANUpscalePackage {
    /// The author one-liner the engine registers.
    public nonisolated static var registration: PackageRegistration {
        .of(RealESRGANUpscalePackage.self)
    }
}
