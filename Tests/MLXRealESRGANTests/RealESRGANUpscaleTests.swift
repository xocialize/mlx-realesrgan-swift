import Testing
import Foundation
import CoreGraphics
import CoreVideo
import ImageIO
import UniformTypeIdentifiers
import MLXToolKit
@testable import MLXRealESRGAN

/// Offline conformance checks — no Metal evaluation. Live upscaling is proven in the
/// `MLXEngine Testing` app (or via the core's xcodebuild test suite, 17 real-inference tests).
struct RealESRGANUpscaleTests {

    @Test func manifestIsImageUpscaleAndPermissive() {
        let m = RealESRGANUpscalePackage.manifest
        #expect(m.capabilities == [.imageUpscale])
        #expect(m.license.weightLicense == .bsd3)
        #expect(m.license.portCodeLicense == .mit)
        #expect(LicensePolicy.permissiveOnly.evaluate(m.license) == .admitted)
    }

    @Test func manifestRequirements() {
        let r = RealESRGANUpscalePackage.manifest.requirements
        #expect(r.requiredBackends.contains(.metalGPU))
        #expect(r.os.minMacOS == SemanticVersion(major: 26, minor: 0, patch: 0))
    }

    @Test func surfaceIsTheCanonicalUpscaleDescriptor() {
        let s = RealESRGANUpscalePackage.manifest.surfaces.first
        #expect(s?.capability == .imageUpscale)
        #expect(s?.parameters.first?.kind == .image)
        #expect(s?.parameters.contains { $0.name == "scale" && !$0.required } == true)
    }

    @Test func registrationConstructs() throws {
        let reg = RealESRGANUpscalePackage.registration
        #expect(reg.manifest.capabilities == [.imageUpscale])
        let pkg = try reg.makePackage(RealESRGANConfiguration())
        #expect(pkg is RealESRGANUpscalePackage)
    }

    @Test func variantsMapToCoreCheckpoints() {
        #expect(RealESRGANConfiguration().variant == .general)
        #expect(RealESRGANVariant.general.coreVariant.rawValue == "general")
        #expect(RealESRGANVariant.generalDenoise.coreVariant.rawValue == "generalWDN")
        #expect(RealESRGANVariant.anime.coreVariant.rawValue == "anime")
    }

    @Test func configurationCodableRoundTrips() throws {
        let c = RealESRGANConfiguration(variant: .anime)
        let back = try JSONDecoder().decode(RealESRGANConfiguration.self, from: JSONEncoder().encode(c))
        #expect(back.variant == .anime)
    }

    @Test func pngRoundTripsThroughPixelBuffer() throws {
        let png = try #require(Self.makePNG(width: 32, height: 32))
        let image = Image(format: .png, data: png, width: 32, height: 32)
        let pb = try RealESRGANUpscalePackage.decodeToPixelBuffer(image)
        #expect(CVPixelBufferGetWidth(pb) == 32)
        let back = try #require(RealESRGANUpscalePackage.encodePNG(pb))
        #expect(back.prefix(4) == Data([0x89, 0x50, 0x4E, 0x47]))
    }

    // MARK: - rawBGRA8 (contract 1.9.0)

    @Test func rawBGRA8RoundTripsBitIdentical() throws {
        let w = 8, h = 4
        let bytes = Data((0..<(w * h * 4)).map { UInt8($0 % 256) })
        let image = Image.rawBGRA8(data: bytes, width: w, height: h)
        let pb = try RealESRGANUpscalePackage.decodeToPixelBuffer(image)
        #expect(CVPixelBufferGetWidth(pb) == w && CVPixelBufferGetHeight(pb) == h)
        let back = try #require(RealESRGANUpscalePackage.encodeRawBGRA8(pb))
        #expect(back.format == .rawBGRA8)
        #expect(back.width == w && back.height == h && back.bytesPerRow == nil)
        #expect(back.data == bytes)
    }

    @Test func rawBGRA8HonorsSourceStride() throws {
        let w = 5, h = 3
        let stride = w * 4 + 16
        var src = Data(count: stride * h)
        for row in 0..<h {
            for col in 0..<(w * 4) { src[row * stride + col] = UInt8((row * 40 + col) % 256) }
        }
        let image = Image.rawBGRA8(data: src, width: w, height: h, bytesPerRow: stride)
        let pb = try RealESRGANUpscalePackage.decodeToPixelBuffer(image)
        let back = try #require(RealESRGANUpscalePackage.encodeRawBGRA8(pb))
        #expect(back.data.count == w * 4 * h)
        for row in 0..<h {
            let expected = src.subdata(in: (row * stride)..<(row * stride + w * 4))
            let got = back.data.subdata(in: (row * w * 4)..<((row + 1) * w * 4))
            #expect(got == expected)
        }
    }

    @Test func rawBGRA8MissingDimensionsThrows() {
        let image = Image(format: .rawBGRA8, data: Data(count: 16))
        #expect(throws: RealESRGANPackageError.self) {
            _ = try RealESRGANUpscalePackage.decodeToPixelBuffer(image)
        }
    }

    @Test func pngAndRawBGRA8DecodeToSamePixels() throws {
        let png = try #require(Self.makePNG(width: 16, height: 16))
        let viaPNG = try RealESRGANUpscalePackage.decodeToPixelBuffer(Image(format: .png, data: png, width: 16, height: 16))
        let raw = try #require(RealESRGANUpscalePackage.encodeRawBGRA8(viaPNG))
        let viaRaw = try RealESRGANUpscalePackage.decodeToPixelBuffer(raw)
        let reRaw = try #require(RealESRGANUpscalePackage.encodeRawBGRA8(viaRaw))
        #expect(reRaw.data == raw.data)
    }

    // MARK: - scale honoring (BRIDGE-029)

    /// `resizePixelBuffer` (the sub-native `scale` path) yields the requested dimensions as a valid
    /// 32BGRA buffer. The full `run()` honoring (native-4× → downsample to `inputDim * scale`) is
    /// proven live in the `MLXEngine Testing` app; this pins the offline-testable resize math.
    @Test func resizePixelBufferProducesRequestedDimensions() throws {
        // Stand in for a native-4× result: a 64×64 BGRA buffer (≡ 16×16 input upscaled 4×).
        let png = try #require(Self.makePNG(width: 64, height: 64))
        let nativePB = try RealESRGANUpscalePackage.decodeToPixelBuffer(
            Image(format: .png, data: png, width: 64, height: 64))
        // Requesting scale 2 on the 16×16 input ⇒ 32×32 (= 16 × 2), not the native 64×64 (= 16 × 4).
        let scaled = try RealESRGANUpscalePackage.resizePixelBuffer(nativePB, toWidth: 32, height: 32)
        #expect(CVPixelBufferGetWidth(scaled) == 32)
        #expect(CVPixelBufferGetHeight(scaled) == 32)
        #expect(CVPixelBufferGetPixelFormatType(scaled) == kCVPixelFormatType_32BGRA)
        // Encodes cleanly as raw BGRA8 at the downsampled size (tightly packed 32×32×4).
        let raw = try #require(RealESRGANUpscalePackage.encodeRawBGRA8(scaled))
        #expect(raw.width == 32 && raw.height == 32)
        #expect(raw.data.count == 32 * 32 * 4)
    }

    static func makePNG(width: Int, height: Int) -> Data? {
        guard let ctx = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8,
                                  bytesPerRow: 0, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        ctx.setFillColor(CGColor(gray: 0.6, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))
        guard let cg = ctx.makeImage() else { return nil }
        let out = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(out, UTType.png.identifier as CFString, 1, nil) else { return nil }
        CGImageDestinationAddImage(dest, cg, nil)
        return CGImageDestinationFinalize(dest) ? out as Data : nil
    }
}
