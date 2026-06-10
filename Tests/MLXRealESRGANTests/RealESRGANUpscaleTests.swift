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
