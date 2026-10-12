import Testing
import Foundation
import CoreGraphics
import CoreVideo
import ImageIO
import UniformTypeIdentifiers
import MLXToolKit
import RealESRGANMLX
@testable import MLXRealESRGAN

/// The 16-bit boundary (contract 1.52.0, AB-A-0158): `.rawRGBA16` and deep PNG decode into the shared tile driver's
/// 16-bit buffer and encode back at 16 bits. Offline — CoreGraphics and CoreVideo only, no Metal.
struct DeepBoundaryTests {

    /// A 16-bit gradient: ~0.1 of an 8-bit step per pixel in red, so an 8-bit hop would collapse it.
    static func deepWords(_ w: Int, _ h: Int) -> [UInt16] {
        var words = [UInt16](repeating: 0, count: w * h * 4)
        for y in 0..<h {
            for x in 0..<w {
                let i = (y * w + x) * 4
                words[i] = UInt16(30_000 + x * 25); words[i + 1] = UInt16(10_000 + y * 7)
                words[i + 2] = UInt16((x * 977 + y * 131) % 65_536); words[i + 3] = UInt16.max
            }
        }
        return words
    }

    static func deepPNG(_ w: Int, _ h: Int) throws -> Data {
        var words = deepWords(w, h)
        let cg = try #require(words.withUnsafeMutableBytes { raw in
            CGContext(data: raw.baseAddress, width: w, height: h, bitsPerComponent: 16, bytesPerRow: w * 8,
                      space: CGColorSpace(name: CGColorSpace.sRGB)!,
                      bitmapInfo: RealESRGANUpscalePackage.rgba16BitmapInfo)?.makeImage()
        })
        let out = NSMutableData()
        let dest = try #require(CGImageDestinationCreateWithData(out, UTType.png.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(dest, cg, nil)
        #expect(CGImageDestinationFinalize(dest))
        return out as Data
    }

    @Test func surfaceDeclaresTheSixteenBitBoundary() {
        let surface = RealESRGANUpscalePackage.manifest.surfaces[0]
        #expect(surface.rawImageFormats == [.rawBGRA8, .rawRGBA16])
        #expect(surface.acceptsRawRGBA16)
    }

    @Test func rawRGBA16RoundTripsBitIdentical() throws {
        let (w, h) = (9, 5)
        let bytes = Self.deepWords(w, h).withUnsafeBytes { Data($0) }
        let pb = try RealESRGANUpscalePackage.decodeToPixelBuffer(.rawRGBA16(data: bytes, width: w, height: h))
        #expect(CVPixelBufferGetPixelFormatType(pb) == MLXTileProcessor.deepPixelFormat)
        let back = try #require(RealESRGANUpscalePackage.encodeRawRGBA16(pb))
        #expect(back.format == .rawRGBA16)
        #expect(back.width == w && back.height == h)
        #expect(back.data == bytes)
    }

    @Test func rawRGBA16HonorsSourceStride() throws {
        let (w, h) = (3, 2)
        let words = Self.deepWords(w, h)
        var strided = Data(count: 32 * h)      // 24 bytes of pixels + 8 of padding per row
        for row in 0..<h {
            let rowBytes = words[(row * w * 4)..<((row + 1) * w * 4)].withUnsafeBytes { Data($0) }
            strided.replaceSubrange((row * 32)..<(row * 32 + 24), with: rowBytes)
        }
        let pb = try RealESRGANUpscalePackage.decodeToPixelBuffer(.rawRGBA16(data: strided, width: w, height: h, bytesPerRow: 32))
        let back = try #require(RealESRGANUpscalePackage.encodeRawRGBA16(pb))
        #expect(back.data == words.withUnsafeBytes { Data($0) })
    }

    @Test func rawRGBA16WithoutDimensionsOrBytesThrows() {
        #expect(throws: RealESRGANPackageError.self) {
            _ = try RealESRGANUpscalePackage.decodeToPixelBuffer(Image(format: .rawRGBA16, data: Data(count: 64)))
        }
        #expect(throws: RealESRGANPackageError.self) {
            _ = try RealESRGANUpscalePackage.decodeToPixelBuffer(.rawRGBA16(data: Data(count: 8), width: 2, height: 2))
        }
    }

    /// A 16-bit PNG decodes at 16 bits (exactly, for an opaque sRGB source) and encodes back to a 16-bit PNG.
    @Test func deepPNGDecodesAndEncodesAtSixteenBits() throws {
        let (w, h) = (16, 8)
        let pb = try RealESRGANUpscalePackage.decodeToPixelBuffer(Image(format: .png, data: try Self.deepPNG(w, h), width: w, height: h))
        #expect(CVPixelBufferGetPixelFormatType(pb) == MLXTileProcessor.deepPixelFormat)
        let raw = try #require(RealESRGANUpscalePackage.encodeRawRGBA16(pb))
        #expect(raw.data == Self.deepWords(w, h).withUnsafeBytes { Data($0) })

        let png = try #require(RealESRGANUpscalePackage.encodePNG(pb))
        let source = try #require(CGImageSourceCreateWithData(png as CFData, nil))
        let cg = try #require(CGImageSourceCreateImageAtIndex(source, 0, nil))
        #expect(cg.bitsPerComponent == 16)
    }

    /// An 8-bit PNG keeps the 8-bit path: a 32BGRA buffer and an 8-bit PNG back.
    @Test func eightBitPNGKeepsTheEightBitPath() throws {
        let png = try #require(RealESRGANUpscaleTests.makePNG(width: 8, height: 8))
        let pb = try RealESRGANUpscalePackage.decodeToPixelBuffer(Image(format: .png, data: png, width: 8, height: 8))
        #expect(CVPixelBufferGetPixelFormatType(pb) == kCVPixelFormatType_32BGRA)
        let back = try #require(RealESRGANUpscalePackage.encodePNG(pb))
        let source = try #require(CGImageSourceCreateWithData(back as CFData, nil))
        let cg = try #require(CGImageSourceCreateImageAtIndex(source, 0, nil))
        #expect(cg.bitsPerComponent == 8)
    }

    /// The sub-native `scale` path resamples a deep buffer at 16 bits: the gentle red ramp keeps more than 256 levels.
    @Test func deepResizeStaysSixteenBit() throws {
        let (w, h) = (400, 8)
        let bytes = Self.deepWords(w, h).withUnsafeBytes { Data($0) }
        let pb = try RealESRGANUpscalePackage.decodeToPixelBuffer(.rawRGBA16(data: bytes, width: w, height: h))
        let scaled = try RealESRGANUpscalePackage.resizePixelBuffer(pb, toWidth: 300, height: 6)
        #expect(CVPixelBufferGetPixelFormatType(scaled) == MLXTileProcessor.deepPixelFormat)
        let raw = try #require(RealESRGANUpscalePackage.encodeRawRGBA16(scaled))
        let words = raw.data.withUnsafeBytes { Array($0.bindMemory(to: UInt16.self)) }
        let reds = Set(stride(from: 0, to: words.count, by: 4).map { words[$0] })
        #expect(reds.count > 256)
    }
}
