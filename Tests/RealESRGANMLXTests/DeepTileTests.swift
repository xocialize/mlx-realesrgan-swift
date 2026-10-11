//
//  DeepTileTests.swift
//
//  The 16-bit path through the shared tile driver (AB-A-0158, MLXToolKit contract 1.52.0
//  `.rawRGBA16`): a `kCVPixelFormatType_64RGBALE` input is read and written at 16 bits, and every
//  other input keeps the 8-bit path. Same method as probe A in TileCompositorFidelityTests — an
//  identity forward on the CPU device, no model, no Metal — at 16-bit resolution.
//

import CoreVideo
import Foundation
import MLX
import Testing
@testable import RealESRGANMLX

private func withCPU<R>(_ body: () throws -> R) rethrows -> R {
    try Device.withDefaultDevice(Device(.cpu), body)
}

/// A 64RGBALE buffer filled from `pixel(x, y) -> (r, g, b)`, opaque.
private func makeRGBA16(width: Int, height: Int,
                        _ pixel: (Int, Int) -> (UInt16, UInt16, UInt16)) -> CVPixelBuffer {
    var pb: CVPixelBuffer?
    let attrs: [String: Any] = [
        kCVPixelBufferPixelFormatTypeKey as String: MLXTileProcessor.deepPixelFormat,
        kCVPixelBufferWidthKey as String: width,
        kCVPixelBufferHeightKey as String: height,
    ]
    precondition(CVPixelBufferCreate(nil, width, height, MLXTileProcessor.deepPixelFormat,
                                     attrs as CFDictionary, &pb) == kCVReturnSuccess)
    let buf = pb!
    CVPixelBufferLockBaseAddress(buf, [])
    defer { CVPixelBufferUnlockBaseAddress(buf, []) }
    let base = CVPixelBufferGetBaseAddress(buf)!
    let bpr = CVPixelBufferGetBytesPerRow(buf)
    for y in 0 ..< height {
        let row = (base + y * bpr).assumingMemoryBound(to: UInt16.self)
        for x in 0 ..< width {
            let (r, g, b) = pixel(x, y)
            row[x * 4 + 0] = r.littleEndian; row[x * 4 + 1] = g.littleEndian
            row[x * 4 + 2] = b.littleEndian; row[x * 4 + 3] = UInt16.max
        }
    }
    return buf
}

/// A 64RGBALE buffer as packed RGBA UInt16.
private func readRGBA16(_ buf: CVPixelBuffer) -> (w: Int, h: Int, px: [UInt16]) {
    let w = CVPixelBufferGetWidth(buf), h = CVPixelBufferGetHeight(buf)
    CVPixelBufferLockBaseAddress(buf, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(buf, .readOnly) }
    let base = CVPixelBufferGetBaseAddress(buf)!
    let bpr = CVPixelBufferGetBytesPerRow(buf)
    var out = [UInt16](repeating: 0, count: w * h * 4)
    for y in 0 ..< h {
        let row = (base + y * bpr).assumingMemoryBound(to: UInt16.self)
        for i in 0 ..< (w * 4) { out[y * w * 4 + i] = UInt16(littleEndian: row[i]) }
    }
    return (w, h, out)
}

/// A gentle 16-bit gradient (≈ 0.1 of an 8-bit step per pixel, the smooth-sky case G-6 is about) plus
/// a full-range channel, so both fine steps and every high byte are exercised.
private func smoothRamp(_ n: Int) -> CVPixelBuffer {
    makeRGBA16(width: n, height: n) { x, y in
        (UInt16(20_000 + x * 26), UInt16((x * 7919 + y * 104_729) % 65_536), UInt16(40_000 + y * 3))
    }
}

@Suite(.serialized)
struct DeepTileTests {

    @Test("A16. identity forward through the tiler is bit-exact at 16 bits, with and without a halo")
    func tiledIdentityIsBitExactAt16Bits() throws {
        try withCPU {
            let input = smoothRamp(704)
            for halo in [0, 32] {
                let tiler = MLXTileProcessor(tileSize: 256, overlap: 32, scale: 1)
                let out = try tiler.process(input, halo: halo) { tile, _ in MLX.eval(tile); return tile }
                #expect(CVPixelBufferGetPixelFormatType(out) == MLXTileProcessor.deepPixelFormat)
                let a = readRGBA16(input), b = readRGBA16(out)
                #expect(a.w == b.w && a.h == b.h)
                let maxAbs = zip(a.px, b.px).map { abs(Int($0) - Int($1)) }.max() ?? -1
                #expect(maxAbs == 0, "halo=\(halo): 16-bit identity is not bit-exact (max|Δ| = \(maxAbs))")
            }
        }
    }

    @Test("A16w. the whole-frame path is bit-exact at 16 bits")
    func wholeFrameIdentityIsBitExactAt16Bits() throws {
        try withCPU {
            let input = smoothRamp(300)
            let tiler = MLXTileProcessor(tileSize: 256, overlap: 32, scale: 1)
            let out = try tiler.processWholeFrame(input) { x in MLX.eval(x); return x }
            #expect(CVPixelBufferGetPixelFormatType(out) == MLXTileProcessor.deepPixelFormat)
            #expect(readRGBA16(out).px == readRGBA16(input).px)
        }
    }

    /// The point of the path: a ramp with ~0.1 of an 8-bit step per pixel must come back with its
    /// sub-8-bit levels, where the 8-bit path collapses them into a staircase.
    @Test("A16s. a sub-8-bit ramp keeps its levels through a ×2 forward")
    func subEightBitLevelsSurviveAnUpscale() throws {
        try withCPU {
            let input = smoothRamp(300)
            let tiler = MLXTileProcessor(tileSize: 128, overlap: 16, scale: 2)
            // Nearest-neighbour ×2: every input level must reappear exactly.
            let up: (MLXArray) -> MLXArray = { x in
                let y = MLX.repeated(MLX.repeated(x, count: 2, axis: 1), count: 2, axis: 2)
                MLX.eval(y)
                return y
            }
            let tiled = try tiler.process(input) { t in up(t) }
            let whole = try tiler.processWholeFrame(input) { t in up(t) }
            let a = readRGBA16(input)
            for out in [tiled, whole] {
                let b = readRGBA16(out)
                #expect(b.w == 600 && b.h == 600)
                let levels = Set(stride(from: 0, to: b.px.count, by: 4).map { b.px[$0] })   // red channel
                let source = Set(stride(from: 0, to: a.px.count, by: 4).map { a.px[$0] })
                #expect(levels == source)
                #expect(levels.count > 256)
            }
        }
    }

    @Test("an 8-bit input still comes back 32BGRA")
    func eightBitInputKeepsTheEightBitPath() throws {
        try withCPU {
            var pb: CVPixelBuffer?
            let attrs: [String: Any] = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                                        kCVPixelBufferWidthKey as String: 64, kCVPixelBufferHeightKey as String: 64]
            #expect(CVPixelBufferCreate(nil, 64, 64, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb) == kCVReturnSuccess)
            let tiler = MLXTileProcessor(tileSize: 32, overlap: 8, scale: 1)
            let tiled = try tiler.process(pb!) { t in MLX.eval(t); return t }
            let whole = try tiler.processWholeFrame(pb!) { t in MLX.eval(t); return t }
            #expect(CVPixelBufferGetPixelFormatType(tiled) == kCVPixelFormatType_32BGRA)
            #expect(CVPixelBufferGetPixelFormatType(whole) == kCVPixelFormatType_32BGRA)
            #expect(!MLXTileProcessor.isDeep(pb!))
        }
    }
}
