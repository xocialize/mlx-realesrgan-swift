//
//  TileCompositorFidelityTests.swift
//
//  WORKORDER-tile-compositor: is the tiled-stills seam lattice a defect in the WRITE
//  layer rather than in the diffusion model? These probes need no model and no Metal —
//  the forward closure is identity (or a synthetic constant), and everything runs on the
//  CPU device.
//
//  Three independent questions, because they fail differently:
//
//   A. IDENTITY BIT-EXACTNESS. Push a known image through the tiler with an identity
//      forward. A correct compositor is bit-exact. Deviations, and where they land
//      relative to the `tileSize - overlap` write grid, are the quantisation claim.
//
//   B. EFFECTIVE WEIGHT MAP. Identity cannot see a *weighting* defect: when every tile
//      agrees, any convex combination returns the same value. So measure the weights
//      directly — run the tiler once per tile with that tile returning 1.0 and every
//      other tile 0.0; the output IS that tile's effective weight. Compare against the
//      normalised weighted average `w_j / Σ w_i` the composite is supposed to be.
//
//   C. FIRST-WRITE ALPHA FLAG. `writeTile` uses `dst[+3] == 0` as an "untouched" flag
//      against a `CVPixelBufferCreate` buffer. Probe whether that memory can come back
//      dirty.
//
//  Geometry under test is the shipping SeedVR2 stills geometry: tileSize 256, overlap 32,
//  scale 1 — i.e. a 224 px write stride, the observed artefact period.
//

import CoreGraphics
import CoreVideo
import Foundation
import ImageIO
import MLX
import Testing
import UniformTypeIdentifiers
@testable import RealESRGANMLX

// MARK: - Fixtures

/// Run with the MLX default device pinned to CPU — these probes never need a kernel, and
/// `swift test` does not reliably stage the Metal bundle (see SRVGGNetTests).
private func withCPU<R>(_ body: () throws -> R) rethrows -> R {
    try Device.withDefaultDevice(Device(.cpu), body)
}

private let kTile = 256
private let kOverlap = 32
private let kStride = kTile - kOverlap   // 224

/// Allocate a BGRA buffer and fill it from `pixel(x, y) -> (r, g, b)`.
private func makeBGRA(width: Int, height: Int,
                      _ pixel: (Int, Int) -> (UInt8, UInt8, UInt8)) -> CVPixelBuffer {
    var pb: CVPixelBuffer?
    let attrs: [String: Any] = [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        kCVPixelBufferWidthKey as String: width,
        kCVPixelBufferHeightKey as String: height,
    ]
    precondition(CVPixelBufferCreate(nil, width, height, kCVPixelFormatType_32BGRA,
                                     attrs as CFDictionary, &pb) == kCVReturnSuccess)
    let buf = pb!
    CVPixelBufferLockBaseAddress(buf, [])
    defer { CVPixelBufferUnlockBaseAddress(buf, []) }
    let p = CVPixelBufferGetBaseAddress(buf)!.assumingMemoryBound(to: UInt8.self)
    let bpr = CVPixelBufferGetBytesPerRow(buf)
    for y in 0 ..< height {
        for x in 0 ..< width {
            let (r, g, b) = pixel(x, y)
            let o = y * bpr + x * 4
            p[o + 0] = b; p[o + 1] = g; p[o + 2] = r; p[o + 3] = 255
        }
    }
    return buf
}

/// Load a PNG, centre-crop to `size × size`, as BGRA. Returns nil if the file is absent.
private func loadBGRA(_ path: String, size: Int) -> CVPixelBuffer? {
    guard let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
          let cg = CGImageSourceCreateImageAtIndex(src, 0, nil) else { return nil }
    let ox = max(0, (cg.width - size) / 2)
    let oy = max(0, (cg.height - size) / 2)
    guard let crop = cg.cropping(to: CGRect(x: ox, y: oy, width: size, height: size)) else { return nil }
    var pb: CVPixelBuffer?
    let attrs: [String: Any] = [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        kCVPixelBufferWidthKey as String: size,
        kCVPixelBufferHeightKey as String: size,
    ]
    precondition(CVPixelBufferCreate(nil, size, size, kCVPixelFormatType_32BGRA,
                                     attrs as CFDictionary, &pb) == kCVReturnSuccess)
    let buf = pb!
    CVPixelBufferLockBaseAddress(buf, [])
    defer { CVPixelBufferUnlockBaseAddress(buf, []) }
    let ctx = CGContext(data: CVPixelBufferGetBaseAddress(buf),
                        width: size, height: size, bitsPerComponent: 8,
                        bytesPerRow: CVPixelBufferGetBytesPerRow(buf),
                        space: CGColorSpaceCreateDeviceRGB(),
                        bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
                            | CGBitmapInfo.byteOrder32Little.rawValue)!
    ctx.draw(crop, in: CGRect(x: 0, y: 0, width: size, height: size))
    return buf
}

/// Read a BGRA buffer as `[y][x] -> (b, g, r)`.
private func readBGRA(_ buf: CVPixelBuffer) -> (w: Int, h: Int, px: [UInt8]) {
    let w = CVPixelBufferGetWidth(buf), h = CVPixelBufferGetHeight(buf)
    CVPixelBufferLockBaseAddress(buf, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(buf, .readOnly) }
    let p = CVPixelBufferGetBaseAddress(buf)!.assumingMemoryBound(to: UInt8.self)
    let bpr = CVPixelBufferGetBytesPerRow(buf)
    var out = [UInt8](repeating: 0, count: w * h * 4)
    for y in 0 ..< h {
        for x in 0 ..< w {
            let s = y * bpr + x * 4, d = (y * w + x) * 4
            out[d + 0] = p[s + 0]; out[d + 1] = p[s + 1]
            out[d + 2] = p[s + 2]; out[d + 3] = p[s + 3]
        }
    }
    return (w, h, out)
}

/// The feather weight `SeamAccumulator` computes, reproduced analytically: half-pixel-centred
/// ramps, so a pair across a seam sums to exactly 1 and no tile's weight is ever exactly 0.
/// `tx`/`ty` are tile-local output coordinates; `outW`/`outH` the tile output extent.
private func featherWeight(tx: Int, ty: Int, outW: Int, outH: Int, ov: Int) -> Float {
    func ramp(_ i: Int, _ extent: Int) -> Float {
        min(min((Float(i) + 0.5) / Float(ov), 1.0),
            min((Float(extent - 1 - i) + 0.5) / Float(ov), 1.0))
    }
    return ramp(tx, outW) * ramp(ty, outH)
}

// MARK: - Probe A — identity bit-exactness

@Suite(.serialized)
struct TileCompositorFidelityTests {

    @Test("A. identity forward through the tiler must be bit-exact")
    func identityIsBitExact() throws {
        try withCPU {
            let corpus = "/Volumes/Satechi/Development/mlxengine-forge/Corpus/stills/signage/SNS_graphic.png"
            var subjects: [(String, CVPixelBuffer)] = []
            if let real = loadBGRA(corpus, size: 2048) { subjects.append(("SNS_graphic 2048²", real)) }
            // Synthetic control: a smooth ramp exercises every code value, and a mid-grey flat
            // is where a 1-LSB lattice would be visible.
            subjects.append(("ramp 704²", makeBGRA(width: 704, height: 704) { x, y in
                (UInt8(x % 256), UInt8(y % 256), UInt8((x + y) % 256))
            }))
            subjects.append(("flat-128 704²", makeBGRA(width: 704, height: 704) { _, _ in (128, 128, 128) }))

            for (name, input) in subjects {
                for halo in [0, 32] {
                    let tiler = MLXTileProcessor(tileSize: kTile, overlap: kOverlap, scale: 1)
                    let out = try tiler.process(input, halo: halo) { tile, _ in
                        MLX.eval(tile)
                        return tile
                    }
                    let a = readBGRA(input), b = readBGRA(out)
                    #expect(a.w == b.w && a.h == b.h)

                    var maxAbs = 0
                    var nDiff = 0
                    // Where do deviations land relative to the 224 write grid?
                    var foldX = [Int](repeating: 0, count: kStride)
                    var foldY = [Int](repeating: 0, count: kStride)
                    var signSum = 0
                    for y in 0 ..< a.h {
                        for x in 0 ..< a.w {
                            let o = (y * a.w + x) * 4
                            var worst = 0
                            for c in 0 ..< 3 {
                                let d = Int(b.px[o + c]) - Int(a.px[o + c])
                                if abs(d) > abs(worst) { worst = d }
                            }
                            if worst != 0 {
                                nDiff += 1
                                signSum += worst > 0 ? 1 : -1
                                maxAbs = max(maxAbs, abs(worst))
                                foldX[x % kStride] += 1
                                foldY[y % kStride] += 1
                            }
                        }
                    }
                    let pct = 100.0 * Double(nDiff) / Double(a.w * a.h)
                    print(String(format: "A  %-18s halo=%2d  max|Δ| = %d   pixels differing = %d (%.4f%%)  net sign = %+d",
                                 (name as NSString).utf8String!, halo, maxAbs, nDiff, pct, signSum))
                    if nDiff > 0 {
                        // Concentration on the write grid: how many of the deviating pixels sit in
                        // the 32-wide overlap band that starts at each 224 boundary?
                        let inBandX = (0 ..< kOverlap).reduce(0) { $0 + foldX[$1] }
                        let inBandY = (0 ..< kOverlap).reduce(0) { $0 + foldY[$1] }
                        print(String(format: "   fold on the %d grid: %d/%d deviations in the first %d columns, %d/%d in the first %d rows",
                                     kStride, inBandX, nDiff, kOverlap, inBandY, nDiff, kOverlap))
                        print("   foldX = \(foldX.enumerated().filter { $0.element > 0 }.map { "\($0.offset):\($0.element)" }.prefix(24).joined(separator: " "))")
                    }
                    #expect(maxAbs == 0,
                            "\(name) halo=\(halo): identity forward is not bit-exact through the compositor (max|Δ| = \(maxAbs))")
                }
            }
        }
    }

    // MARK: - Probe B — effective weight map

    @Test("B. the composite must be a normalised weighted average of the contributing tiles")
    func effectiveWeightsAreNormalised() throws {
        try withCPU {
            // The 4-tile corner under test: tiles with origins (0,0), (224,0), (0,224),
            // (224,224) all cover the square x,y ∈ [224, 256).
            let quad = [(0, 0), (224, 0), (0, 224), (224, 224)]

            // ⚠️ Each run uses a DIFFERENT canvas size so `process`'s CVPixelBufferCreate gets a
            // fresh (zeroed) allocation rather than the previous run's recycled bytes — see
            // probe C. 704 + k keeps the quad's geometry identical (only the clamped last
            // tile origin moves, and it is nowhere near the corner).
            var effective = [[Float]]()
            var sides = [Int]()
            for (k, (tx0, ty0)) in quad.enumerated() {
                let side = 704 + k
                sides.append(side)
                let blank = makeBGRA(width: side, height: side) { _, _ in (0, 0, 0) }
                let tiler = MLXTileProcessor(tileSize: kTile, overlap: kOverlap, scale: 1)
                let out = try tiler.process(blank) { _, region in
                    let on = (region.x == tx0 && region.y == ty0)
                    let a = MLXArray(
                        [Float](repeating: on ? 1.0 : 0.0, count: kTile * kTile * 3),
                        [1, kTile, kTile, 3])
                    MLX.eval(a)
                    return a
                }
                let r = readBGRA(out)
                var w = [Float](repeating: 0, count: side * side)
                for p in 0 ..< side * side { w[p] = Float(r.px[p * 4 + 2]) / 255.0 }
                effective.append(w)
            }
            func eff(_ j: Int, _ x: Int, _ y: Int) -> Float { effective[j][y * sides[j] + x] }

            func analytic(_ tx0: Int, _ ty0: Int, _ x: Int, _ y: Int) -> Float {
                let tx = x - tx0, ty = y - ty0
                guard tx >= 0, tx < kTile, ty >= 0, ty < kTile else { return 0 }
                return featherWeight(tx: tx, ty: ty, outW: kTile, outH: kTile, ov: kOverlap)
            }

            print("\nB. effective vs normalised weights at the 4-tile corner (x,y ∈ [224,256))")
            print("      pixel        measured effective w              correct normalised w         maxerr  Σmeas")
            var worstErr: Float = 0
            var worstAt = (0, 0)
            for y in stride(from: 224, to: 256, by: 4) {
                for x in stride(from: 224, to: 256, by: 4) {
                    let meas = (0 ..< 4).map { eff($0, x, y) }
                    let raw = quad.map { analytic($0.0, $0.1, x, y) }
                    let s = raw.reduce(0, +)
                    let norm = s > 0 ? raw.map { $0 / s } : raw
                    let err = zip(meas, norm).map { abs($0 - $1) }.max() ?? 0
                    if err > worstErr { worstErr = err; worstAt = (x, y) }
                    print(String(format: "   (%4d,%4d)  [%.3f %.3f %.3f %.3f]   [%.3f %.3f %.3f %.3f]   %.3f  %.3f",
                                 x, y, meas[0], meas[1], meas[2], meas[3],
                                 norm[0], norm[1], norm[2], norm[3], err, meas.reduce(0, +)))
                }
            }
            print(String(format: "   WORST corner weight error = %.4f at (%d,%d)", worstErr, worstAt.0, worstAt.1))

            // The pure 1-D seam (two tiles only), for contrast — the case the current writer
            // gets right by construction.
            print("\nB. effective vs normalised on the 1-D vertical seam (y = 112, x ∈ [224,256))")
            var worst1D: Float = 0
            for x in stride(from: 224, to: 256, by: 4) {
                let y = 112
                let meas = [eff(0, x, y), eff(1, x, y)]
                let raw = [analytic(0, 0, x, y), analytic(224, 0, x, y)]
                let s = raw.reduce(0, +)
                let norm = s > 0 ? raw.map { $0 / s } : raw
                let err = zip(meas, norm).map { abs($0 - $1) }.max() ?? 0
                worst1D = max(worst1D, err)
                print(String(format: "   x=%4d  meas [%.3f %.3f]  norm [%.3f %.3f]  err %.3f",
                             x, meas[0], meas[1], norm[0], norm[1], err))
            }
            print(String(format: "   WORST 1-D seam weight error = %.4f", worst1D))

            #expect(worstErr < 0.01,
                    "4-tile corner composite is not a normalised weighted average (max weight error \(worstErr))")
        }
    }

    // MARK: - Probe E — duplicate clamped origins must not re-run the forward

    @Test("E. the forward runs exactly once per unique clamped tile origin")
    func forwardRunsOncePerUniqueOrigin() throws {
        try withCPU {
            // (side, unique origins per axis): the exactly-one-tile frame that measured FOUR
            // identical diffusion calls per frame (V12-D gate probe, 2026-07-31); a sub-tile
            // frame; a frame whose last stride origin clamps onto the previous one
            // (2048 = 224·9 + 32); and a full multi-tile frame.
            let cases: [(Int, Int)] = [(256, 1), (200, 1), (2048, 9), (704, 3)]
            for (side, perAxis) in cases {
                let input = makeBGRA(width: side, height: side) { x, y in
                    (UInt8(x % 256), UInt8(y % 256), UInt8((x + y) % 256))
                }
                let tiler = MLXTileProcessor(tileSize: kTile, overlap: kOverlap, scale: 1)
                var origins = [[Int]]()
                let out = try tiler.process(input) { tile, region in
                    origins.append([region.x, region.y])
                    MLX.eval(tile)
                    return tile
                }
                let unique = Set(origins)
                #expect(origins.count == perAxis * perAxis,
                        "\(side)²: forward ran \(origins.count)× for \(unique.count) unique origins")
                #expect(unique.count == origins.count,
                        "\(side)²: duplicate clamped origins re-ran the forward")
                // The dedup must not change output bytes (identity forward is bit-exact).
                let a = readBGRA(input), b = readBGRA(out)
                #expect(a.px == b.px, "\(side)²: deduplicated grid changed the composite")
            }
        }
    }

    // MARK: - Probe C — the alpha "untouched" flag

    @Test("C. CVPixelBufferCreate memory can it come back with a non-zero alpha byte")
    func alphaFlagIsTrustworthy() throws {
        // Allocate, dirty, release, reallocate — many times — and count how often a freshly
        // created buffer of the tiler's output geometry hands back a non-zero alpha byte.
        let w = 2048, h = 2048
        var dirtyRuns = 0
        var worstNonZero = 0
        for iter in 0 ..< 24 {
            var pb: CVPixelBuffer?
            let attrs: [String: Any] = [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey as String: w,
                kCVPixelBufferHeightKey as String: h,
            ]
            guard CVPixelBufferCreate(nil, w, h, kCVPixelFormatType_32BGRA,
                                      attrs as CFDictionary, &pb) == kCVReturnSuccess,
                  let buf = pb else { continue }
            CVPixelBufferLockBaseAddress(buf, [])
            let p = CVPixelBufferGetBaseAddress(buf)!.assumingMemoryBound(to: UInt8.self)
            let bpr = CVPixelBufferGetBytesPerRow(buf)
            var nonZero = 0
            for y in 0 ..< h {
                for x in 0 ..< w where p[y * bpr + x * 4 + 3] != 0 { nonZero += 1 }
            }
            if nonZero > 0 { dirtyRuns += 1; worstNonZero = max(worstNonZero, nonZero) }
            // Dirty it thoroughly so a recycled allocation would show up on the next iteration.
            if iter % 2 == 0 {
                memset(p, 0xFF, bpr * h)
            }
            CVPixelBufferUnlockBaseAddress(buf, [])
        }
        print(String(format: "\nC. fresh %dx%d CVPixelBufferCreate: %d/24 allocations had a non-zero alpha byte (worst %d pixels)",
                     w, h, dirtyRuns, worstNonZero))
    }

    // MARK: - Probe D — what it looks like

    /// Second-run realism: process image A, then image B at the same size, exactly as any
    /// multi-image or multi-frame session does. B's output buffer is A's recycled memory.
    /// Dumps B's identity-forward output and a ×8 error map so the artefact can be LOOKED at.
    @Test("D. dump the second-run identity output and its error map")
    func dumpSecondRunArtefact() throws {
        try withCPU {
            let dumpDir = ProcessInfo.processInfo.environment["TILE_DUMP"]
                ?? "/private/tmp/claude-501/-Volumes-Satechi-Development/822535e3-ebe1-426a-bfbf-b0e956bf68ae/scratchpad/tilecomp"
            let tag = ProcessInfo.processInfo.environment["TILE_DUMP_TAG"] ?? "before"
            try? FileManager.default.createDirectory(atPath: dumpDir, withIntermediateDirectories: true)

            let corpus = "/Volumes/Satechi/Development/mlxengine-forge/Corpus/stills/signage/"
            guard let imgA = loadBGRA(corpus + "Bayer_photo.png", size: 2048),
                  let imgB = loadBGRA(corpus + "SNS_graphic.png", size: 2048) else {
                print("D. corpus unavailable — skipped"); return
            }
            let tiler = MLXTileProcessor(tileSize: kTile, overlap: kOverlap, scale: 1)
            let idf: (MLXArray, TileRegion) throws -> MLXArray = { t, _ in MLX.eval(t); return t }
            _ = try tiler.process(imgA, halo: 0, forward: idf)     // run 1 — buffer allocated fresh
            let outB = try tiler.process(imgB, halo: 0, forward: idf)  // run 2 — recycles run 1's bytes

            let src = readBGRA(imgB), got = readBGRA(outB)
            var maxAbs = 0, nDiff = 0
            var errAmp = [UInt8](repeating: 0, count: src.w * src.h * 4)
            for p in 0 ..< src.w * src.h {
                var worst = 0
                for c in 0 ..< 3 {
                    let d = Int(got.px[p * 4 + c]) - Int(src.px[p * 4 + c])
                    if abs(d) > abs(worst) { worst = d }
                    errAmp[p * 4 + c] = UInt8(min(255, abs(d) * 8))
                }
                errAmp[p * 4 + 3] = 255
                if worst != 0 { nDiff += 1; maxAbs = max(maxAbs, abs(worst)) }
            }
            print(String(format: "\nD. second run (SNS after Bayer, same 2048² geometry): max|Δ| = %d, pixels differing = %d (%.2f%%)",
                         maxAbs, nDiff, 100.0 * Double(nDiff) / Double(src.w * src.h)))
            writePNG(got.px, src.w, src.h, "\(dumpDir)/\(tag)_secondrun_out.png")
            writePNG(errAmp, src.w, src.h, "\(dumpDir)/\(tag)_secondrun_err8x.png")
            writePNG(src.px, src.w, src.h, "\(dumpDir)/reference_SNS.png")
            print("   dumped to \(dumpDir)/\(tag)_secondrun_{out,err8x}.png")
        }
    }
}

/// Write packed BGRA bytes out as a PNG.
private func writePNG(_ bgra: [UInt8], _ w: Int, _ h: Int, _ path: String) {
    var data = bgra
    data.withUnsafeMutableBytes { raw in
        let ctx = CGContext(data: raw.baseAddress, width: w, height: h, bitsPerComponent: 8,
                            bytesPerRow: w * 4, space: CGColorSpaceCreateDeviceRGB(),
                            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
                                | CGBitmapInfo.byteOrder32Little.rawValue)!
        guard let cg = ctx.makeImage(),
              let dest = CGImageDestinationCreateWithURL(
                URL(fileURLWithPath: path) as CFURL, UTType.png.identifier as CFString, 1, nil)
        else { return }
        CGImageDestinationAddImage(dest, cg, nil)
        CGImageDestinationFinalize(dest)
    }
}
