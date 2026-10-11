// MLXTileProcessor.swift
//
// Role: Tile-based driver for the MLX-Swift backed playback tiers
//       (`EfRLFN_Playback`, `SRVGGNetCompact_Playback`). Splits a full
//       BGRA `CVPixelBuffer` into overlapping NHWC tiles, runs each
//       through the supplied MLX forward closure, and feather-blends the
//       upscaled tiles into a BGRA output buffer.
//
// Why a parallel helper (not `TileProcessor`)?
//   - `TileProcessor` (legacy / CoreML) is hard-typed to `MLModel.prediction`
//     and CHW MLMultiArray inputs.
//   - The MLX path uses NHWC `MLXArray` inputs and a Swift closure for the
//     forward (so we don't have to pass concrete EfRLFN / SRVGGNetCompact
//     types into a shared driver).
//   - Same feathered-overlap blending math as `TileProcessor.writeTile`,
//     kept byte-faithful so seams look identical regardless of backend.
//
// Conventions:
//   - BGRA → RGB float32 [0, 1] on the way in (matches both upstream
//     EfRLFN and SRVGGNetCompact training preprocessing).
//   - RGB float32 [0, 1] → BGRA UInt8 on the way out with feathered seam
//     blending and per-pixel clamp.
//   - 16-bit (AB-A-0158, MLXToolKit contract 1.52.0 `.rawRGBA16`): a
//     `kCVPixelFormatType_64RGBALE` input (`MLXTileProcessor.deepPixelFormat`)
//     is read as RGBA UInt16 / 65535 and written back as 64RGBALE — the output
//     always has the input's depth. Every other format takes the 8-bit path,
//     byte for byte as before.
//   - `MLX.eval(output)` is called inside the per-tile forward — required
//     before pulling values into CVPixelBuffer (MLX is lazy; an unevaluated
//     tensor materialises as zeros, the silent killer from mlx-porting).
//   - NHWC throughout — matches CLAUDE.md and the underlying MLX modules.
//   - Marked `@unchecked Sendable` because closures capture MLX state
//     that's not formally Sendable; the calling tier already enforces
//     single-thread access through the `async` boundary.

import CoreImage
import CoreVideo
import Foundation
import MLX
import MLXProfiling

/// Where a tile sits in the *input* frame, in input-pixel space.
///
/// Handed to the region-aware `process(_:forward:)` closure so per-tile work that
/// needs global context — e.g. slicing a whole-image noise field so it stays
/// continuous across seams (the SeedVR2 tiled-refine case) — knows which region
/// the tile covers. `x`/`y` are the clamped tile origin; `width`/`height` are the
/// in-frame extent (≤ `tileSize` only at a right/bottom edge). The tensor the
/// closure receives is always `tileSize × tileSize` — pixels beyond
/// `width`/`height` are edge-replicated padding.
public struct TileRegion: Sendable {
    public let x: Int
    public let y: Int
    public let width: Int
    public let height: Int
    /// Margin of **real surrounding image content** included on every side of the tensor handed to the
    /// closure (`0` for the plain overloads). With a halo the tensor is `(tileSize + 2·halo)²` and its
    /// top-left corner sits at `(x - halo, y - halo)` in input-pixel space; the driver crops the halo
    /// off the closure's output before blending, so the closure never has to.
    ///
    /// 🔑 **Index whole-image fields at `x`/`y`, not at `x - halo`** — a field drawn over the image
    /// *padded by `halo` on every side* is indexed by the unpadded origin, which keeps the arithmetic
    /// non-negative and identical between neighbours. (The SeedVR2 tiled-refine noise field does this.)
    public let halo: Int

    public init(x: Int, y: Int, width: Int, height: Int, halo: Int = 0) {
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.halo = halo
    }
}

/// Tile driver for an MLX-Swift forward closure. The closure receives an
/// `[1, H, W, 3]` NHWC RGB float32 tensor and returns an upscaled tensor
/// `[1, H*scale, W*scale, 3]`. The driver handles CVPixelBuffer extraction,
/// stitching, and feather blending.
public struct MLXTileProcessor: @unchecked Sendable {

    let tileSize: Int
    let overlap: Int
    let scale: Int

    /// Shared CIContext for the input format normalisation below. Building a
    /// CIContext is expensive, so keep one alive for the process lifetime.
    private static let ciContext = CIContext(options: [.cacheIntermediates: false])

    public init(tileSize: Int, overlap: Int, scale: Int) {
        self.tileSize = tileSize
        self.overlap = overlap
        self.scale = scale
    }

    /// The 16-bit layout the driver reads and writes natively: RGBA, 16 bits per component, little-endian,
    /// 0…65535 — MLXToolKit's `.rawRGBA16` (contract 1.52.0) as a `CVPixelBuffer`. Hand the driver a buffer in
    /// this format and the result comes back in it, at 16 bits; anything else runs the 8-bit BGRA path.
    public static let deepPixelFormat: OSType = kCVPixelFormatType_64RGBALE

    /// Whether `buffer` takes the 16-bit path.
    public static func isDeep(_ buffer: CVPixelBuffer) -> Bool {
        CVPixelBufferGetPixelFormatType(buffer) == deepPixelFormat
    }

    /// The input as the driver reads it: a 16-bit buffer as-is, anything else normalised to 32BGRA.
    private func normalizedInput(_ input: CVPixelBuffer) -> CVPixelBuffer {
        Self.isDeep(input) ? input : normalizedBGRA(input)
    }

    /// Normalise any incoming `CVPixelBuffer` to packed 32BGRA.
    ///
    /// CRITICAL: `FFmpegDecoder` (and the conversion pipeline generally) emits
    /// **NV12** (`kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange`) — biplanar
    /// YUV — not BGRA. The tile extraction below reads the buffer as packed
    /// 4-byte BGRA via `CVPixelBufferGetBaseAddress`; handed an NV12 buffer it
    /// reads the Y (luma) plane as if it were BGRA, producing a sheared,
    /// grayscale garbage frame that the SR model then "upscales". This is the
    /// bug that made the Phase C.4 A/B score VMAF ≈ 0 on detailed content
    /// (smooth content survived the shear, which is why it wasn't uniformly
    /// zero). CoreImage decodes NV12 → RGB with the correct YCbCr matrix and
    /// video-range expansion, so render through it whenever the input isn't
    /// already BGRA. A buffer that is already BGRA passes straight through.
    private func normalizedBGRA(_ input: CVPixelBuffer) -> CVPixelBuffer {
        if CVPixelBufferGetPixelFormatType(input) == kCVPixelFormatType_32BGRA {
            return input
        }
        let width = CVPixelBufferGetWidth(input)
        let height = CVPixelBufferGetHeight(input)
        var converted: CVPixelBuffer?
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:],
        ]
        guard CVPixelBufferCreate(nil, width, height, kCVPixelFormatType_32BGRA,
                                  attrs as CFDictionary, &converted) == kCVReturnSuccess,
              let out = converted else {
            return input  // fall back; extraction will be wrong but won't crash
        }
        let ci = CIImage(cvPixelBuffer: input)
        Self.ciContext.render(ci, to: out)
        return out
    }

    /// Process a full BGRA frame through the supplied MLX forward closure.
    ///
    /// - Parameters:
    ///   - input: source BGRA `CVPixelBuffer`.
    ///   - forward: closure that takes an `[1, tileH, tileW, 3]` NHWC RGB
    ///     float32 tensor and returns the upscaled result. The closure is
    ///     responsible for calling `MLX.eval` on its return value before
    ///     returning, so the driver can safely read floats out.
    /// - Returns: upscaled BGRA `CVPixelBuffer` at `scale` × the input
    ///   dimensions.
    public func process(
        _ rawInput: CVPixelBuffer,
        forward: (MLXArray) throws -> MLXArray
    ) throws -> CVPixelBuffer {
        try process(rawInput) { tile, _ in try forward(tile) }
    }

    /// Region-aware variant of `process(_:forward:)`: identical driving, but the
    /// closure also receives the tile's `TileRegion` (input-pixel space). Use this
    /// when the per-tile forward needs to know *where* the tile came from — e.g. to
    /// slice a precomputed whole-image field (noise, guidance) so it stays
    /// continuous across tile seams.
    public func process(
        _ rawInput: CVPixelBuffer,
        forward: (MLXArray, TileRegion) throws -> MLXArray
    ) throws -> CVPixelBuffer {
        try process(rawInput, halo: 0, forward: forward)
    }

    /// **Halo (context-padding) variant.** Each tile is handed a `(tileSize + 2·halo)²` window whose
    /// margin is *real neighbouring image content* (edge-replicated only where the window leaves the
    /// frame); the driver crops that margin off the closure's output and feather-blends the `tileSize²`
    /// centre exactly as the plain overloads do.
    ///
    /// 🔑 **Why this is not the same lever as `overlap`.** Overlap blends two outputs that were
    /// produced from different context and therefore already disagree; a halo makes them agree in the
    /// first place, because each tile's receptive field actually contains its neighbourhood instead of a
    /// replicated border. For a fully-convolutional network with no downsampling the two are nearly
    /// equivalent, and this overload buys nothing — it exists for backends whose per-tile output depends
    /// on the whole window (windowed attention, diffusion), where a border is not a neighbourhood.
    ///
    /// ⚠️ **Not free:** at `tileSize` 256 a 64 px halo processes 384², i.e. **2.25× the pixels per
    /// tile**. `halo: 0` delegates to the plain path and is byte-identical to it.
    ///
    /// - Parameter halo: context margin in **input** pixels, `>= 0`. Backends with alignment
    ///   constraints (e.g. a `/16` window/patch scheme) must choose a `halo` that keeps
    ///   `tileSize + 2·halo` aligned — the driver does not round it.
    public func process(
        _ rawInput: CVPixelBuffer,
        halo: Int,
        forward: (MLXArray, TileRegion) throws -> MLXArray
    ) throws -> CVPixelBuffer {
        precondition(halo >= 0, "halo must be non-negative, got \(halo)")
        let input = normalizedInput(rawInput)
        // 16 bits in ⇒ 16 bits out (`deepPixelFormat`); every other input runs the 8-bit path unchanged.
        let deep = Self.isDeep(input)
        let outFormat = deep ? Self.deepPixelFormat : kCVPixelFormatType_32BGRA
        let width = CVPixelBufferGetWidth(input)
        let height = CVPixelBufferGetHeight(input)
        let outWidth = width * scale
        let outHeight = height * scale
        let step = max(tileSize - overlap, 1)

        var outputBuffer: CVPixelBuffer?
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: outFormat,
            kCVPixelBufferWidthKey as String: outWidth,
            kCVPixelBufferHeightKey as String: outHeight,
        ]
        let status = CVPixelBufferCreate(
            nil, outWidth, outHeight, outFormat,
            attrs as CFDictionary, &outputBuffer
        )
        guard status == kCVReturnSuccess, let output = outputBuffer else {
            throw PlaybackTierError.inferenceError(
                "CVPixelBufferCreate failed (status=\(status))"
            )
        }

        CVPixelBufferLockBaseAddress(input, .readOnly)
        CVPixelBufferLockBaseAddress(output, [])
        defer {
            CVPixelBufferUnlockBaseAddress(input, .readOnly)
            CVPixelBufferUnlockBaseAddress(output, [])
        }

        guard let srcBase = CVPixelBufferGetBaseAddress(input) else {
            throw PlaybackTierError.inferenceError("CVPixelBuffer base address is nil (input)")
        }
        guard let dstBase = CVPixelBufferGetBaseAddress(output) else {
            throw PlaybackTierError.inferenceError("CVPixelBuffer base address is nil (output)")
        }
        let srcPtr = srcBase.assumingMemoryBound(to: UInt8.self)
        let srcBPR = CVPixelBufferGetBytesPerRow(input)
        let dstBPR = CVPixelBufferGetBytesPerRow(output)

        // Pre-allocate the per-tile float32 scratch buffer so we don't
        // thrash the allocator for every tile of a 1080p frame. With a halo the
        // window handed to the closure is `(tileSize + 2·halo)²`.
        let windowSize = tileSize + 2 * halo
        var tileScratch = [Float](repeating: 0, count: windowSize * windowSize * 3)

        // Per-tile spans (MLX_PROFILE=1). `forward` evals inside each tile, so a span is the
        // honest full tile cost (extract → forward/eval → readout → feather-blend). No-op when
        // MLX_PROFILE is unset.
        let prof = MLXProfiler.shared
        var tileIndex = 0

        // Feathered composite accumulator. Tiles arrive in raster order, so only the rows one
        // tile row can still touch stay open — `tileSize·scale` of them. See `SeamAccumulator`
        // for why the old in-place 8-bit blend was replaced.
        var accum = SeamAccumulator(outWidth: outWidth, outHeight: outHeight,
                                    rows: tileSize * scale)

        // Clamped tile origins with duplicates collapsed. Clamping alone
        // (`min(tileX, max(0, width - tileSize))`) makes every stride origin within
        // `overlap` of the far edge — and ALL of them when the extent is ≤ tileSize —
        // land on the same position, so the identical tile would be extracted and run
        // through `forward` once per duplicate. The normalised composite cancels the
        // repeats out, so this is pure re-compute (measured 4× the diffusion cost per
        // frame at exactly-one-tile geometry). Same deduplicated grid as
        // `SeedVR2TemporalRefiner.tileGrid`.
        var tileYs: [Int] = []
        for tileY in stride(from: 0, to: height, by: step) {
            let y = min(tileY, max(0, height - tileSize))
            if tileYs.last != y { tileYs.append(y) }
        }
        var tileXs: [Int] = []
        for tileX in stride(from: 0, to: width, by: step) {
            let x = min(tileX, max(0, width - tileSize))
            if tileXs.last != x { tileXs.append(x) }
        }

        for y in tileYs {
            // Every output row above this tile row's first row is final: flush and slide.
            accum.flush(upTo: y * scale, dst: dstBase, dstBPR: dstBPR, deep: deep)
            for x in tileXs {
                // Cooperative cancellation (CAN gate): one checkpoint per 64² tile — the
                // natural yield point of the tiled forward. Sync code on the run's task sees
                // the flag; the CancellationError propagates unchanged to the wrapper.
                try Task.checkCancellation()
                let span = prof.begin("upscale", "tile", index: tileIndex)
                let tw = min(tileSize, width - x)
                let th = min(tileSize, height - y)

                // Extract BGRA → NHWC RGB float32 [0, 1] into the scratch buffer. The window
                // is the tile plus its halo margin; any part of the window outside the FRAME
                // clamps to the nearest valid pixel (edge replication, same approach as
                // TileProcessor) — but inside the frame the halo reads REAL neighbouring
                // pixels, which is the entire point of context padding.
                if deep {
                    fillWindowScratch16(
                        src: srcBase,
                        srcBPR: srcBPR,
                        windowX: x - halo, windowY: y - halo, windowSize: windowSize,
                        frameW: width, frameH: height,
                        scratch: &tileScratch
                    )
                } else {
                    fillWindowScratch(
                        src: srcPtr,
                        srcBPR: srcBPR,
                        windowX: x - halo, windowY: y - halo, windowSize: windowSize,
                        frameW: width, frameH: height,
                        scratch: &tileScratch
                    )
                }

                // Wrap scratch as `[1, windowSize, windowSize, 3]` NHWC.
                let tileArray = MLXArray(tileScratch, [1, windowSize, windowSize, 3])
                var upscaled = try forward(tileArray, TileRegion(x: x, y: y, width: tw, height: th,
                                                                 halo: halo))
                // `forward` is contractually expected to MLX.eval before
                // returning; we re-eval defensively so a stray lazy graph
                // can't silently zero the read-out.
                MLX.eval(upscaled)

                var shape = upscaled.shape
                guard shape.count == 4,
                      shape[0] == 1,
                      shape[3] == 3 else {
                    throw PlaybackTierError.inferenceError(
                        "unexpected forward output shape \(shape); want [1, H, W, 3]"
                    )
                }
                if halo > 0 {
                    // Crop the halo off the output so only the tile's own centre is blended.
                    // The margin exists to give the forward context, never to be written out.
                    let hs = halo * scale
                    let core = tileSize * scale
                    guard shape[1] >= core + 2 * hs, shape[2] >= core + 2 * hs else {
                        throw PlaybackTierError.inferenceError(
                            "halo forward output \(shape) smaller than window "
                            + "\((core + 2 * hs))² — closure must process the full window"
                        )
                    }
                    upscaled = upscaled[0..., hs ..< (hs + core), hs ..< (hs + core), 0...]
                    MLX.eval(upscaled)
                    shape = upscaled.shape
                }
                let outH = shape[1]
                let outW = shape[2]
                let outFloats = upscaled.asArray(Float.self)

                accum.add(
                    outFloats: outFloats,
                    outH: outH, outW: outW,
                    x: x * scale, y: y * scale,
                    tw: tw * scale, th: th * scale,
                    overlapScaled: overlap * scale
                )
                prof.end(span)
                tileIndex += 1
            }
        }
        accum.flush(upTo: outHeight, dst: dstBase, dstBPR: dstBPR, deep: deep)

        return output
    }

    // MARK: - Whole-frame fast path

    /// Adaptive entry point: run the whole frame through `forward` in a
    /// single pass when it fits `wholeFrameMaxPixels`, otherwise fall back
    /// to the tiled `process(_:forward:)`.
    ///
    /// EfRLFN and SRVGGNetCompact are both fully convolutional with **no
    /// internal downsampling**, so a whole-frame forward is numerically
    /// equivalent to the tiled path minus the feather-blend seams — and runs
    /// one GPU dispatch instead of one per tile (≈170 tiles for a 1080p frame
    /// at 128/16). The tiled path stays as the memory-bounded fallback for
    /// frames above the budget (e.g. the 4K corpus clip).
    ///
    /// - Parameter wholeFrameMaxPixels: input-pixel ceiling for the
    ///   single-pass path. Pass `0` to force tiling.
    public func processAdaptive(
        _ input: CVPixelBuffer,
        wholeFrameMaxPixels: Int,
        forward: (MLXArray) throws -> MLXArray
    ) throws -> CVPixelBuffer {
        let width = CVPixelBufferGetWidth(input)
        let height = CVPixelBufferGetHeight(input)
        if wholeFrameMaxPixels > 0, width * height <= wholeFrameMaxPixels {
            return try processWholeFrame(input, forward: forward)
        }
        return try process(input, forward: forward)
    }

    /// Single-pass forward over the entire frame. No tiling, no feather
    /// blending — the whole BGRA frame is lifted to one `[1, H, W, 3]` NHWC
    /// RGB float32 tensor, run through `forward`, and the `[1, H*scale,
    /// W*scale, 3]` result is written straight back to a BGRA buffer with a
    /// per-pixel `[0, 1]` clamp.
    public func processWholeFrame(
        _ rawInput: CVPixelBuffer,
        forward: (MLXArray) throws -> MLXArray
    ) throws -> CVPixelBuffer {
        let input = normalizedInput(rawInput)
        let deep = Self.isDeep(input)
        let width = CVPixelBufferGetWidth(input)
        let height = CVPixelBufferGetHeight(input)
        let outWidth = width * scale
        let outHeight = height * scale

        let output = try makeOutputBuffer(outWidth, outHeight,
                                          format: deep ? Self.deepPixelFormat : kCVPixelFormatType_32BGRA)

        CVPixelBufferLockBaseAddress(input, .readOnly)
        CVPixelBufferLockBaseAddress(output, [])
        defer {
            CVPixelBufferUnlockBaseAddress(input, .readOnly)
            CVPixelBufferUnlockBaseAddress(output, [])
        }

        guard let srcBase = CVPixelBufferGetBaseAddress(input) else {
            throw PlaybackTierError.inferenceError("CVPixelBuffer base address is nil (input)")
        }
        guard let dstBase = CVPixelBufferGetBaseAddress(output) else {
            throw PlaybackTierError.inferenceError("CVPixelBuffer base address is nil (output)")
        }
        let srcPtr = srcBase.assumingMemoryBound(to: UInt8.self)
        let dstPtr = dstBase.assumingMemoryBound(to: UInt8.self)
        let srcBPR = CVPixelBufferGetBytesPerRow(input)
        let dstBPR = CVPixelBufferGetBytesPerRow(output)

        // Single whole-frame span (MLX_PROFILE=1) — the untiled counterpart of the per-tile
        // spans above: lift → forward/eval → readout. No-op when MLX_PROFILE is unset.
        let span = MLXProfiler.shared.begin("upscale", "frame", note: "\(width)x\(height) whole-frame")

        // BGRA → NHWC RGB float32 [0, 1] for the whole frame.
        var scratch = [Float](repeating: 0, count: width * height * 3)
        if deep {
            // RGBA16 → NHWC RGB float32 [0, 1]: the same reciprocal-multiply form as the 8-bit lift.
            scratch.withUnsafeMutableBufferPointer { buf in
                let base = buf.baseAddress!
                let inv = Float(1.0 / 65535.0)
                for yy in 0 ..< height {
                    let row = (srcBase + yy * srcBPR).assumingMemoryBound(to: UInt16.self)
                    for xx in 0 ..< width {
                        let d = (yy * width + xx) * 3
                        base[d + 0] = Float(UInt16(littleEndian: row[xx * 4 + 0])) * inv
                        base[d + 1] = Float(UInt16(littleEndian: row[xx * 4 + 1])) * inv
                        base[d + 2] = Float(UInt16(littleEndian: row[xx * 4 + 2])) * inv
                    }
                }
            }
        } else {
            scratch.withUnsafeMutableBufferPointer { buf in
                let base = buf.baseAddress!
                let inv = Float(1.0 / 255.0)
                for yy in 0 ..< height {
                    let rowOff = yy * srcBPR
                    for xx in 0 ..< width {
                        let off = rowOff + xx * 4
                        let b = Float(srcPtr[off + 0]) * inv
                        let g = Float(srcPtr[off + 1]) * inv
                        let r = Float(srcPtr[off + 2]) * inv
                        let d = (yy * width + xx) * 3
                        base[d + 0] = r
                        base[d + 1] = g
                        base[d + 2] = b
                    }
                }
            }
        }

        let inArray = MLXArray(scratch, [1, height, width, 3])
        let upscaled = try forward(inArray)
        // `forward` contractually evals before returning; re-eval defensively
        // so a stray lazy graph can't zero the read-out (mlx-porting).
        MLX.eval(upscaled)

        let shape = upscaled.shape
        guard shape.count == 4, shape[0] == 1, shape[3] == 3 else {
            throw PlaybackTierError.inferenceError(
                "unexpected whole-frame output shape \(shape); want [1, H, W, 3]"
            )
        }
        let oH = shape[1]
        let oW = shape[2]
        let outFloats = upscaled.asArray(Float.self)

        let copyH = min(oH, outHeight)
        let copyW = min(oW, outWidth)
        if deep {
            // RGB float32 → RGBA16, rounded once, opaque — the 8-bit write below at 65535 steps.
            outFloats.withUnsafeBufferPointer { buf in
                let base = buf.baseAddress!
                for yy in 0 ..< copyH {
                    let row = (dstBase + yy * dstBPR).assumingMemoryBound(to: UInt16.self)
                    for xx in 0 ..< copyW {
                        let s = (yy * oW + xx) * 3
                        for c in 0 ..< 3 {
                            row[xx * 4 + c] = UInt16((max(0, min(1, base[s + c])) * 65535).rounded()).littleEndian
                        }
                        row[xx * 4 + 3] = UInt16.max
                    }
                }
            }
            MLXProfiler.shared.end(span)
            return output
        }
        outFloats.withUnsafeBufferPointer { buf in
            let base = buf.baseAddress!
            for yy in 0 ..< copyH {
                let dstRow = yy * dstBPR
                for xx in 0 ..< copyW {
                    let s = (yy * oW + xx) * 3
                    let r = max(0, min(1, base[s + 0]))
                    let g = max(0, min(1, base[s + 1]))
                    let b = max(0, min(1, base[s + 2]))
                    let d = dstRow + xx * 4
                    // ROUND, not truncate — `UInt8(_:)` discards the fraction, which is a
                    // systematic −0.5 LSB darkening across the whole frame. This path is the
                    // reference arm the tiled path is measured against, so the two must
                    // quantise the same way (WORKORDER-tile-compositor).
                    dstPtr[d + 0] = UInt8((b * 255).rounded())
                    dstPtr[d + 1] = UInt8((g * 255).rounded())
                    dstPtr[d + 2] = UInt8((r * 255).rounded())
                    dstPtr[d + 3] = 255
                }
            }
        }
        MLXProfiler.shared.end(span)

        return output
    }

    /// Allocate an output `CVPixelBuffer` (32BGRA, or `deepPixelFormat`) at the given dimensions.
    private func makeOutputBuffer(_ outWidth: Int, _ outHeight: Int, format: OSType) throws -> CVPixelBuffer {
        var outputBuffer: CVPixelBuffer?
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: format,
            kCVPixelBufferWidthKey as String: outWidth,
            kCVPixelBufferHeightKey as String: outHeight,
        ]
        let status = CVPixelBufferCreate(
            nil, outWidth, outHeight, format,
            attrs as CFDictionary, &outputBuffer
        )
        guard status == kCVReturnSuccess, let output = outputBuffer else {
            throw PlaybackTierError.inferenceError(
                "CVPixelBufferCreate failed (status=\(status))"
            )
        }
        return output
    }

    // MARK: - Tile extraction (BGRA → NHWC RGB float32)

    /// Fill `scratch` with the `windowSize × windowSize × 3` RGB float32 view of the BGRA source
    /// window starting at `(windowX, windowY)` — which may be negative or overhang the frame when a
    /// halo is in play. Pixels outside the FRAME clamp to the nearest valid pixel (matches
    /// `TileProcessor`'s edge-extension behaviour so an overhang doesn't produce a hard cut); pixels
    /// inside the frame are always read as-is, so a halo margin carries real neighbouring content.
    /// With `halo == 0` this reduces exactly to the old per-tile extraction (the clamp then only ever
    /// fires where the old tile-region clamp fired: a frame smaller than the tile).
    @inline(__always)
    private func fillWindowScratch(
        src: UnsafePointer<UInt8>,
        srcBPR: Int,
        windowX: Int, windowY: Int, windowSize: Int,
        frameW: Int, frameH: Int,
        scratch: inout [Float]
    ) {
        scratch.withUnsafeMutableBufferPointer { buf in
            let base = buf.baseAddress!
            let scale = Float(1.0 / 255.0)
            for ty in 0 ..< windowSize {
                let srcY = min(max(windowY + ty, 0), frameH - 1)
                for tx in 0 ..< windowSize {
                    let srcX = min(max(windowX + tx, 0), frameW - 1)
                    let off = srcY * srcBPR + srcX * 4
                    // BGRA → RGB.
                    let b = Float(src[off + 0]) * scale
                    let g = Float(src[off + 1]) * scale
                    let r = Float(src[off + 2]) * scale
                    let dstOff = (ty * windowSize + tx) * 3
                    base[dstOff + 0] = r
                    base[dstOff + 1] = g
                    base[dstOff + 2] = b
                }
            }
        }
    }

    /// `fillWindowScratch` for a `deepPixelFormat` source: RGBA UInt16 → RGB float32 [0, 1], same window and
    /// edge-replication rules.
    @inline(__always)
    private func fillWindowScratch16(
        src: UnsafeMutableRawPointer,
        srcBPR: Int,
        windowX: Int, windowY: Int, windowSize: Int,
        frameW: Int, frameH: Int,
        scratch: inout [Float]
    ) {
        scratch.withUnsafeMutableBufferPointer { buf in
            let base = buf.baseAddress!
            let scale = Float(1.0 / 65535.0)
            for ty in 0 ..< windowSize {
                let srcY = min(max(windowY + ty, 0), frameH - 1)
                let row = (src + srcY * srcBPR).assumingMemoryBound(to: UInt16.self)
                for tx in 0 ..< windowSize {
                    let srcX = min(max(windowX + tx, 0), frameW - 1)
                    let dstOff = (ty * windowSize + tx) * 3
                    base[dstOff + 0] = Float(UInt16(littleEndian: row[srcX * 4 + 0])) * scale
                    base[dstOff + 1] = Float(UInt16(littleEndian: row[srcX * 4 + 1])) * scale
                    base[dstOff + 2] = Float(UInt16(littleEndian: row[srcX * 4 + 2])) * scale
                }
            }
        }
    }

    // MARK: - Tile write with feathered overlap blending
}

/// Float32 accumulator for the feathered tile composite.
///
/// 🚨 **Why this replaced the previous in-place 8-bit blend** (`existing·(1−w) + new·w`
/// written straight into the destination BGRA buffer, WORKORDER-tile-compositor, measured
/// 2026-07-30). That writer had three defects, all of which put error on the
/// `tileSize − overlap` grid — i.e. exactly the tiled-refine "seam lattice":
///
///  1. **It was not a normalised weighted average.** Over-compositing tile-by-tile with only
///     the *new* tile's weight is order-dependent. Along a 1-D seam two complementary ramps
///     happen to agree, but where **four tiles meet** the weight is a product of two ramps and
///     the first-written tile keeps far too much: measured effective weights at the corner
///     were `[0.31 0.01 0.23 0.44]` where the normalised average is `[0.05 0.05 0.44 0.47]`
///     — up to **0.267 absolute weight error**, on a lattice at the write stride.
///  2. **It quantised on every write.** Overlap pixels were stamped, read back as `UInt8`,
///     blended and truncated again (`UInt8(_:)` discards the fraction, it does not round), so
///     the overlap bands carried a systematic downward bias the tile interiors did not:
///     6412 pixels of a 2048² identity forward came out **−1, every one of them negative**,
///     concentrated in the 32 px bands on the 224 grid.
///  3. **It used the destination alpha byte as an "untouched" flag.** `CVPixelBufferCreate`
///     recycles memory — measured 23 of 24 fresh 2048² allocations came back with alpha 255
///     everywhere — so from the second frame onward the first write into a feathered pixel
///     blended against **the previous frame's content** instead of stamping. An identity
///     forward on the second run differed from its input in **28% of pixels, max |Δ| 250**.
///
/// The fix is the textbook composite: accumulate `Σ(wᵢ·xᵢ)` and `Σ(wᵢ)` in float32, divide
/// once, quantise once **with rounding**. Order-independent, corner-correct, and it never
/// reads the destination buffer, so recycled memory cannot leak in.
///
/// **Bounded memory.** Tiles arrive in raster order, so only the output rows a tile row can
/// still touch need to stay open: the band is `tileSize · scale` rows, flushed and slid
/// forward as the tile rows advance. That is 63 MB for the Real-ESRGAN 4K ×4 worst case
/// (64 px tiles, 15360 px wide) and 8 MB for the SeedVR2 2048² refine — not proportional to
/// the frame.
private struct SeamAccumulator {
    let outWidth: Int
    let outHeight: Int
    let rows: Int

    /// First output row currently held in the band.
    private var y0 = 0
    /// `rows × outWidth × 3` premultiplied colour, and `rows × outWidth` weight.
    private var colour: [Float]
    private var weight: [Float]

    init(outWidth: Int, outHeight: Int, rows: Int) {
        self.outWidth = outWidth
        self.outHeight = outHeight
        self.rows = max(1, min(rows, outHeight))
        self.colour = [Float](repeating: 0, count: self.rows * outWidth * 3)
        self.weight = [Float](repeating: 0, count: self.rows * outWidth)
    }

    /// Feather profile for one axis, **half-pixel centred** so it is strictly positive and the
    /// two ramps across a seam sum to exactly 1.
    ///
    /// The previous `tx / ov` form hit exactly 0 on a tile's own border, which made the 2-D
    /// product weight 0 for *every* tile covering an image-edge pixel (0/0 under
    /// normalisation) and made the pair of ramps across a seam sum to `(ov−1)/ov` rather than
    /// 1 — a residual the weight probe measured as 0.029 even on the well-behaved 1-D seam.
    @inline(__always)
    private static func ramp(_ i: Int, _ extent: Int, _ ov: Int) -> Float {
        let lo = (Float(i) + 0.5) / Float(ov)
        let hi = (Float(extent - 1 - i) + 0.5) / Float(ov)
        return min(min(lo, 1.0), min(hi, 1.0))
    }

    /// Add one upscaled tile (NHWC RGB float32, contiguous) to the band.
    ///
    /// - Precondition: `flush(upTo: y)` has already run, so `y >= y0`, and the tile fits the
    ///   band (`th <= rows`). Both hold for raster-order tiles with `rows == tileSize·scale`.
    mutating func add(
        outFloats: [Float],
        outH: Int, outW: Int,
        x: Int, y: Int, tw: Int, th: Int,
        overlapScaled: Int
    ) {
        let ov = max(overlapScaled, 1)
        let writeH = min(outH, th)
        let writeW = min(outW, tw)
        let bandY0 = y0

        outFloats.withUnsafeBufferPointer { src in
            colour.withUnsafeMutableBufferPointer { col in
                weight.withUnsafeMutableBufferPointer { wgt in
                    let s = src.baseAddress!, c = col.baseAddress!, w = wgt.baseAddress!
                    for ty in 0 ..< writeH {
                        let dstY = y + ty
                        guard dstY < outHeight else { continue }
                        let row = dstY - bandY0
                        guard row >= 0, row < rows else { continue }
                        let wy = Self.ramp(ty, outH, ov)
                        for tx in 0 ..< writeW {
                            let dstX = x + tx
                            guard dstX < outWidth else { continue }
                            let sOff = (ty * outW + tx) * 3
                            let r = max(0, min(1, s[sOff + 0]))
                            let g = max(0, min(1, s[sOff + 1]))
                            let b = max(0, min(1, s[sOff + 2]))
                            let wt = Self.ramp(tx, outW, ov) * wy
                            let p = row * outWidth + dstX
                            c[p * 3 + 0] += r * wt
                            c[p * 3 + 1] += g * wt
                            c[p * 3 + 2] += b * wt
                            w[p] += wt
                        }
                    }
                }
            }
        }
    }

    /// Resolve and write every band row below `yEnd` into the destination, then slide the band
    /// so it starts at `yEnd`. Rows no tile ever covered are written as opaque black rather
    /// than left alone — the destination buffer's prior contents are never a valid answer.
    mutating func flush(upTo yEnd: Int, dst rawDst: UnsafeMutableRawPointer, dstBPR: Int, deep: Bool = false) {
        let end = min(yEnd, outHeight)
        guard end > y0 else { return }
        let n = end - y0
        let dst = rawDst.assumingMemoryBound(to: UInt8.self)

        colour.withUnsafeMutableBufferPointer { col in
            weight.withUnsafeMutableBufferPointer { wgt in
                let c = col.baseAddress!, w = wgt.baseAddress!
                if deep {
                    // The same divide-once, round-once resolve, into RGBA16 (`deepPixelFormat`).
                    for row in 0 ..< min(n, rows) {
                        let out = (rawDst + (y0 + row) * dstBPR).assumingMemoryBound(to: UInt16.self)
                        for x in 0 ..< outWidth {
                            let p = row * outWidth + x
                            let sw = w[p]
                            for ch in 0 ..< 3 {
                                let v = sw > 0 ? (max(0, min(1, c[p * 3 + ch] / sw)) * 65535).rounded() : 0
                                out[x * 4 + ch] = UInt16(v).littleEndian
                            }
                            out[x * 4 + 3] = UInt16.max
                        }
                    }
                } else {
                    for row in 0 ..< min(n, rows) {
                        let dstRow = (y0 + row) * dstBPR
                        for x in 0 ..< outWidth {
                            let p = row * outWidth + x
                            let sw = w[p]
                            let d = dstRow + x * 4
                            if sw > 0 {
                                let inv = 1.0 / sw
                                // Divide once, quantise once, ROUND — truncating here is what put a
                                // one-sided −1 LSB bias in the overlap bands.
                                let r = (max(0, min(1, c[p * 3 + 0] * inv)) * 255).rounded()
                                let g = (max(0, min(1, c[p * 3 + 1] * inv)) * 255).rounded()
                                let b = (max(0, min(1, c[p * 3 + 2] * inv)) * 255).rounded()
                                dst[d + 0] = UInt8(b); dst[d + 1] = UInt8(g); dst[d + 2] = UInt8(r)
                            } else {
                                dst[d + 0] = 0; dst[d + 1] = 0; dst[d + 2] = 0
                            }
                            dst[d + 3] = 255
                        }
                    }
                }
                // Slide: keep the still-open rows, zero what they vacate.
                let keep = rows - min(n, rows)
                if keep > 0 {
                    memmove(c, c + n * outWidth * 3, keep * outWidth * 3 * MemoryLayout<Float>.size)
                    memmove(w, w + n * outWidth, keep * outWidth * MemoryLayout<Float>.size)
                }
                memset(c + keep * outWidth * 3, 0, (rows - keep) * outWidth * 3 * MemoryLayout<Float>.size)
                memset(w + keep * outWidth, 0, (rows - keep) * outWidth * MemoryLayout<Float>.size)
            }
        }
        y0 = end
    }
}
