// SRVGGNetCompact_Playback.swift
//
// Role: Concrete `PlaybackTier` backed by the vendored SRVGGNetCompact
//       MLX-Swift port (Task #28). Three variants — general, generalWDN,
//       anime — each load from a distinct `Resources/realesr_*_x4.safetensors`
//       bundle.
//
// Plan ref: Forge-CodingPlan-v1.0.md §C (playback A/B baseline) / §C.5
// ADR:      Docs/ADRs/0006-phase-c-unfreeze-efrlfn.md §"Ship criterion"
// Upstream: https://github.com/xinntao/Real-ESRGAN (BSD-3-Clause)
// Weights:  Resources/realesr_{general,general_wdn,anime}_x4.safetensors
//
// Tile shape: 128 / 8 as of 0.7.0. It was 64 / 8, matching upstream
// `inference_realesrgan_video.py`, on the reasoning that "a coarser grid
// would lose more to the seam-blend overlap than it gains in per-tile
// throughput." 🚨 **That reasoning was wrong on both halves and is kept
// here as the record of it.** Overlap is absolute, not a ratio, so a
// coarser grid loses nothing to seam blending — and 64 was the SLOWEST of
// {64,128,256} on both the MLX-GPU and CoreAI-ANE backends. Measured:
// 4.65× faster end to end, with the seam z-test detecting a seam at 64
// and none at 128. See `inputTileSize` for both receipts.
//
// Lazy init: weights load on first `upscale(_:)` behind an `NSLock` so
// the protocol-conformance / surface-query path stays free.

import CoreVideo
import Foundation
import MLX
import MLXNN

/// SRVGGNetCompact playback tier — MLX-Swift, BSD-3-Clause.
///
/// Marked `@unchecked Sendable` because the MLX module + lock are mutated
/// across the `async` boundary; the lock guards the only crossing point.
public final class SRVGGNetCompact_Playback: PlaybackTier, @unchecked Sendable {

    /// Vendored SRVGGNetCompact variant.
    public enum Variant: String, Sendable {
        case general        // realesr_general_x4.safetensors    (num_conv=32, ~1.21 M)
        case generalWDN     // realesr_general_wdn_x4.safetensors (num_conv=32, ~1.21 M)
        case anime          // realesr_anime_x4.safetensors      (num_conv=16, ~0.62 M)

        /// Resource stem (no extension) of the vendored safetensors file.
        public var safetensorsName: String {
            switch self {
            case .general:    return "realesr_general_x4"
            case .generalWDN: return "realesr_general_wdn_x4"
            case .anime:      return "realesr_anime_x4"
            }
        }

        /// The vendored checkpoint's on-disk URL inside the package bundle. Present in every
        /// correct build product (the weights ship with the code — no download, ever); `nil`
        /// only if the resource was stripped from the bundle.
        public var bundledWeightsURL: URL? {
            Bundle.module.url(forResource: safetensorsName, withExtension: "safetensors")
        }

        /// Stable `PlaybackTier.name` identifier per variant.
        var tierName: String {
            switch self {
            case .general:    return "srvggnet-general-x4"
            case .generalWDN: return "srvggnet-general-wdn-x4"
            case .anime:      return "srvggnet-anime-x4"
            }
        }
    }

    // MARK: - PlaybackTier surface

    public let name: String
    public let scaleFactor: Int
    /// Tile geometry for the tiled path. **The default is 128 as of 0.7.0** — the previous 64 was
    /// measurably the worst of the three sizes on BOTH backends (`GAP-PROGRAM.md` V13, 1080p→×4 per
    /// frame):
    ///
    ///     MLX-GPU fp32   t64 1440 ms · t128  808 ms · t256 878 ms
    ///     CoreAI-ANE     t64 1053 ms · t128  852 ms · t256 953 ms
    ///
    /// 🔑 **t128 is the optimum on both, and t64 cost ~1.8× on the shipping backend.** Small tiles lose
    /// to per-dispatch overhead; very large ones lose to working-set pressure. End-to-end through the
    /// app's shipping path the win is larger than the per-frame ladder suggests — 2048²→8192² ×4 went
    /// **7.68 s → 1.65 s (4.65×)**, because tile count falls quadratically (1369 → 324 positions).
    ///
    /// ⚠️ **`tileOverlap` is an ABSOLUTE pixel count and must NOT be scaled with the tile.** What the
    /// overlap has to cover is the network's receptive field, which is a fixed number of pixels and does
    /// not grow when the tile does. So 8 px at t128 is exactly as much margin as 8 px at t64 — while
    /// there are *fewer* seams — which is why enlarging the tile *reduces* total seam artefact rather
    /// than increasing it, despite the smaller overlap *ratio*.
    ///
    /// ✅ **That is measured, not reasoned.** Seam test on `SNS_graphic` 8192², restricted to flat blocks
    /// (the only regions where a 1-LSB discontinuity is visible), each arm scored at its own stride and
    /// z-tested against a null sweep of ~166 off-strides:
    ///
    ///     t64  @ stride 224:  z = +1.50, boundary/interior 1.384  → SEAM (null 95th pct +0.84)
    ///     t128 @ stride 480:  z = +0.46, boundary/interior 1.056  → none above noise
    ///
    /// The arms agree numerically (p99 |Δ| = 2/255; 186 of 67 M pixels exceed 32), and what disagreement
    /// there is concentrates 1.54× on t64's own 224 lattice — i.e. t128 differs from t64 chiefly *at the
    /// seams t64 has*. Faster and cleaner, not a trade.
    ///
    /// ⚠️ **Do not "improve" this by raising `tileOverlap` alongside the tile.** Overlap ratio is the
    /// wrong intuition here and costs throughput for no margin.
    public let inputTileSize: Int
    public let tileOverlap: Int

    public var inputResolution: (width: Int, height: Int) {
        (inputTileSize, inputTileSize)
    }

    public var outputResolution: (width: Int, height: Int) {
        (inputTileSize * scaleFactor, inputTileSize * scaleFactor)
    }

    /// The vendored variant this instance wraps. Exposed for tests / logs.
    public let variant: Variant

    // MARK: - Internals

    private let model: SRVGGNetCompact
    private let tileProcessor: MLXTileProcessor
    private let weightsURL: URL
    private let loadLock = NSLock()
    private var weightsLoaded = false

    /// `mx.compile`-traced forward, built lazily once weights are loaded.
    /// Same rationale as `EfRLFN_Playback`: kept identical so the Phase C.4
    /// A/B compares the two backends under the same inference strategy.
    private var compiledForward: (@Sendable (MLXArray) -> MLXArray)?

    /// Whole-frame fast-path ceiling, in INPUT pixels. SRVGGNetCompact is fully convolutional (no
    /// internal downsampling), so it shares EfRLFN's whole-frame treatment; inputs above this fall back
    /// to tiling at `inputTileSize` (128 by default).
    ///
    /// 🔑 **Injectable, and the default deliberately stays permissive: whole-frame is the QUALITY path
    /// and the capability must not be capped in the package.** Tiling bounds memory but introduces
    /// seams; whole-frame has none. A 128 GB machine should be allowed to take it at sizes a 16 GB
    /// machine cannot, so **the host gates this per machine — the package does not decide for it.**
    ///
    /// ⚠️ **Know what the default costs before leaving it alone.** Measured (`GAP-PROGRAM.md` V12,
    /// process `phys_footprint`, ×4): **3.22 GB at SD** but **21.24 GB at 1920×1080** and 24.64 GB at
    /// 1080×1920 — because this net runs at its native ×4 internally, so a 1080p input builds a
    /// 7680×4320 intermediate. ⚠️ And note `processAdaptive` tests `<=`, so **1920×1080 lands exactly on
    /// the default ceiling and takes the whole-frame path** — the most common video resolution is on the
    /// expensive side by one comparison. That is *correct* under a quality-first policy and *fatal* on a
    /// 16 GB machine, which is precisely why the choice belongs to the host.
    ///
    /// Suggested host policy: derive from the memory budget, preferring whole-frame wherever it fits.
    /// Pass `0` to force tiling always.
    public let wholeFrameMaxPixels: Int

    // MARK: - Init

    /// Initialise an SRVGGNetCompact playback tier.
    ///
    /// The three vendored v3 checkpoints all ship at x4; the architecture
    /// itself supports any integer scale ≥ 1, but mismatched weight + scale
    /// pairs would fail at safetensors load time. We pin scale=4 here for
    /// the same forward-compat reason `EfRLFN_Playback` does.
    /// - Parameter wholeFrameMaxPixels: input-pixel ceiling for the single-pass path. Defaults to
    ///   `1920 * 1080`, unchanged from before this became injectable, so existing callers keep their
    ///   exact behaviour. See the property's note for what that default costs.
    /// - Parameters:
    ///   - inputTileSize: **changed 64 → 128 in 0.7.0.** Faster (4.65× end-to-end at 8192² output) and
    ///     seam-free where 64 was not; see the property for both measurements. A caller that needs the
    ///     old geometry passes `inputTileSize: 64` explicitly.
    ///   - tileOverlap: absolute pixels, unchanged at 8 — and deliberately NOT scaled with the tile.
    public init(variant: Variant, wholeFrameMaxPixels: Int = 1920 * 1080,
                inputTileSize: Int = 128, tileOverlap: Int = 8) throws {
        self.variant = variant
        self.name = variant.tierName
        self.scaleFactor = 4
        self.wholeFrameMaxPixels = wholeFrameMaxPixels
        self.inputTileSize = inputTileSize
        self.tileOverlap = tileOverlap

        guard let url = Bundle.module.url(
            forResource: variant.safetensorsName,
            withExtension: "safetensors"
        ) else {
            throw PlaybackTierError.weightsNotFound(variant.safetensorsName)
        }
        self.weightsURL = url

        switch variant {
        case .general:    self.model = SRVGGNetCompact.general()
        case .generalWDN: self.model = SRVGGNetCompact.generalWDN()
        case .anime:      self.model = SRVGGNetCompact.anime()
        }

        self.tileProcessor = MLXTileProcessor(
            tileSize: inputTileSize,
            overlap: tileOverlap,
            scale: scaleFactor
        )
    }

    // MARK: - PlaybackTier impl

    public func upscale(_ buffer: CVPixelBuffer) async throws -> CVPixelBuffer {
        let run = try ensureReady()
        do {
            return try tileProcessor.processAdaptive(
                buffer,
                wholeFrameMaxPixels: wholeFrameMaxPixels
            ) { tile in
                let y = run(tile)
                MLX.eval(y)
                return y
            }
        } catch let err as PlaybackTierError {
            throw err
        } catch is CancellationError {
            // CAN-2: never launder a cancellation — the per-tile checkpoint's
            // CancellationError must reach the engine unchanged.
            throw CancellationError()
        } catch {
            throw PlaybackTierError.inferenceError(String(describing: error))
        }
    }

    // MARK: - Weights + compile

    /// Load weights (once) and build the compiled forward (once). See
    /// `EfRLFN_Playback.ensureReady` for the rationale; kept identical so the
    /// A/B runs both backends through the same inference strategy.
    private func ensureReady() throws -> @Sendable (MLXArray) -> MLXArray {
        loadLock.lock()
        defer { loadLock.unlock() }
        if let f = compiledForward { return f }
        if !weightsLoaded {
            do {
                try model.loadWeights(from: weightsURL)
                weightsLoaded = true
            } catch let err as SRVGGNetCompactError {
                throw PlaybackTierError.modelLoadFailed(String(describing: err))
            } catch {
                throw PlaybackTierError.modelLoadFailed(String(describing: error))
            }
        }
        let m = model
        let f = compile { x in m(x) }
        compiledForward = f
        return f
    }
}
