import Foundation
import MLXToolKit
import RealESRGANMLX

/// Which vendored SRVGGNetCompact checkpoint to load (all bundled in the core — no download).
public enum RealESRGANVariant: String, Codable, Sendable, CaseIterable {
    /// General-purpose x4 (the forge ADR-0008 shipped winner). Default.
    case general
    /// General-purpose x4 with denoising (WDN).
    case generalDenoise
    /// Anime-optimized x4 (smaller, faster).
    case anime

    var coreVariant: SRVGGNetCompact_Playback.Variant {
        switch self {
        case .general: return .general
        case .generalDenoise: return .generalWDN
        case .anime: return .anime
        }
    }
}

/// Init-time configuration for `RealESRGANUpscalePackage` (C9). All variants are **vendored in
/// the core package bundle** — the weights (~2 MB each) ship with the code, so a fresh machine
/// never downloads anything and `load()` never touches the network.
///
/// Materialization posture (engine ≥ 0.19.0): `ModelStorable` so the engine can stamp its store
/// root (MAT-1), and `WeightPrewarming` over the bundle URLs — always-present prewarm paths are
/// the engine's sanctioned signal that a bundled package is ready, so `needsDownload` reads
/// `false` on a fresh install instead of waiting forever for a store marker. `WeightSourcing`
/// is deliberately NOT declared: there is no fresh-machine network source, and declaring one
/// would either lie about the missing set (MAT-4) or force a download of weights already in the
/// binary. (The MAT gate has no bundled-weights vocabulary yet — engine-side gap, filed.)
public struct RealESRGANConfiguration: PackageConfiguration, ModelStorable {
    public var variant: RealESRGANVariant
    /// Engine-chosen models root. Stamped by the engine from its `ModelStore`; unused today
    /// (weights are bundled) but keeps the config store-addressable (MAT-1). Environment-
    /// specific → excluded from `Codable`.
    public var modelsRootDirectory: URL?

    public init(variant: RealESRGANVariant = .general, modelsRootDirectory: URL? = nil) {
        self.variant = variant
        self.modelsRootDirectory = modelsRootDirectory
    }

    private enum CodingKeys: String, CodingKey {
        case variant
    }
}

/// Bundled weights read as permanently prewarm-able: the paths always exist, so the engine's
/// `needsDownload` heuristic short-circuits to `false` (nothing to fetch) and the prewarmer
/// pages the ~2 MB checkpoint before first load.
extension RealESRGANConfiguration: WeightPrewarming {
    public var prewarmPaths: [URL] {
        [variant.coreVariant.bundledWeightsURL].compactMap { $0 }
    }
}

/// `QuantConfigured` (engine 1.14): all vendored SRVGGNetCompact checkpoints run at fp32 (the single
/// declared footprint quant), so the governor charges the matching `QuantFootprint(.fp32, …)` rather
/// than the largest-that-fits fallback. One quant today; the protocol keeps it honest if a quantized
/// variant is added.
extension RealESRGANConfiguration: QuantConfigured {
    public var quant: Quant { .fp32 }
}
