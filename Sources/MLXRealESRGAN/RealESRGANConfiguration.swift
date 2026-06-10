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

/// Init-time configuration for `RealESRGANUpscalePackage` (C9). All variants are vendored in the
/// core package bundle, so there is no models-root/download concern.
public struct RealESRGANConfiguration: PackageConfiguration {
    public var variant: RealESRGANVariant

    public init(variant: RealESRGANVariant = .general) {
        self.variant = variant
    }
}
