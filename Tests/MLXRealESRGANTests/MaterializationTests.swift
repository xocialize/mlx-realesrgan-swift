// MaterializationTests.swift — Real-ESRGAN's materialization posture, offline.
//
// This package is BUNDLED-WEIGHTS: every variant's checkpoint (~2 MB) is vendored in the core
// bundle, so a fresh machine has the weights the moment the binary exists — there is nothing to
// materialize. That maps onto the engine contract as:
//   • MAT-1 (ModelStorable) — conformed; the engine can stamp its store root.
//   • WeightPrewarming over the bundle URLs — always-present paths, so `needsDownload` reads
//     false on a fresh install (the store-marker heuristic would otherwise read "needs
//     download" forever for a package that never downloads).
//   • MAT-2..5 (WeightSourcing) — deliberately NOT declared: there is no fresh-machine network
//     source, and a declared source would either report a dishonest missing set (MAT-4) or
//     force a pointless download of weights already in the binary. The MAT gate lacks
//     bundled-weights vocabulary today; when the engine grows it, revisit.

import Foundation
import MLXServeConformance
import MLXServeCore
import MLXToolKit
import XCTest
@testable import MLXRealESRGAN

final class MaterializationTests: XCTestCase {

    /// MAT-1: the engine can stamp the store root. (The full report is expected NOT to pass —
    /// MAT-2 requires a WeightSourcing declaration this bundled package deliberately omits.)
    func testMAT1ModelStorable() {
        let report = MaterializationConformance.check(
            freshConfiguration: RealESRGANConfiguration())
        guard let mat1 = report.checks.first(where: { $0.name.hasPrefix("MAT-1") }) else {
            return XCTFail("no MAT-1 check in report:\n\(report.summary)")
        }
        XCTAssertTrue(mat1.passed, mat1.note)
    }

    /// Every selectable variant's checkpoint is actually IN the built bundle — the invariant the
    /// whole bundled posture rests on (a stripped resource must fail here, not at first upscale).
    func testEveryVariantBundlesItsWeights() {
        for variant in RealESRGANVariant.allCases {
            let url = variant.coreVariant.bundledWeightsURL
            XCTAssertNotNil(url, "\(variant) missing from bundle")
            if let url {
                XCTAssertTrue(FileManager.default.fileExists(atPath: url.path), url.path)
            }
        }
    }

    /// Prewarm paths are non-empty and exist on a FRESH configuration (no store, no explicit
    /// dirs) — this is what makes the engine's `needsDownload` report false for bundled weights.
    func testFreshConfigurationPrewarmsFromBundle() {
        for variant in RealESRGANVariant.allCases {
            let paths = RealESRGANConfiguration(variant: variant).prewarmPaths
            XCTAssertFalse(paths.isEmpty, "\(variant): empty prewarm paths")
            XCTAssertTrue(
                paths.allSatisfy { FileManager.default.fileExists(atPath: $0.path) },
                "\(variant): prewarm path missing on a fresh configuration")
        }
    }

    /// The end-to-end G6 symptom, through the REAL engine: a freshly registered bundled package
    /// must NOT read as needing a download (pre-retrofit, the store-marker heuristic reported
    /// `true` forever and consumers routed the fast-upscale tier to a download UI that never
    /// completes). Register is offline — no weights load until `prepare`/`run`.
    func testEngineNeedsDownloadIsFalseOnFreshRegistration() async throws {
        let engine = MLXServeEngine()
        _ = try await engine.register(
            RealESRGANUpscalePackage.registration, configuration: RealESRGANConfiguration())
        let needs = await engine.needsDownload(.imageUpscale)
        XCTAssertFalse(needs, "bundled weights present, nothing to download")
    }

    func testCodableRoundTrip() throws {
        let cfg = RealESRGANConfiguration(
            variant: .anime, modelsRootDirectory: URL(fileURLWithPath: "/x"))
        let decoded = try JSONDecoder().decode(RealESRGANConfiguration.self,
                                               from: JSONEncoder().encode(cfg))
        XCTAssertEqual(decoded.variant, .anime)
        XCTAssertNil(decoded.modelsRootDirectory)   // environment-specific, never encoded
    }
}
