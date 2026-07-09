// MaterializationTests.swift — Real-ESRGAN's materialization posture, offline.
//
// This package is BUNDLED-WEIGHTS: every variant's checkpoint (~2 MB) is vendored in the core
// bundle, so a fresh machine has the weights the moment the binary exists — there is nothing to
// materialize. Since engine 0.24.0 (contract 1.17) that posture is first-class MAT vocabulary:
//   • MAT-1 (ModelStorable) — the engine can stamp its store root.
//   • MAT-2..4 (BundledWeightSourcing) — the selected variant's checkpoint is declared as a
//     bundled source; the gate verifies it PRESENT on a fresh machine (the inverse of
//     WeightSourcing's required-missing network rule), and `needsDownload` reads false.
//   • WeightSourcing stays undeclared by design: there is no fresh-machine network source.
//   • WeightPrewarming keeps its own job — cold-start page-in of the checkpoint before load().

import Foundation
import MLXServeConformance
import MLXServeCore
import MLXToolKit
import XCTest
@testable import MLXRealESRGAN

final class MaterializationTests: XCTestCase {

    /// The FULL MAT gate passes per selectable variant (contract 1.17 bundled vocabulary):
    /// MAT-1 store-stampable, MAT-2 bundled source declared, MAT-3 hygiene, MAT-4 the vendored
    /// checkpoint verified PRESENT on a fresh configuration.
    func testFullMATGatePassesPerVariant() {
        for variant in RealESRGANVariant.allCases {
            let report = MaterializationConformance.check(
                freshConfiguration: RealESRGANConfiguration(variant: variant))
            XCTAssertTrue(report.passed, "\(variant):\n\(report.summary)")
        }
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
