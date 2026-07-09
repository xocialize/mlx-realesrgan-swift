// CancellationTests.swift — Real-ESRGAN through the engine's CAN gate (offline, no MLX kernels).
// CAN-1/2 drive the real run() pre-cancelled (the entry checkpoint fires before notLoaded
// validation or image decode); CAN-3 is the document of record for the checkpoint cadence:
// the core's tile driver checks cancellation once per 64² tile
// (RealESRGANMLX/MLXTileProcessor.process, top of the tile loop) and the wrapper/upscale
// rethrow the CancellationError unchanged (SRVGGNetCompact_Playback.upscale explicitly
// exempts CancellationError from the PlaybackTierError wrap).

import Foundation
import MLXServeConformance
import MLXToolKit
import XCTest
@testable import MLXRealESRGAN

final class CancellationTests: XCTestCase {

    // MARK: - CAN-1 / CAN-2 — pre-cancelled run() propagation + classification

    func testCANGatePreCancelledRun() async {
        // Stub config; construction is cheap (C13) and the entry checkpoint throws before
        // validation, decode, or the (bundled) weights are touched, so this is offline-safe.
        let package = RealESRGANUpscalePackage(configuration: RealESRGANConfiguration())
        let report = await CancellationConformance.checkRun(
            package: package,
            request: ImageUpscaleRequest(image: Image(format: .png, data: Data())))
        XCTAssertTrue(report.passed, report.summary)
    }

    // MARK: - CAN-3 — checkpoint-cadence declaration (the document of record)

    func testCANCadenceDeclaration() {
        // peakActivationBytes 5.24 GB ≥ 2 GB ⇒ long-run implied; the sub-second
        // exemption is not available even though a single tile's forward is fast.
        XCTAssertTrue(CancellationConformance.longRunImplied(by: RealESRGANUpscalePackage.manifest))

        let report = CancellationConformance.checkCadence(
            manifest: RealESRGANUpscalePackage.manifest,
            posture: .cadence([
                // The tiled forward checks Task.checkCancellation once per 64² tile
                // (MLXTileProcessor.process, top of the tile loop). "chunk" = one tile —
                // the finest real seam of the run. The ≤1080p whole-frame fast path is a
                // single monolithic MLX eval; there the entry checkpoint is the only seam,
                // which is honest: that path completes in well under a second.
                .init(phase: .upsample, unit: .chunk),
            ]))
        XCTAssertTrue(report.passed, report.summary)
    }
}
