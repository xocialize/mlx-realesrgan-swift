# Efficiency Adoption Brief — `mlx-realesrgan-swift` (Real-ESRGAN, `imageUpscale`)

> **For a session-specific agent.** Self-contained: audit + tasks to adopt the MLXEngine
> library-efficiency contract (engine 1.14.0 / 0.15.0). Load the `mlx-swift-integration` skill and read
> `references/package-efficiency.md` (incl. "Gotchas & measurement") + `references/memory-harness.md`
> first. Brief shape follows the BiRefNet/LTX template. Audited + executed 2026-06-30.

## Why this one matters
Real-ESRGAN is the **upscale** link of the ForgeOptimizer chain (IQA → restore → upscale → colorize,
alongside BiRefNet matting). Single-component but **tiled** (64² tiles, feathered seams) — the footprint
is the tile working set + the 4× output buffer, almost all activation. With the 1.14 split it co-resides
with the rest of the chain on weights while sharing ONE activation reserve.

## Package at a glance
- **Wrapper:** `RealESRGANUpscalePackage` (`Sources/MLXRealESRGAN/`), core `SRVGGNetCompact_Playback` (`RealESRGANMLX`). Capability `imageUpscale`. **Single-component**, tiled 4× SR.
- **One declared footprint (fp32):** three vendored SRVGGNetCompact checkpoints (`general` default / `generalDenoise` / `anime`) — all in the core bundle, **no download**; ~1-2 MB weights each.
- **Home:** `mlxengine-image/PROD/`.

## Engine dependency status
- `Package.swift` pinned `mlx-engine-swift` **`from: "0.10.0"`**, resolved 0.10.0. **P0 = `swift package update`** → 0.15.0 (the pin admits it; no manifest edit). Done.

## Audit vs. the four levers

| Lever | State | Finding | Priority |
|---|---|---|---|
| Engine dep | 🟡→🟢 | resolved 0.10.0; re-resolved to 0.15.0 | **P0 (done)** |
| 1. Split footprint | ❌→🟢 | was flat `QuantFootprint(.fp32, 1.0 GB)` — and **under-declared** the measured ~2.2 GB tiled peak. Now split | **P1 (done)** |
| QuantConfigured | ❌→🟢 | config had `variant`, no `quant`; added `quant == .fp32` + conformance | **P1 (done)** |
| 2. mmap/lazy load | 🟢 | vendored bundle weights; core lazy-loads tensors on first upscale; weights are ~MB | note only |
| 3. Per-stage evict | ➖ | single-component — no multi-stage pipeline | n/a |
| 4. BudgetAware | ➖ | fp32 is the validated runtime; no memory/quality dtype lever | defer |

## P0 — engine update (done)
`swift package update mlx-engine-swift` → resolved **0.10.0 → 0.15.0**. Builds clean against the 1.14 contract.

## QuantConfigured (done)
`RealESRGANConfiguration` conforms via an extension exposing `quant == .fp32` (all vendored checkpoints
run fp32 — the single declared footprint quant), so the governor charges the matching `QuantFootprint`.

## P1 — split declared (done)
The forward is **tiled**, so the working set is tile-bounded (it does NOT scale with full input area).
Re-measured via the new `realesrgan-smoke` target through the real `MLXServeEngine` (register → run):

| Input → output | floor (resident) | peak | activation | run |
|---|---|---|---|---|
| 512²→2048² | 3 MB | 2170 MB | ~2.2 GB | 0.9 s |
| 1024²→4096² | 3 MB | 1807 MB (lower — tile-bounded) | ~1.8 GB | 2.1 s |

The 1024² input peaked *lower* than 512² — confirming the **tile (not input-area) drives activation**;
the 512²→2048² case is the worst-case envelope. Declared:
```swift
QuantFootprint(.fp32, residentBytes: 32_000_000, peakActivationBytes: 2_200_000_000)
```
**Engine charge:** the flat fp32 1.0 GB (which under-declared the ~2.2 GB tiled peak) becomes a **32 MB**
resident floor plus a **shared** ~2.2 GB transient reserve (one across residents). Output validated
non-uniform (luma 0…1.0, ×4 applied). Regression tests added (`splitFootprintDeclared`, `quantConfigured`).

## Already good (don't regress)
- Vendored-weights load (no download); `appliedScale` reporting; sub-native scale post-downsample (BRIDGE-029); rawBGRA8 in/out.

## Deferred — P2 (n/a single-component), P3 (n/a), P4 (BudgetAware: fp32 is the runtime, no dtype lever).

## Definition of done
- [x] `swift package update` → engine 0.15.0.
- [x] Config conforms to `QuantConfigured` (fp32).
- [x] Split declared, re-measured via `realesrgan-smoke` (512²/1024² envelopes), provenance comments kept; tile noted as the activation driver.
- [x] Smoke + offline tests green; valid non-uniform output captured.
- [x] BudgetAware deferred (note).
- [x] Update the Real-ESRGAN row in `mlx-engine-swift/docs/model-registry.md`: Eff ✅, Eng 0.15.0.

## Outcome (executed 2026-06-30)
P0+QuantConfigured+P1 done as above. Notable finding: activation is **tile-bounded** (1024² input peaked
below 512²), and the old flat 1.0 GB **under-declared** the ~2.2 GB tiled peak. The split right-sizes the
charge and frees it into the shared reserve. Tiling behavior unchanged (P2 instruction honored).
