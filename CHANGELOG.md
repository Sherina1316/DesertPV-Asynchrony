# Changelog

## v24.22.5-r4-dual-stage-reference-fusion

- Rebuilt strictly from the v24.22.2 R4 baseline; R5/R6 changes are not included.
- Separated authoritative 2024 parcel presence from strong supervised-training cores. Any positive SHP overlap remains a certain 2024 PV parcel, while only genuine interior cores can enter formal train/calibration/test roles.
- Added an independent SHP-internal 2011–2024 build-out branch using annual six-band Landsat and each parcel's own 2024 spectral state.
- Retained the complete R4 full-domain 2011–2025 detector as a separate branch.
- Fused the two branches only in Stage 8; outside-SHP behavior, 2025 remote discovery, ecological evaluation, mapping and reports retain their existing interfaces.
- Added local background centring, robust spectral similarity, distributed parcel support, partial-buildout reconstruction, temporal persistence and abrupt-onset detection.
- Prevented the 2024 endpoint from being copied blindly into earlier years.
- Added five Stage-8 diagnostic raster families for the two branches and their fusion.
- Added eight targeted regression/integration tests; the complete suite reports 132 passed.

## v24.22.2-mixed-parcel-optimized

- Preserved the 21-stage workflow, output schema, directory hierarchy, file names, maps, and report layout.
- Added distributed mixed-pixel parcel diagnostics using support fraction, dilation coverage, 3×3 sector coverage, boundary/interior support, and evidence quantiles.
- Added a parcel-completion path for 30 m cells that each contain only a fraction of PV panels, while retaining annual Landsat-evidence and multi-year persistence gates.
- Prevented a localized high-confidence core or partial construction strip from filling the complete 2024 parcel prematurely.
- Added whole-object historical onset reconstruction for parcels visible before their pixelwise probability crosses the original threshold, addressing early omissions such as Unit 107.
- Added a mixed-pixel anchor rescue route so a confirmed 2024 parcel with usable imagery is kept on an `anchor_guided_candidate` spatial route instead of being reduced to area-only output, addressing blank maps such as Unit 35.
- Added soft positive prototype construction for small or mixed 2024 anchors and robust scale borrowing when the anchor prototype is nearly homogeneous.
- Protected distributed mixed-pixel 2024 parcels during boundary refinement, but only for pixels with current weak PV support; genuinely unsupported sand-like boundary pixels remain removable.
- Retained the v24.22.1 thin-ring suppression, small-block protection, historical object evidence, and 2025 annular-change rejection.
- Integrated the selected-unit calibration fallback and Stage-3/Stage-4 resume-contract repairs from the R3 hotfix, strictly limited to relaxed selected-unit smoke tests.
- Added seven mixed-parcel regression tests. All 124 collected tests passed when run in split batches; the monolithic suite reaches completion of all test assertions but leaves a pre-existing worker process alive in this container, so the one-shot command can exceed the external timeout after reporting the final passes.

## v24.22.1-detail-optimized

- Preserved the existing workflow, output paths, figures, and `24.22.0` output schema.
- Added image-evidence-based historical enhancement inside trusted 2024 reference objects.
- Downweighted the deterministic 2024 endpoint during installation-year inference.
- Added object-level early-presence detection and adaptive seed rules for small PV blocks.
- Protected valid small manual reference components during boundary refinement.
- Suppressed thin wrap-around boundary additions and removed final closing that could recreate complete rings.
- Required multiple negative cues before removing a 2024 boundary pixel.
- Added conservative thin-shell stabilization for weak panel edges while preserving interior phased buildout.
- Rejected annular 2025 boundary displacement without blocking patchy or remote expansion.
- Added checkpoint revision invalidation so original v24.22 Stage-7/8 caches are recalculated.
- Added eight focused regression tests for the revised behavior.

## v24.22.0

- Added direct 2024→2025 multiband change evidence.
- Added terminal novelty score.
- Added pre-existing-state risk and prior dark-persistence measures.
- Added terminal-break dominance relative to earlier annual fluctuations.
- Added persistent non-PV risk.
- Applied novelty gates to both legacy remote and full-domain remote pathways.
- Required object-level novelty and direct-change fractions.
- Restricted gap bridging to low pre-existing-risk support.
- Added six Stage-7 diagnostic raster families.
- Added per-object rejection reasons and novelty metrics.
- Preserved full-domain, distance-independent discovery of genuinely new 2025 sites.
