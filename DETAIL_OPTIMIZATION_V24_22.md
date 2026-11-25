# v24.22.1 Detail Algorithm Optimization

## Scope

This revision changes only the detailed PV reconstruction logic in Stages 7 and 8. The 21-stage workflow, directory hierarchy, raster/table names, report structure, plotting code, and public output schema remain unchanged.

The optimization addresses four observed failure modes:

1. A 2024 manually mapped PV polygon is correct, but weak historical pixels inside the same physical PV object are assigned an unrealistically late installation year.
2. Small manually interpreted PV blocks are lost because object-size and seed-count rules are too strict.
3. A slightly oversized 2024 polygon creates an artificial complete outer-ring change instead of a limited boundary correction.
4. A complete narrow ring around a 2024 component is incorrectly interpreted as 2025 expansion.

## Algorithm changes

### 1. Historical reconstruction inside the 2024 reference

- The forced 2024 endpoint is downweighted in installation-year likelihood calculations and is no longer treated as equivalent to an independently observed annual image.
- Historical presence is evaluated jointly at pixel and connected-object levels.
- Moderate but persistent early-year evidence can support an installation year from 2011 onward.
- Actual annual Landsat-derived evidence remains mandatory: PV-state score, positive prototype similarity, PV fraction, local darkness, sand/confuser risk, and temporal persistence are combined. The 2024 vector is not copied backward.
- Object-level evidence recovers weak panel edges only when a supported internal core exists.

### 2. Small PV block protection

- Small 2024 reference components are retained during boundary refinement.
- Historical seed requirements adapt to component area; a small, spectrally supported block is not rejected merely because it contains fewer than two high-confidence pixels.
- The protection applies to valid manually interpreted components, not to arbitrary remote dark objects.

### 3. Thin-ring boundary stabilization

- 2024 additions are screened for annular geometry using boundary-contact fraction, distance to the reference, and thinness.
- A narrow addition wrapping most of a component is treated as likely boundary displacement and suppressed.
- Exceptionally strong isolated patches remain eligible.
- Broad morphological closing at the final boundary-refinement step was removed to prevent a deleted ring from being recreated.
- Boundary removal now requires low current PV-state support plus multiple negative cues; absence of a recent installation break alone cannot delete an old stable PV panel.

### 4. Historical growth-front completion

- A thin mixed-pixel shell can be stabilized only when the component already has high supported coverage and the missing region is primarily peripheral.
- Interior missing blocks are not automatically filled, preserving genuine phased construction.
- Temporal support is evaluated across a configurable multi-year window to reduce single-year image noise.

### 5. 2025 annular-change rejection

- Adjacent 2025 candidates that form a thin, nearly complete ring around a 2024 component are rejected as boundary displacement.
- Patchy, directional, or remote expansion remains available through the existing gain-supported and novelty-gated pathways.

## Compatibility and rerun behavior

The output `schema_version` remains `24.22.0`. A separate internal checkpoint field is added:

```text
algorithm_revision = 24.22.2-mixed-parcel-optimized
```

Therefore, existing Stage-7 and Stage-8 checkpoints created by the original v24.22 code are not reused. Re-running from Stage 7 automatically recalculates the optimized products without changing downstream paths or figures.

Recommended resume command:

```bash
bash /home/wsy/resume_china_inland_desert_pv_v24_22_novelty_gated_remote_change.sh \
  --mode test --from-step 7 --rerun-completed --foreground
```

For a full dataset run, use the same deployment command as v24.22 after replacing the code directory, or resume from Stage 7 with the full-run mode used by the existing deployment.

## Principal configuration controls

The new controls are present in both the canonical and authoritative configuration sections:

- `small_reference_component_pixels`
- `reference_internal_history_enhancement`
- `historical_component_*`
- `reference_endpoint_likelihood_weight`
- `preexisting_*`
- `suppress_annular_boundary_additions`
- `annular_*`
- `boundary_removal_minimum_negative_cues`
- `historical_shell_*`
- `terminal_suppress_annular_boundary_change`
- `terminal_annular_*`

Defaults are conservative: they prioritize preservation of the trusted 2024 manual inventory, require image evidence for historical backdating, and reject only geometrically clear complete-ring artifacts.

## Validation status

The original detail-optimization tests remain passing. The current v24.22.2 package collects 123 tests; all assertions passed in split batches, including the new mixed-pixel parcel tests. The eight new regression tests specifically cover early persistent PV, true late installation, weak object-edge recovery, partial-buildout preservation, small-block retention, oversized-boundary correction, 2024 annulus suppression, and 2025 annulus rejection. Modified Python modules also passed compilation checks.

The archive does not contain the full 142-unit Landsat dataset or independent historical labels. Therefore, software validation does not by itself establish field accuracy. Final scientific validation should compare representative units against visual interpretation and report installation-year error, IoU/boundary F1, omission of small blocks, and the frequency of complete-ring changes.

## Selected-unit calibration smoke-test hotfix

For explicitly selected-unit smoke tests only, probability calibration now falls back to the selected direct-detection units that have a valid 2024 probability raster when quality-aware rebalancing leaves no formal calibration-role/direct-unit intersection. The fallback is disabled for the formal full-domain experiment.
