# Training Contract and Quality Gates Design

## Goal

Fix the project's final dataset contract so later work does not drift, then define the
first formal implementation sub-project under the roadmap: stronger quality gates.

This design locks down the meaning of exported training pairs:

- `LR` must remain low-resolution in pixel size.
- `HR` must remain high-resolution in pixel size.
- `LR` should be geometrically aligned to `HR` as much as possible.
- `LR` should be brightness and color matched to `HR` as much as possible.
- `LR` and `HR` must preserve a stable scale relationship so they can be used directly
  as super-resolution training pairs.

## Why This Contract Matters

The project already has three processing phases:

1. frame selection
2. alignment
3. color matching

Without a fixed export contract, these phases can drift into a different goal:
producing visually nice intermediate images at the wrong size.

The user requirement is stricter than that. The project is not just generating aligned
previews. It is building a training dataset for super-resolution. That means the final
export must optimize for training semantics, not for intermediate inspection:

- exported `LR` is the training input
- exported `HR` is the supervision target
- the pair must be low-res versus high-res
- the pair must be spatially corresponding

## Current Baseline

The codebase already supports:

- phase 1 frame selection
- phase 2 alignment
- phase 3 color matching
- quality report generation
- final dataset export

Recent export behavior already moved in the right direction by separating:

- `final_lr_source`
- `gate_lr_source`
- `final_lr_resize_mode`

This is the key foundation for the new contract because it separates:

- which image content is exported
- which image content is evaluated for gating
- which size the final exported `LR` should keep

## Fixed Training Contract

The final exported dataset contract is:

```text
training pair = final LR + final HR

final LR:
  low-resolution pixel size
  geometrically aligned to HR as much as possible
  brightness/color matched to HR as much as possible
  preserves raw LR scale relationship

final HR:
  high-resolution pixel size
  remains the photo-derived target
```

This means intermediate phase outputs have different roles:

- `raw LR` is the authoritative low-resolution size reference
- `aligned LR` is the geometry-corrected intermediate result
- `color_matched LR` is the color-corrected intermediate result
- final export decides which content to use, but must still respect raw LR size when
  the chosen content came from an aligned or color-matched branch

## Export Semantics

The project should keep the following meaning fixed:

### `final_lr_source`

Controls which branch provides the final LR image content:

- `raw`
- `aligned`
- `color_matched`

### `gate_lr_source`

Controls which branch is used to decide whether a sample passes export quality gates.

### `final_lr_resize_mode`

Controls the output size behavior of exported `LR`.

- `copy`
  Preserve the chosen `final_lr_source` image size exactly.
- `match_raw`
  Resize the chosen `final_lr_source` content back to the original raw LR size.

## Recommended Export Policy

The recommended project policy should now be explicit:

1. Use `raw` as the final size anchor.
2. Allow `aligned` and `color_matched` to improve geometry and photometric consistency.
3. If final exported content comes from `aligned` or `color_matched`, export it with
   `final_lr_resize_mode: match_raw`.

This produces the intended training pair:

- content is closer to `HR`
- exported `LR` remains low-resolution
- exported `LR` and `HR` maintain a stable scale relationship

Recommended production-style export examples:

```yaml
export:
  final_lr_source: raw
  gate_lr_source: aligned
  final_lr_resize_mode: copy
```

This keeps the original low-resolution signal in the training input while using the
aligned branch to decide whether the sample is trustworthy enough.

```yaml
export:
  final_lr_source: color_matched
  gate_lr_source: color_matched
  final_lr_resize_mode: match_raw
```

This exports content from the most corrected branch, but still restores the final `LR`
to the original low-resolution size.

## Roadmap Order

The four formal roadmap lines should be implemented in this order:

1. stronger quality gates
2. stronger engineering layer
3. stronger alignment backends
4. stronger color backends

This order is intentional.

### Why Start with Quality Gates

Without reliable evaluation signals, later alignment and color upgrades cannot be
compared objectively.

### Why Engineering Layer Comes Second

Once the project starts integrating larger backends such as LoFTR, RAFT, GMFlow, or
SAM-style models, it will need unified model registration, caching, version control,
device routing, orchestration, and runtime statistics.

### Why Alignment and Color Backends Come After That

Those improvements become safer and cheaper once the project has both:

- a good evaluation ruler
- a stable execution foundation

## First Sub-Project: Quality Gates

The first formal implementation sub-project is `quality gates`.

Its purpose is not to change the final dataset contract. Its purpose is to make export
decisions measurable, configurable, and explainable.

## Candidate Approaches

Three reasonable designs exist.

### Approach 1: Single-Branch Metrics Only

Only compute and evaluate metrics for `gate_lr_source`.

Benefits:

- simplest implementation
- smallest CSV expansion
- smallest config surface

Costs:

- poor visibility into raw versus aligned versus color-matched behavior
- weak support for later algorithm comparison

### Approach 2: Record Three Branches, Gate One Branch

Compute metrics for `raw`, `aligned`, and `color_matched`, but use only
`gate_lr_source` to decide pass/fail during export.

Benefits:

- preserves a simple export decision path
- gives full visibility into all branches
- supports later algorithm analysis without redesigning the report format
- matches the current architecture and user goal well

Costs:

- larger report schema
- more metric computation work

### Approach 3: Full Multi-Branch Gating

Compute metrics for all branches and allow fully configurable joint gating policies.

Benefits:

- strongest flexibility
- supports sophisticated acceptance rules

Costs:

- highest config complexity
- highest testing burden
- too heavy for the first quality-gates milestone

## Recommendation

Adopt Approach 2 for the first implementation.

That means:

- the report records metrics for `raw`, `aligned`, and `color_matched`
- export still uses `gate_lr_source` as the primary pass/fail branch
- config naming and code structure should leave a path open for future multi-branch
  gating

This is the best balance between immediate value and implementation cost.

## Quality Gates Scope

The first implementation should add stronger but still practical signals.

### Reported Metrics

For each available branch among `raw`, `aligned`, and `color_matched`, report:

- `MAE`
- `PSNR`
- `SSIM`
- dimension check result
- aspect-ratio consistency result
- border or crop artifact signal

From alignment diagnostics, also expose flow-related signals when available:

- `flow_status`
- mean flow magnitude
- flow outlier signal or anomaly flag

The report should continue to be plain CSV plus existing preview outputs.

### Gating Behavior

Export should continue to accept or reject each sample using a clear reason code.

First-pass gating should remain centered on `gate_lr_source`, plus existing alignment
status checks.

Examples of gate types:

- alignment status required
- flow status required
- minimum alignment confidence
- maximum branch-to-HR `MAE`
- minimum branch-to-HR `PSNR`
- minimum branch-to-HR `SSIM`
- dimension check required
- border artifact score must stay below threshold

All new gates should be optional.

## Branch Metric Naming

To support future comparability, metric naming should be explicit rather than
overloaded.

Preferred report naming pattern:

```text
raw_to_hr_mae
raw_to_hr_psnr
raw_to_hr_ssim

aligned_to_hr_mae
aligned_to_hr_psnr
aligned_to_hr_ssim

color_matched_to_hr_mae
color_matched_to_hr_psnr
color_matched_to_hr_ssim
```

Boolean or categorical checks should follow the same pattern where applicable.

## Configuration Direction

The first implementation should preserve the simple top-level export contract while
opening room for later per-branch thresholds.

Recommended direction:

```yaml
export:
  gate_lr_source: aligned
  quality_gates:
    gate_branch:
      max_mae: 30.0
      min_psnr: 18.0
      min_ssim: 0.45
      require_dimension_check: true
      max_border_artifact: 12.0
```

Or, if the existing config style prefers flatter keys, the first pass can stay flatter
as long as the branch semantics remain explicit.

The important part is not the exact nesting. The important part is:

- gating remains branch-aware
- branch naming is explicit
- all thresholds stay optional

## Non-Goals for This Sub-Project

The first quality-gates milestone does not need to implement:

- HTML compare tooling
- learned quality models
- face detectors or semantic masks
- multi-branch boolean expressions
- fully advanced flow analysis
- final decisions based on all branches at once

These can come later after the first metric and export-gating baseline is stable.

## Interface Stability Requirements

This design should keep future upgrades cheap.

### For Alignment Backends

Future backends such as LoFTR, LightGlue, RAFT, GMFlow, SAM-style masked alignment, or
LLM/fusion routing should plug into the same alignment contract and emit diagnostics
that quality gates can consume without changing export semantics.

### For Color Backends

Future color methods such as histogram matching, masked transfer, LUT fitting, Retinex,
or small neural color models should also fit the same reporting and export structure.

### For Engineering Layer

Model registration, cache control, version pinning, device assignment, resumable runs,
and runtime statistics should enhance the same pipeline rather than redefining the data
contract.

## Error Handling

Quality metrics must fail soft per sample.

If a metric cannot be computed:

- keep the sample row in the report
- leave the metric empty or record a stable missing value representation
- let export reject only if a threshold requires that metric
- emit a clear rejection reason such as `gate_source_to_hr_ssim_missing`

The batch should continue running.

## Testing Strategy

The first implementation should add tests in this order:

1. report-level metric columns and computations
2. report behavior when aligned or color-matched files are missing
3. export config validation for new thresholds
4. export rejection reasons for each new gate
5. pipeline integration for report plus export with the new gates

Synthetic image fixtures are sufficient for most unit tests. Real-data smoke remains
useful later for threshold tuning, not for exact metric assertions.

## Success Criteria

This sub-project is successful when:

- the final training-pair contract is preserved
- `LR` remains low-resolution in final export
- `HR` remains high-resolution in final export
- the report exposes branch-separated quality metrics
- export can reject samples using stronger optional thresholds
- rejection reasons are explicit and machine-readable
- later alignment and color upgrades can be compared using the same report structure

## Open Design Decision Deferred

The first quality-gates implementation should not yet decide whether future export
policies may combine multiple branches at once.

The design intentionally keeps that option open, but does not require it now.
