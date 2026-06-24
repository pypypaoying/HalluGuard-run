# HalluGuard-LRBN + NST Complementarity Report

Status: implementation in progress.

## Parent

Main claim-clean parent: `unified_revin_rdn_hybrid`.

## Candidate Variants

- `unified_revin_rdn_hybrid`: parent baseline.
- `nst_lightweight`: shared-runner NST-style stationarization baseline.
- `lrbn_unified_nst_residual`: claim-clean residual stationarization on top of
  LRBN normalized coordinates.
- `lrbn_nst_output_blend`: diagnostic train-split blend of LRBN and NST
  branches.

## Results

Pending smoke/full run.

## Claim Boundary

This experiment does not claim to reproduce the full Non-stationary Transformer
architecture unless the official attention de-stationary factors are integrated
and evaluated. The first pass tests whether NST-style stationarization is
complementary to HalluGuard-LRBN under the existing fair adapter protocol.
