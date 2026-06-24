# HalluGuard-LRBN Future-Center Restoration Report

Status: initial validation in progress.

## Hypothesis

The LRBN+NST gain decomposition showed that most improvement came from
level/center restoration. A targeted center-selection mechanism should capture
that signal more cleanly than a generic NST branch.

## Variants

- `unified_revin_rdn_hybrid`: parent baseline.
- `future_center_static`: global trainable mixture of instance mean, boundary
  anchor, tail median, and short trend extrapolated anchor.
- `future_center_static_drift`: static mixture plus a small horizon drift.
- `future_center_selector`: feature-conditioned softmax over the same anchors.
- `future_center_selector_drift`: selector plus learned horizon drift.

## Initial Results

Pending.
