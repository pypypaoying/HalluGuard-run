# HalluGuard-LRBN Summary

Learnable reversible boundary normalization ablations for robust anchor, residual gate, horizon gate, and unified RevIN-RDN hybrid.

- Completed rows: 80 / 80
- Test threshold leakage: False
- seq_len: 96
- tail_len: 48

## Variant Summary

- `future_center_residual_shift` / `DLinear`: completed 8 / 8, mean MSE 5.651895012833654, mean MAE 1.8076002611697597, mean MSE delta vs raw -27.630665764226016, blocked 0
- `future_center_residual_shift` / `PatchTST`: completed 8 / 8, mean MSE 5.8229405318929235, mean MAE 1.8489298098726235, mean MSE delta vs raw -0.3088563985084405, blocked 0
- `future_center_residual_shift_cap015` / `DLinear`: completed 8 / 8, mean MSE 5.65162723737706, mean MAE 1.807450929153161, mean MSE delta vs raw -27.615850662417145, blocked 0
- `future_center_residual_shift_cap015` / `PatchTST`: completed 8 / 8, mean MSE 5.85318406620524, mean MAE 1.8523637387824528, mean MSE delta vs raw 0.007926316587418353, blocked 0
- `future_center_selector` / `DLinear`: completed 8 / 8, mean MSE 5.648922074387425, mean MAE 1.8056805920488777, mean MSE delta vs raw -27.76531025808861, blocked 0
- `future_center_selector` / `PatchTST`: completed 8 / 8, mean MSE 5.815322895714727, mean MAE 1.847592314094849, mean MSE delta vs raw -0.4722183744035747, blocked 0
- `raw_no_correction` / `DLinear`: completed 8 / 8, mean MSE 9.180323123453949, mean MAE 2.2483446980850332, mean MSE delta vs raw 0.0, blocked 0
- `raw_no_correction` / `PatchTST`: completed 8 / 8, mean MSE 5.844938262863285, mean MAE 1.8516697711273498, mean MSE delta vs raw 0.0, blocked 0
- `unified_revin_rdn_hybrid` / `DLinear`: completed 8 / 8, mean MSE 5.655497250632147, mean MAE 1.8081324433262052, mean MSE delta vs raw -27.567475740429863, blocked 0
- `unified_revin_rdn_hybrid` / `PatchTST`: completed 8 / 8, mean MSE 5.825162314542845, mean MAE 1.8498515115411975, mean MSE delta vs raw -0.29072934136697576, blocked 0
- `future_center_residual_shift` / `ALL`: completed 16 / 16, mean MSE 5.737417772363288, mean MAE 1.8282650355211916, mean MSE delta vs raw -13.969761081367228, blocked 0
- `future_center_residual_shift_cap015` / `ALL`: completed 16 / 16, mean MSE 5.752405651791149, mean MAE 1.8299073339678067, mean MSE delta vs raw -13.803962172914863, blocked 0
- `future_center_selector` / `ALL`: completed 16 / 16, mean MSE 5.732122485051077, mean MAE 1.8266364530718633, mean MSE delta vs raw -14.118764316246093, blocked 0
- `raw_no_correction` / `ALL`: completed 16 / 16, mean MSE 7.512630693158616, mean MAE 2.0500072346061917, mean MSE delta vs raw 0.0, blocked 0
- `unified_revin_rdn_hybrid` / `ALL`: completed 16 / 16, mean MSE 5.740329782587495, mean MAE 1.8289919774337011, mean MSE delta vs raw -13.92910254089842, blocked 0
