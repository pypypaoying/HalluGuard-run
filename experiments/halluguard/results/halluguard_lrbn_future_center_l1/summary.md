# HalluGuard-LRBN Summary

Learnable reversible boundary normalization ablations for robust anchor, residual gate, horizon gate, and unified RevIN-RDN hybrid.

- Completed rows: 80 / 80
- Test threshold leakage: False
- seq_len: 96
- tail_len: 48

## Variant Summary

- `future_center_selector` / `DLinear`: completed 8 / 8, mean MSE 5.648922074387425, mean MAE 1.8056805920488777, mean MSE delta vs raw -27.76531025808861, blocked 0
- `future_center_selector` / `PatchTST`: completed 8 / 8, mean MSE 5.815322895714727, mean MAE 1.847592314094849, mean MSE delta vs raw -0.4722183744035747, blocked 0
- `future_center_selector_drift` / `DLinear`: completed 8 / 8, mean MSE 5.636603920367677, mean MAE 1.8036290008530111, mean MSE delta vs raw -27.856492279676978, blocked 0
- `future_center_selector_drift` / `PatchTST`: completed 8 / 8, mean MSE 5.835382787384155, mean MAE 1.8521610448349117, mean MSE delta vs raw -0.041025954881305726, blocked 0
- `future_center_static` / `DLinear`: completed 8 / 8, mean MSE 5.645053291756932, mean MAE 1.8046580719484069, mean MSE delta vs raw -27.803479320152928, blocked 0
- `future_center_static` / `PatchTST`: completed 8 / 8, mean MSE 5.8418211825954085, mean MAE 1.851711530952227, mean MSE delta vs raw -0.09557355510148519, blocked 0
- `raw_no_correction` / `DLinear`: completed 8 / 8, mean MSE 9.180323123453949, mean MAE 2.2483446980850332, mean MSE delta vs raw 0.0, blocked 0
- `raw_no_correction` / `PatchTST`: completed 8 / 8, mean MSE 5.844938262863285, mean MAE 1.8516697711273498, mean MSE delta vs raw 0.0, blocked 0
- `unified_revin_rdn_hybrid` / `DLinear`: completed 8 / 8, mean MSE 5.655497250632147, mean MAE 1.8081324433262052, mean MSE delta vs raw -27.567475740429863, blocked 0
- `unified_revin_rdn_hybrid` / `PatchTST`: completed 8 / 8, mean MSE 5.825162314542845, mean MAE 1.8498515115411975, mean MSE delta vs raw -0.29072934136697576, blocked 0
- `future_center_selector` / `ALL`: completed 16 / 16, mean MSE 5.732122485051077, mean MAE 1.8266364530718633, mean MSE delta vs raw -14.118764316246093, blocked 0
- `future_center_selector_drift` / `ALL`: completed 16 / 16, mean MSE 5.735993353875917, mean MAE 1.8278950228439617, mean MSE delta vs raw -13.948759117279142, blocked 0
- `future_center_static` / `ALL`: completed 16 / 16, mean MSE 5.743437237176171, mean MAE 1.8281848014503166, mean MSE delta vs raw -13.949526437627208, blocked 0
- `raw_no_correction` / `ALL`: completed 16 / 16, mean MSE 7.512630693158616, mean MAE 2.0500072346061917, mean MSE delta vs raw 0.0, blocked 0
- `unified_revin_rdn_hybrid` / `ALL`: completed 16 / 16, mean MSE 5.740329782587495, mean MAE 1.8289919774337011, mean MSE delta vs raw -13.92910254089842, blocked 0
