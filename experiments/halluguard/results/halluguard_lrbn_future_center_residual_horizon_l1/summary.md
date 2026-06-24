# HalluGuard-LRBN Summary

Learnable reversible boundary normalization ablations for robust anchor, residual gate, horizon gate, and unified RevIN-RDN hybrid.

- Completed rows: 96 / 96
- Test threshold leakage: False
- seq_len: 96
- tail_len: 48

## Variant Summary

- `future_center_horizon_residual_gate` / `DLinear`: completed 8 / 8, mean MSE 5.649340943364515, mean MAE 1.8068383417939553, mean MSE delta vs raw -27.699462310677944, blocked 0
- `future_center_horizon_residual_gate` / `PatchTST`: completed 8 / 8, mean MSE 5.835552123261706, mean MAE 1.8508271076267013, mean MSE delta vs raw -0.1214894210808394, blocked 0
- `future_center_horizon_residual_gate_strong` / `DLinear`: completed 8 / 8, mean MSE 5.654428171054372, mean MAE 1.8063782926994227, mean MSE delta vs raw -27.640033560142186, blocked 0
- `future_center_horizon_residual_gate_strong` / `PatchTST`: completed 8 / 8, mean MSE 5.824440850461398, mean MAE 1.849295133685437, mean MSE delta vs raw -0.26593603946781685, blocked 0
- `future_center_horizon_selector` / `DLinear`: completed 8 / 8, mean MSE 5.6395135034586, mean MAE 1.8040404168510333, mean MSE delta vs raw -27.856965132686923, blocked 0
- `future_center_horizon_selector` / `PatchTST`: completed 8 / 8, mean MSE 5.825653861970141, mean MAE 1.8493616432611126, mean MSE delta vs raw -0.26696848146724134, blocked 0
- `future_center_selector` / `DLinear`: completed 8 / 8, mean MSE 5.648922074387425, mean MAE 1.8056805920488777, mean MSE delta vs raw -27.76531025808861, blocked 0
- `future_center_selector` / `PatchTST`: completed 8 / 8, mean MSE 5.815322895714727, mean MAE 1.847592314094849, mean MSE delta vs raw -0.4722183744035747, blocked 0
- `raw_no_correction` / `DLinear`: completed 8 / 8, mean MSE 9.180323123453949, mean MAE 2.2483446980850332, mean MSE delta vs raw 0.0, blocked 0
- `raw_no_correction` / `PatchTST`: completed 8 / 8, mean MSE 5.844938262863285, mean MAE 1.8516697711273498, mean MSE delta vs raw 0.0, blocked 0
- `unified_revin_rdn_hybrid` / `DLinear`: completed 8 / 8, mean MSE 5.655497250632147, mean MAE 1.8081324433262052, mean MSE delta vs raw -27.567475740429863, blocked 0
- `unified_revin_rdn_hybrid` / `PatchTST`: completed 8 / 8, mean MSE 5.825162314542845, mean MAE 1.8498515115411975, mean MSE delta vs raw -0.29072934136697576, blocked 0
- `future_center_horizon_residual_gate` / `ALL`: completed 16 / 16, mean MSE 5.74244653331311, mean MAE 1.8288327247103284, mean MSE delta vs raw -13.910475865879393, blocked 0
- `future_center_horizon_residual_gate_strong` / `ALL`: completed 16 / 16, mean MSE 5.739434510757885, mean MAE 1.82783671319243, mean MSE delta vs raw -13.952984799805, blocked 0
- `future_center_horizon_selector` / `ALL`: completed 16 / 16, mean MSE 5.73258368271437, mean MAE 1.8267010300560726, mean MSE delta vs raw -14.061966807077077, blocked 0
- `future_center_selector` / `ALL`: completed 16 / 16, mean MSE 5.732122485051077, mean MAE 1.8266364530718633, mean MSE delta vs raw -14.118764316246093, blocked 0
- `raw_no_correction` / `ALL`: completed 16 / 16, mean MSE 7.512630693158616, mean MAE 2.0500072346061917, mean MSE delta vs raw 0.0, blocked 0
- `unified_revin_rdn_hybrid` / `ALL`: completed 16 / 16, mean MSE 5.740329782587495, mean MAE 1.8289919774337011, mean MSE delta vs raw -13.92910254089842, blocked 0
