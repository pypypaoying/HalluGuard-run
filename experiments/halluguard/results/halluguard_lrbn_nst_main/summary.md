# HalluGuard-LRBN Summary

Learnable reversible boundary normalization ablations for robust anchor, residual gate, horizon gate, and unified RevIN-RDN hybrid.

- Completed rows: 64 / 64
- Test threshold leakage: False
- seq_len: 96
- tail_len: 48

## Variant Summary

- `lrbn_nst_conservative_gate` / `DLinear`: completed 8 / 8, mean MSE 5.567236270904859, mean MAE 1.7833489582347104, mean MSE delta vs raw -21.00839789751225, blocked 0
- `lrbn_nst_conservative_gate` / `PatchTST`: completed 8 / 8, mean MSE 5.49609322421714, mean MAE 1.778453635291393, mean MSE delta vs raw 0.07241823213034149, blocked 0
- `lrbn_nst_feature_gate` / `DLinear`: completed 8 / 8, mean MSE 5.572116673070061, mean MAE 1.7838423731651427, mean MSE delta vs raw -20.918787576548166, blocked 0
- `lrbn_nst_feature_gate` / `PatchTST`: completed 8 / 8, mean MSE 5.418452594350727, mean MAE 1.7680535516802496, mean MSE delta vs raw -1.0085640575087216, blocked 0
- `raw_no_correction` / `DLinear`: completed 8 / 8, mean MSE 7.52359273939077, mean MAE 2.0428398284690954, mean MSE delta vs raw 0.0, blocked 0
- `raw_no_correction` / `PatchTST`: completed 8 / 8, mean MSE 5.488909354050996, mean MAE 1.7769200075304599, mean MSE delta vs raw 0.0, blocked 0
- `unified_revin_rdn_hybrid` / `DLinear`: completed 8 / 8, mean MSE 5.56506187752376, mean MAE 1.7833103219897466, mean MSE delta vs raw -21.014513323538072, blocked 0
- `unified_revin_rdn_hybrid` / `PatchTST`: completed 8 / 8, mean MSE 5.4839320025479665, mean MAE 1.7767609636266923, mean MSE delta vs raw 0.030435679085725212, blocked 0
- `lrbn_nst_conservative_gate` / `ALL`: completed 16 / 16, mean MSE 5.531664747560999, mean MAE 1.7809012967630518, mean MSE delta vs raw -10.467989832690956, blocked 0
- `lrbn_nst_feature_gate` / `ALL`: completed 16 / 16, mean MSE 5.495284633710394, mean MAE 1.7759479624226961, mean MSE delta vs raw -10.963675817028443, blocked 0
- `raw_no_correction` / `ALL`: completed 16 / 16, mean MSE 6.506251046720883, mean MAE 1.9098799179997779, mean MSE delta vs raw 0.0, blocked 0
- `unified_revin_rdn_hybrid` / `ALL`: completed 16 / 16, mean MSE 5.524496940035863, mean MAE 1.7800356428082191, mean MSE delta vs raw -10.492038822226174, blocked 0
