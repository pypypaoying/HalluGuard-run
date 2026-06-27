# Stage 6 FOMC Summary

## Verdict

- Compact go: `False`
- Spectral delta vs LRBN: `-1.158871%`
- Rolling delta vs LRBN: `-0.597737%`
- Spectral minus rolling: `-0.561135%`
- Spectral harm: `0.467448`
- Coverage gap: `6.658664pp`
- Protocol guard pass: `True`
- Test threshold leakage: `False`

## Online Adapter Results

| method | n | mse | mae | lrbn_mse | lrbn_mae | mse_delta_vs_lrbn | mse_delta_pct_vs_lrbn | mae_delta_pct_vs_lrbn | harm_rate | win_rate | mean_win_size | mean_loss_size | win_loss_ratio | top5_loss_contribution | coverage | selected_count | selected_harm_rate | buffer_size | mean_q90_width | mean_pointwise_coverage90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| spectral_adapter | 768 | 4.837441 | 1.680623 | 4.894158 | 1.682162 | -0.056717 | -1.158871 | -0.091480 | 0.467448 | 0.532552 | 0.339005 | 0.264887 | 1.279810 | 0.459378 | 0.000000 | 0 | 0.000000 | 128 | 9.705355 | 0.966587 |
| rolling_mean_residual | 768 | 4.864903 | 1.699420 | 4.894158 | 1.682162 | -0.029254 | -0.597737 | 1.025934 | 0.531250 | 0.468750 | 0.953886 | 0.786597 | 1.212674 | 0.427916 | 0.000000 | 0 | 0.000000 | 128 | 9.705355 | 0.968438 |
| no_update | 768 | 4.894158 | 1.682162 | 4.894158 | 1.682162 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0.000000 | 128 | 9.705355 | 0.964959 |
| time_ema_residual | 768 | 4.901755 | 1.706658 | 4.894158 | 1.682162 | 0.007598 | 0.155240 | 1.456221 | 0.541667 | 0.458333 | 1.016509 | 0.874150 | 1.162855 | 0.475175 | 0.000000 | 0 | 0.000000 | 128 | 9.705355 | 0.968051 |

## Spectral Autocorrelation

| dataset | backbone | horizon | seed | band | lag1_autocorr | mean_energy |
| --- | --- | --- | --- | --- | --- | --- |
| ETTh1 | DLinear | 96 | 2026 | 0 | 0.546364 | 4965.054635 |
| ETTh1 | DLinear | 96 | 2026 | 1 | 0.660185 | 92.608503 |
| ETTh1 | DLinear | 96 | 2026 | 2 | 0.569271 | 35.858958 |
| ETTh1 | DLinear | 96 | 2026 | 3 | 0.565652 | 26.154526 |
| ETTh1 | DLinear | 192 | 2026 | 0 | 0.704632 | 10901.221383 |
| ETTh1 | DLinear | 192 | 2026 | 1 | 0.757570 | 182.680362 |
| ETTh1 | DLinear | 192 | 2026 | 2 | 0.757634 | 72.010887 |
| ETTh1 | DLinear | 192 | 2026 | 3 | 0.680397 | 54.107020 |
| ETTh1 | PatchTST | 96 | 2026 | 0 | 0.541446 | 5089.520105 |
| ETTh1 | PatchTST | 96 | 2026 | 1 | 0.584010 | 173.924377 |
| ETTh1 | PatchTST | 96 | 2026 | 2 | 0.411432 | 130.369703 |
| ETTh1 | PatchTST | 96 | 2026 | 3 | 0.481402 | 133.215330 |
| ETTh1 | PatchTST | 192 | 2026 | 0 | 0.685424 | 11167.854931 |
| ETTh1 | PatchTST | 192 | 2026 | 1 | 0.514809 | 364.266157 |
| ETTh1 | PatchTST | 192 | 2026 | 2 | 0.477876 | 262.178208 |
| ETTh1 | PatchTST | 192 | 2026 | 3 | 0.528023 | 261.817137 |
| ETTm1 | DLinear | 96 | 2026 | 0 | 0.080758 | 2977.956236 |
| ETTm1 | DLinear | 96 | 2026 | 1 | 0.008661 | 29.512537 |
| ETTm1 | DLinear | 96 | 2026 | 2 | 0.115685 | 10.104784 |
| ETTm1 | DLinear | 96 | 2026 | 3 | 0.127705 | 7.618224 |