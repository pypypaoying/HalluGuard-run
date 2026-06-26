# HalluGuard 后续创新方向即时验证报告

## 结论摘要

我把 `HalluGuard_research_directions_validation_plan.docx` 中的创新方向转成了一套样本级可证伪验证实验，并已在本地跑完。

本轮不是完整论文大表，而是“方向筛选 / go-no-go pilot”。输入为一个紧凑但真实的 forecast set：

- Dataset: `ETTm1`, `ETTh1`
- Backbone: `DLinear`, `PatchTST`
- Horizon: `96`, `192`
- Seed: `2026`
- Samples: `1536`
- Action rows: `7680`
- Split: validation 用于训练/校准轻量诊断，test 只用于评估

输出目录：

`experiments/halluguard/results/research_direction_validation/`

最重要结论：

1. **Residual alignment 主命题成立得比较强**：`HalluGuard-LRBN` 在 test 上 mean MSE delta 为 `-1.5331`，A>1 比例 `0.642`，说明当前主线确实存在“修正方向与 residual 对齐”的信号。
2. **No-harm selective correction 当前特征不成立**：LRBN harm classifier 的 risk AUC 只有 `0.434`，50% coverage 的 harm rate `0.424` 反而高于全覆盖 `0.358`。这说明不能直接把现有特征拿来做 harm router。
3. **残差低维结构很强，但权重可预测性弱**：top10 PCA explained variance 平均 `0.914`，test reconstruction EVR `0.940`，但 basis weight R2 `-0.882`。这说明 residual basis corrector 有上限，但需要更好的 target-free 权重预测器。
4. **动态一致性投影有明确信号**：简单 `boundary_projection` 仅用 validation 选 strength 后，在 test 上 MSE delta `-0.2250`，约 `-3.50%`，支持“边界/局部动态投影”继续作为下一代方法线。
5. **多尺度编辑能降 MSE，但机制证据还弱**：unsupported high-frequency shrink test delta `-0.3159`，但 highfreq mismatch 与 highfreq residual 的 Spearman 只有 `0.085`。这更像“有效编辑原型”，还不是成熟 amplitude-phase support claim。
6. **能量/critic 方向值得继续，但暂时只能作为 scorer**：true future vs raw/corrupted 的 feature critic AUC `0.938`，separability 很强；但本轮没有验证 score gradient 与 residual 对齐，所以不能声称 diffusion/refinement 已成立。
7. **Regime-invariant 有弱到中等信号**：平均 top-action rate `0.513`，cross-domain consistency `0.667`。可继续做更稳的 regime encoder，但当前 KMeans 原型还不够论文级。
8. **TSFM critic 方向 blocked**：本地没有 Chronos/TimesFM/Moirai forecast 文件，因此没有伪造 TSFM disagreement。

## 运行命令

生成紧凑 forecast 输入：

```bash
python scripts/run_tablea_full.py \
  --datasets ETTm1,ETTh1 \
  --backbones DLinear,PatchTST \
  --horizons 96,192 \
  --seeds 2026 \
  --methods raw_no_correction,HalluGuard-LRBN,matched_sparse_smoothing,naive_smoothing,ema_smoothing,median_smoothing \
  --data-root external/ETDataset \
  --output-dir experiments/halluguard/results/research_direction_validation/forecast_inputs \
  --epochs 2 \
  --max-train-windows 256 \
  --max-eval-windows 96 \
  --device cpu \
  --skip-existing
```

运行方向验证：

```bash
python experiments/halluguard/run_research_direction_validation.py \
  --metrics-csv experiments/halluguard/results/research_direction_validation/forecast_inputs/combined_metrics.csv \
  --output-dir experiments/halluguard/results/research_direction_validation
```

## Direction Verdicts

| Direction | Verdict | Evidence |
|---|---:|---|
| E1 Residual alignment | promising | Best action `HalluGuard-LRBN`, test delta `-1.53306`, A>1 rate `0.642` |
| E2 Oracle action separability | weak | Accuracy `0.428`, majority `0.411`, shuffled `0.402` |
| E3 No-harm selective correction | weak | Risk AUC `0.434`; 50% coverage harm `0.424` vs full `0.358` |
| E4 Residual basis decomposition | promising but incomplete | top10 PCA EVR `0.914`, test recon EVR `0.940`, but weight R2 `-0.882` |
| E5 Dynamic consistency projection | promising | `boundary_projection` delta `-0.224956` (`-3.500%`) |
| E6 Multiscale amplitude-phase support | promising but mechanism weak | edit delta `-0.315904`; highfreq mismatch/residual Spearman `0.085` |
| E7 Energy critic separability | promising as scorer | critic AUC `0.938`; gradient alignment not evaluated |
| E8 TSFM disagreement | blocked | no local foundation forecast files |
| E9 Regime-invariant correction | promising but early | top-action rate `0.513`, cross-domain consistency `0.667` |

## Action Alignment

| Action | Test mean MSE delta | Harm rate | A>1 rate | Mean cosine |
|---|---:|---:|---:|---:|
| HalluGuard-LRBN | `-1.5331` | `0.358` | `0.642` | `0.433` |
| ema_smoothing | `-0.3667` | `0.000` | `1.000` | `0.311` |
| naive_smoothing | `-0.3552` | `0.000` | `1.000` | `0.304` |
| median_smoothing | `-0.2973` | `0.025` | `0.975` | `0.282` |
| matched_sparse_smoothing | `-0.2047` | `0.268` | `0.721` | `0.151` |

Interpretation:

- LRBN has the largest absolute MSE gain but nontrivial harm risk.
- Smoothing controls are very safe in this compact pilot, but their corrections are smaller and more smoothing-like.
- This supports keeping LRBN as the main clean-claim line while adding better harm control only if the risk signal improves.

## Risk-Coverage Finding

For `HalluGuard-LRBN`, the validation-trained harm classifier failed:

| Coverage | Selected mean delta | Selected harm | Full harm | Risk AUC |
|---:|---:|---:|---:|---:|
| 0.25 | `-0.4901` | `0.411` | `0.358` | `0.434` |
| 0.50 | `-0.4663` | `0.424` | `0.358` | `0.434` |
| 0.75 | `-1.0592` | `0.380` | `0.358` | `0.434` |
| 1.00 | `-1.5331` | `0.358` | `0.358` | `0.434` |

Conclusion: current target-free features do **not** support a no-harm selective correction claim. The next router should not be based on this feature set alone.

## Residual Basis Finding

Residuals are highly low-dimensional:

- mean top5 PCA EVR: `0.841`
- mean top10 PCA EVR: `0.914`
- mean top20 PCA EVR: `0.960`
- mean test reconstruction EVR: `0.940`
- mean DCT top20 EVR: `0.836`

But target-free feature predictability is poor:

- basis weight R2: `-0.882`

Conclusion: decomposed residual corrector has real headroom, but the hard part is predicting basis weights without target leakage. This should be treated as a second-stage method, not immediate production.

## Dynamic Projection Finding

Validation-calibrated projection prototypes:

| Variant | Best strength | Test MSE delta | Harm rate | A>1 rate |
|---|---:|---:|---:|---:|
| boundary_projection | `0.75` | `-0.224956` | `0.339` | `0.661` |
| dynamic_combo_projection | `0.50` | `-0.135971` | `0.434` | `0.566` |
| slope_projection | `0.00` | `0.000000` | `0.000` | `0.000` |
| curvature_projection | `0.00` | `0.000000` | `0.000` | `0.000` |

Conclusion: boundary consistency is the actionable dynamic constraint. Slope/curvature alone were not supported by validation here.

## Multiscale Finding

- scale residual energy Gini: about `0.511`, suggesting residual energy is moderately concentrated across scales.
- highfreq mismatch vs highfreq residual Spearman: `0.085` on test, weak.
- multiscale unsupported-HF shrink delta: `-0.315904`.

Conclusion: scale-wise editing is worth continuing, but current high-frequency support score is too weak for a strong amplitude-phase mechanism claim.

## Critic Finding

Feature critic distinguishing true future from raw/corrupted trajectories:

- AUC: `0.938`
- AUPRC: `0.843`

Conclusion: trajectory critic separability is strong. The next falsification test is whether a differentiable critic gradient or critic-selected projection direction aligns with residual, not just whether it classifies corrupted samples.

## Regime Finding

KMeans on target-free features found regimes with:

- mean top-action rate: `0.513`
- cross-domain top-action consistency: `0.667`

Conclusion: there is some regime structure, but current clusters are not clean enough. Regime should be used as a diagnostic layer or combined with better mechanism labels, not as a full router yet.

## Recommended Next Experiments

1. **Make dynamic boundary projection the next concrete method prototype.**
   - It is simple, mechanism-aligned, validation-calibrated, and already improves test MSE in this pilot.
   - Compare it against LRBN, EMA/median/naive smoothing, and matched sparse smoothing under equal coverage.

2. **Build a critic-assisted projection selector, not a critic-gradient refiner yet.**
   - Critic separability is strong, but gradient alignment is untested.
   - First use critic score to decide when boundary projection is plausible.

3. **Do not prioritize no-harm router with the current features.**
   - Risk AUC below random means this path is currently misleading.
   - If revived, add stronger features: projection-specific alignment proxies, critic score, residual-basis proxy, and regime labels.

4. **Keep residual basis decomposition as a high-upside but harder line.**
   - The residual is low-dimensional, but target-free basis-weight prediction failed.
   - Next step should be sign prediction of the first few residual modes with richer features, not full residual MLP.

5. **TSFM critic requires real foundation forecasts before any claim.**
   - Prepare a separate adapter for Chronos/TimesFM/Moirai outputs.
   - Until then, TSFM remains blocked.

## Limitations

- This is a compact local pilot, not the final full TableA.
- Only `ETTm1/ETTh1`, `DLinear/PatchTST`, horizons `96/192`, seed `2026` were used.
- All results are valid as direction screening, not paper-grade final evidence.
- No TSFM forecast files were available.
- Critic gradient alignment was not evaluated.
