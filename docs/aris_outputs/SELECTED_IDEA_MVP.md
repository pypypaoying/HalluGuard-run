# Selected Idea MVP

输入依据：`docs/aris_outputs/TOP3_EXPERIMENT_PLAN.md`

决策结论：选择 **Idea 9: HalluGuard Trend-Frequency Test-Time Correction** 作为当前项目立即推进的 MVP。

## 1. 选中的 idea

选中 idea：**HalluGuard Trend-Frequency Test-Time Correction**。

选择理由：

- 结构创新明确：它不是改主干模型，而是在 inference-time 增加一个黑盒输出可靠性层，专门处理趋势/频谱背离这一 forecast dynamics violation。
- 不是模块堆砌：没有引入 Mamba、Transformer、LLM 或多模态拼接，只围绕趋势规则和频谱规则做可解释校正。
- 实验成本最低：若已有 baseline 预测输出，可完全离线评估；即使没有，也只需先跑最小 baseline。
- 2 周内可完成 MVP：第一版只需要预测输出、ground truth、规则触发器、校正器和评测脚本。
- Ablation 清楚：no correction、naive smoothing、trend-only、frequency-only、trend+frequency、random trigger 都是直接可比的。
- 失败也有价值：如果趋势/频谱规则无法预测错误或会误伤真实转折，就能明确说明当前公开 TSF 设置下 dynamics-level hallucination 不是主要瓶颈。

为什么不选另外两个：

- **不优先选 Idea 2**：Asymmetric Patch Decoder 需要改模型 head 和 loss，并重新训练多个变体；它仍然很有价值，但与 TimeMosaic/TimesFM 的 patch/segment decoding overlap 更强，MVP 成本高于 Idea 9。
- **不优先选 Idea 5**：Recovery Gate 的 paper story 很好，但当前项目没有现成多变量通道交互代码；需要先建立 channel mask、stress test 和通道贡献分析框架，2 周内风险更高。

## 2. Paper Story 草案

当前时间序列预测研究通常只报告 MSE/MAE，却很少检查预测序列是否违反历史上下文中的基本动态规律。尤其在长视野或零样本预测中，模型可能输出平均误差不算极端、但趋势或频谱结构明显不可信的 forecast。我们将这类错误视为 output-space dynamics violation，并提出一个 black-box test-time guard：HalluGuard。HalluGuard 不访问模型隐状态，也不重新训练模型，只根据 context 与 forecast 的趋势和频谱一致性触发保守校正。该方法的目标不是取代主模型，而是在预测输出层提供一个低成本可靠性干预。论文故事可以围绕三个问题展开：这些 dynamics violation 是否真实存在；它们是否与传统误差指标脱钩；一个简单、可解释的 test-time correction 是否能减少 violation 而不损害 MSE/MAE。若实验成立，贡献是一个轻量、模型无关、可插拔的 TSF 可靠性层；若实验失败，也能澄清 hallucination-style 指标在常用 TSF benchmark 中的适用边界。

## 3. 方法结构

HalluGuard 的 MVP 是一个离线或推理时后处理算法，流程如下：

1. 输入历史窗口 `x`、模型预测 `y_hat`，以及可选 ground truth `y` 用于评估。
2. 从 `x` 中估计历史趋势、局部频谱结构和高频能量比例。
3. 从 `y_hat` 中估计预测趋势、预测频谱结构和异常高频能量。
4. 用验证集阈值判断预测是否触发趋势 violation、频谱 violation，或二者之一。
5. 若不触发，直接返回原预测。
6. 若触发，执行最小幅度校正：
   - trend correction：削弱预测趋势相对历史趋势的异常偏离；
   - frequency correction：压制预测中无依据的异常高频成分；
   - combined correction：同时使用趋势与频谱校正。
7. 输出校正后的预测 `y_tilde`，并记录触发类型、校正强度、额外延迟和 violation 指标。

该结构作用在 **inference-time correction** 与 **evaluation** 环节，不改变 dataloader、主模型、embedding、patching、loss 或训练流程。

## 4. 数学表达

符号定义：

- 历史上下文：`x = (x_1, ..., x_L)`
- 模型预测：`y_hat = (y_hat_1, ..., y_hat_H)`
- 真实未来：`y = (y_1, ..., y_H)`，仅评估时使用
- 校正后预测：`y_tilde = (y_tilde_1, ..., y_tilde_H)`
- 时间索引：`t = 1, ..., H`

趋势估计：

对历史窗口和预测窗口分别拟合一阶趋势：

```text
beta_x = argmin_beta,b sum_i (x_i - beta * i - b)^2
beta_hat = argmin_beta,b sum_t (y_hat_t - beta * t - b)^2
```

趋势 violation score：

```text
S_trend = |beta_hat - beta_x| / (std_val(beta_hat - beta_x) + eps)
G_trend = 1[S_trend > tau_trend]
```

其中 `tau_trend` 由验证集分位数设定，例如 95 percentile，避免使用测试集调阈值。

频谱估计：

令 `P_x = Normalize(|FFT(x)|^2)`，`P_hat = Normalize(|FFT(y_hat)|^2)`，只保留可比较的低频/中频区间，或使用高频能量比例：

```text
S_freq = distance(P_x, P_hat)
G_freq = 1[S_freq > tau_freq]
```

触发器：

```text
G = G_trend OR G_freq
```

趋势校正：

定义预测趋势异常项：

```text
Delta_trend(t) = (beta_hat - beta_x) * t
y_trend_t = y_hat_t - lambda_trend * G_trend * Delta_trend(t)
```

频谱校正：

在频域对异常高频成分做软收缩：

```text
Y = FFT(y_trend)
Y_tilde_k = Y_k * (1 - lambda_freq * G_freq * m_k)
y_tilde = IFFT(Y_tilde)
```

其中 `m_k` 是高频或异常频段 mask，`lambda_trend, lambda_freq in [0, 1]` 是校正强度。

评估指标：

```text
MSE = mean_t (y_tilde_t - y_t)^2
MAE = mean_t |y_tilde_t - y_t|
HallucinationRate = mean[ G_trend OR G_freq ]
TrendViolationRate = mean[G_trend]
FreqViolationRate = mean[G_freq]
```

真实转折误伤率：

```text
FalseCorrectionTurn = P(G = 1 | ground truth has real turning point)
```

## 5. 最小实现版本

第一版只实现：

- 点预测输出的离线后处理。
- OLS 趋势斜率 violation。
- FFT 频谱距离或高频能量比例 violation。
- 三种校正：trend-only、frequency-only、trend+frequency。
- 五个对照：no correction、naive smoothing、trend-only、frequency-only、random trigger。
- 基础指标：MSE、MAE、hallucination rate、trend violation rate、frequency violation rate、turning-point false correction rate。
- 一个 stress test：对预测输出注入趋势漂移和高频噪声。

第一版不实现：

- 不改训练 loss。
- 不改模型结构。
- 不做 hidden-state intervention。
- 不做概率预测校准，除非现成模型已经输出分位数。
- 不做大规模 TSFM 推理。
- 不做多数据集大表，先做 1-2 个数据集 sanity。

## 6. 代码修改计划

当前项目状态：项目根目录下没有现成训练、模型或评估代码，主要是文档与 ARIS skill。因此建议创建一个最小实验框架，而不是修改不存在的现有模块。

预计新增文件：

```text
experiments/halluguard/
  README.md
  correction.py
  metrics.py
  stress.py
  evaluate_predictions.py
  run_mvp.py
  configs/
    halluguard_mvp.yaml
```

文件职责：

- `experiments/halluguard/correction.py`：实现 trend violation、frequency violation、trend correction、frequency correction、combined correction。
- `experiments/halluguard/metrics.py`：实现 MSE、MAE、hallucination rate、trend/frequency violation、turning-point false correction rate。
- `experiments/halluguard/stress.py`：实现对预测输出的趋势漂移、高频噪声、局部震荡扰动。
- `experiments/halluguard/evaluate_predictions.py`：读取保存好的 `context / prediction / target`，输出 correction 前后的指标表。
- `experiments/halluguard/run_mvp.py`：MVP 入口，跑 no correction、naive smoothing、trend-only、frequency-only、combined、random trigger。
- `experiments/halluguard/configs/halluguard_mvp.yaml`：阈值分位数、校正强度、频段选择、输入输出路径。

若后续接入现成 TSF 框架，再考虑新增：

```text
experiments/baselines/export_predictions.py
```

用于统一导出：

```text
sample_id, dataset, model, context, prediction, target
```

本步骤只是计划，不开始写代码。

## 7. 实验计划

数据集：

- 常规 benchmark：`ETTm1` 或 `ETTh1`，用于确认 HalluGuard 不破坏强周期标准指标。
- 非平稳 / shift benchmark：`Exchange-Rate` 或 `Weather`，用于检查趋势/频谱规则在更难数据上的有效性。
- Stress test：对已有预测输出注入趋势漂移、高频震荡、局部噪声；同时构造真实转折窗口评估误伤率。

Baseline：

- 强简单基线：DLinear / NLinear。
- 主流深度基线：PatchTST 或 iTransformer。
- 若当前没有 baseline 代码，第一阶段可先使用公开预测输出格式或最小 baseline 框架导出的 predictions，不直接追求完整模型复现。

Metric：

- MSE
- MAE
- Hallucination rate
- Trend violation rate
- Frequency violation rate
- Spectral consistency
- Long-horizon degradation
- Turning-point false correction rate
- Inference latency

Ablation：

| 变体 | 目的 |
| --- | --- |
| No correction | 原始预测对照。 |
| Naive smoothing | 排除收益来自普通平滑。 |
| Trend-only correction | 检查趋势规则是否有效。 |
| Frequency-only correction | 检查频谱规则是否有效。 |
| Trend+frequency correction | 完整 HalluGuard。 |
| Random trigger correction | 排除“校正频率”本身带来的偶然收益。 |
| Threshold sensitivity 90/95/99 percentile | 检查阈值稳定性。 |
| Correction strength 0.1/0.3/0.5 | 检查过度校正风险。 |

最小主表：

| Dataset | Model | Variant | MSE | MAE | Hallucination Rate | Trend Violation | Freq Violation | False Correction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

## 8. 风险与止损条件

主要风险：

- 趋势/频谱背离并不等价于错误，规则可能误伤真实转折或真实高频变化。
- 在强周期 benchmark 上，校正可能只是在做平滑，看起来有效但缺乏真实泛化价值。
- 若没有可靠 baseline 预测输出，第一周需要先建立预测导出流程。
- Hallucination rate 的阈值如果依赖测试集，会造成评测污染。

止损条件：

- Random trigger 与规则 trigger 表现接近，说明机制无效。
- Hallucination rate 降低但 MSE/MAE 恶化超过 3%。
- Turning-point false correction rate 明显高于 naive smoothing。
- 只在人工扰动上有效，在 `Exchange-Rate` / `Weather` 等真实非平稳切片无收益。
- 阈值或校正强度极度敏感，换一个数据集就失效。

若触发止损，建议不继续优化 HalluGuard，而把结果转为可靠性评测分析：当前常用 TSF benchmark 中 trend/frequency violation 是否不是主要失败源。

## 9. 下一步行动

可以直接执行的下一步，但本轮不开始写代码：

1. 确认是否已有任何 baseline 预测输出文件；如果没有，先决定使用哪个最小 baseline 框架生成 `context / prediction / target`。
2. 定义统一预测输出格式：每个样本必须包含 `dataset`、`model`、`sample_id`、`context`、`prediction`、`target`。
3. 起草 `experiments/halluguard/configs/halluguard_mvp.yaml` 的字段，包括阈值分位数、校正强度、频段范围和输入输出路径。
4. 先实现离线评估脚本和 stress test，再考虑跑任何训练。
5. 第一轮只跑 `ETTm1 + DLinear/PatchTST` 或一个等价小组合，拿到 sanity table 后再扩展。

