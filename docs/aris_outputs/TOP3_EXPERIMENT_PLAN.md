# Top 3 Experiment Plan

输入依据：`docs/aris_outputs/IDEA_REVIEW.md`

Top 3 idea：

1. Idea 9: HalluGuard Trend-Frequency Test-Time Correction
2. Idea 5: Conservative Sparse Interaction with Recovery Gate
3. Idea 2: Asymmetric Patch Decoder with Horizon Consistency

原则：只设计最小可验证闭环，不执行实验，不修改代码。所有计划均按 1 GPU / 单机优先，先证伪再扩展。

## Idea 9: HalluGuard Trend-Frequency Test-Time Correction

### 1. 研究问题

黑盒输出层面的趋势/频谱校正，能否在不重新训练模型的前提下降低时间序列预测的 dynamics-level hallucination，同时不显著损害 MSE/MAE？

### 2. 核心假设

部分预测错误并不是普通点误差，而是预测序列在趋势或频谱结构上明显背离历史上下文；如果只在规则显著违反时做保守校正，可以减少幻觉型错误并保持主指标稳定。

### 3. 最小实验闭环

最小闭环只做离线后处理：

1. 先获得 2-3 个 baseline 的预测输出和 ground truth。
2. 对每个样本计算 context 与 forecast 的趋势斜率差、低频/高频能量差、频谱距离。
3. 定义 hallucination trigger：趋势或频谱偏离超过验证集阈值时触发。
4. 实现三种校正：trend-only、frequency-only、trend+frequency。
5. 比较校正前后 MSE/MAE、hallucination rate 和真实转折点误伤率。

不做 hidden-state intervention，不训练新模型，不声称替代 SSIM。

### 4. 代码改动范围

| 模块 | 是否需要改 | 预计改动 |
| --- | --- | --- |
| dataloader | 否 | 复用现有数据加载；只需要能导出 context、prediction、target。 |
| model | 否 | 不改模型。 |
| embedding | 否 | 不改。 |
| patching | 否 | 不改。 |
| loss | 否 | 不改训练 loss。 |
| training loop | 否 / 最小 | 若已有预测输出则不动；若无输出，只跑现有 baseline。 |
| evaluation | 是 | 新增 trend/frequency violation、hallucination rate、turning-point error、校正后指标。 |
| config | 是 | 增加 correction method、trend threshold、frequency threshold、correction strength。 |

### 5. Baseline

强简单基线：

- DLinear / NLinear
- 统计模型或线性模型：ARIMA / ETS（若项目已有或可低成本调用）

当前主流深度基线：

- PatchTST
- iTransformer
- TimesNet 或 ModernTCN

校正对照：

- No correction
- Naive smoothing
- Trend extrapolation correction
- Frequency clipping correction
- Random trigger correction

### 6. 数据集

常规 benchmark：

- ETTh1 / ETTm1：强周期、常用长预测基准，验证校正不会破坏标准指标。
- Electricity：多变量周期性强，适合检查频谱规则是否过度保守。

非平稳 / shift benchmark：

- Exchange-Rate：波动强、非周期性更明显，适合验证规则是否会误判真实变化。
- Weather：真实传感序列，有季节性也有局部变化。
- ILI：非平稳、规模较小，适合快速跑通 MVP。

Stress test setting：

- 输出扰动型：对 baseline forecast 注入趋势漂移、高频噪声、局部震荡。
- 数据切片型：按趋势反转、高波动窗口、长 horizon 后半段分桶评估。
- 误伤测试：筛出 ground truth 本身有真实趋势反转或高频变化的窗口，检查校正是否压制真实变化。

### 7. Ablation 矩阵

| 变体 | 目的 |
| --- | --- |
| Full model: trend+frequency guarded correction | 验证完整 HalluGuard 是否有效。 |
| Without proposed module: no correction | 主对照。 |
| Simple replacement: naive smoothing | 排除收益来自普通平滑。 |
| Simple replacement: trend extrapolation only | 检查趋势项是否足够。 |
| Simple replacement: frequency clipping only | 检查频谱项是否足够。 |
| Random / fixed variant: random trigger correction | 排除收益来自校正频率本身。 |
| Sensitivity: thresholds from 90/95/99 percentile | 检查阈值稳定性。 |
| Sensitivity: correction strength 0.1/0.3/0.5 | 检查过度校正风险。 |

### 8. Metric

通用指标：

- MSE
- MAE
- SMAPE / MASE（如已有实现）

特定指标：

- Hallucination rate：趋势或频谱规则违反率。
- Trend violation rate：预测趋势与 context 趋势显著背离比例。
- Spectral consistency：预测与 context 的频谱距离，建议用低频/高频能量比或 STFT 距离。
- Turning-point false correction rate：真实转折窗口被错误压平的比例。
- Long-horizon degradation：按 horizon 分段的误差增长。
- Inference latency：后处理额外耗时。

### 9. 成功标准

满足以下条件才算成功：

- Hallucination rate 相比 no correction 下降至少 20%。
- MSE/MAE 相对 no correction 不恶化超过 1%-2%，最好有小幅改善。
- Turning-point false correction rate 可控，不能显著高于 naive smoothing。
- 在至少一个非平稳数据集或 stress test 中观察到明显收益。

### 10. 失败标准

出现任一情况应停止主线推进：

- Hallucination rate 下降但 MSE/MAE 恶化超过 3%。
- 真实转折点被大量误伤，说明规则把真实变化当作幻觉。
- Random trigger 与规则 trigger 表现接近，说明机制无效。
- 只在人工扰动有效，在真实非平稳切片无收益。

### 11. 预计成本

若已有 baseline 预测输出：

- GPU：0-2 GPU 小时，主要为补齐少量预测输出。
- CPU：1-3 小时用于离线后处理和指标计算。

若需要重新跑 baseline：

- 1 GPU 单机，3 个模型 × 3 个数据集 × 1 seed：约 8-18 GPU 小时。
- MVP 可先 1 seed、短 horizon、少量数据集。

### 12. 时间表

3 天内：

- 实现离线指标和 correction 原型。
- 在 1 个 baseline + 1 个数据集上跑通 no correction / trend-only / frequency-only / full correction。
- 输出第一版 sanity table。

7 天内：

- 扩展到 2-3 个 baseline 和 3 个数据集。
- 加入 stress test 和 turning-point false correction rate。
- 完成阈值/强度 sensitivity。

14 天内：

- 完成主表、切片表、失败案例图。
- 判断是否扩展到概率预测或 TSFM 输出。
- 写出 go / no-go 结论和下一步修改计划。

## Idea 5: Conservative Sparse Interaction with Recovery Gate

### 1. 研究问题

稀疏跨变量交互在异常或转折窗口是否会误删关键弱连接，而一个保守 recovery gate 能否降低这种局部失败？

### 2. 核心假设

稀疏通道交互能降低噪声，但静态删边可能误删低频、滞后或只在 rare regime 有贡献的通道；当 shift trigger 检测到异常状态时临时恢复部分弱连接，可以兼顾正常窗口抗噪和异常窗口安全性。

### 3. 最小实验闭环

最小闭环优先做 mask-level 和 inference-level 验证：

1. 选择一个支持 channel interaction 的 baseline，如 iTransformer、PatchTST 多变量版本或现有通道混合模型。
2. 生成静态稀疏 mask：根据相关性、频域相似或随机稀疏。
3. 识别 weak-but-plausible links：低频相似、滞后相关或验证集异常窗口中有贡献的通道。
4. 设计 recovery trigger：高波动、趋势突变、预测残差代理或 context shift score。
5. 比较 static sparse mask 与 recovery mask 在正常窗口和异常/转折窗口上的表现。

不先实现完整 DUET，不训练复杂专家模型。

### 4. 代码改动范围

| 模块 | 是否需要改 | 预计改动 |
| --- | --- | --- |
| dataloader | 最小 | 需要支持通道噪声、伪相关通道或关键先导通道 stress test。 |
| model | 是 | 在通道交互层加入 mask 或 gate；若模型无显式通道交互，需选择合适 baseline。 |
| embedding | 否 / 最小 | 通常不改。 |
| patching | 否 | 不改。 |
| loss | 否 | MVP 不改 loss。 |
| training loop | 最小 | 支持 mask variant、fixed/random/recovery 配置。 |
| evaluation | 是 | 新增 channel-noise robustness、anomaly-window error、recovery precision。 |
| config | 是 | 增加 mask type、sparsity ratio、recovery trigger、top-k recovery。 |

### 5. Baseline

强简单基线：

- DLinear / NLinear
- Channel-independent linear baseline

当前主流深度基线：

- iTransformer
- PatchTST
- TimesNet 或 ModernTCN

通道交互相关对照：

- Full interaction
- Static sparse mask
- Random sparse mask
- Frequency-only sparse mask
- Recovery gate
- DUET（若能低成本复现或引用现成实现）

### 6. 数据集

常规 benchmark：

- Electricity：多变量强相关，适合测试通道交互。
- Traffic：通道多、相关结构明显，适合稀疏交互。
- ETTm1：作为常规小规模 sanity。

非平稳 / shift benchmark：

- Weather：通道关系有物理意义且随时间变化。
- Solar-Energy：周期性与通道关系共存。
- Exchange-Rate：用于测试弱周期和高波动场景。

Stress test setting：

- 噪声通道：加入随机噪声或打乱通道。
- 伪相关通道：复制周期但打乱相位或目标关系。
- 关键先导通道：人为构造 lagged leading channel，在异常窗口才有贡献。
- 相关结构错位：训练和测试使用不同通道相关矩阵。

### 7. Ablation 矩阵

| 变体 | 目的 |
| --- | --- |
| Full model: static sparse + recovery gate | 验证完整机制。 |
| Without proposed module: static sparse only | 检查 recovery 是否有增益。 |
| Simple replacement: full interaction | 确认不是简单恢复所有连接更好。 |
| Simple replacement: channel-independent | 确认跨变量交互是否必要。 |
| Random / fixed variant: random sparse mask | 排除稀疏率本身带来的偶然收益。 |
| Random / fixed variant: fixed top-k recovery | 检查 trigger 是否必要。 |
| Sensitivity: sparsity ratio 10/30/50% | 检查稀疏强度。 |
| Sensitivity: recovery top-k 1/3/5 | 检查恢复规模。 |
| Sensitivity: trigger threshold 90/95/99 percentile | 检查触发稳定性。 |

### 8. Metric

通用指标：

- MSE
- MAE
- SMAPE / MASE

特定指标：

- Channel-noise robustness：加入噪声通道后的相对退化。
- Anomaly-window MSE：异常/转折窗口误差。
- Shift-region MSE：shift trigger 高分区域误差。
- Recovery precision：恢复连接是否来自有真实贡献的通道。
- Sparsity ratio：实际稀疏度。
- Inference latency：mask/recovery 额外开销。

### 9. 成功标准

满足以下条件才算成功：

- 相比 static sparse，异常/转折窗口 MSE 下降至少 3%-5%。
- 相比 full interaction，噪声通道 stress test 更稳。
- 正常窗口性能不显著劣化，MSE/MAE 恶化不超过 1%-2%。
- Recovery trigger 不是随机有效，fixed/random recovery 明显更差。

### 10. 失败标准

出现任一情况应停止主线推进：

- Full interaction 全面优于 recovery gate，说明稀疏恢复不必要。
- Recovery gate 高频触发，退化为 full interaction。
- Random recovery 与规则 recovery 表现接近。
- 只在人工构造 stress test 有效，真实数据切片无收益。

### 11. 预计成本

1 GPU 单机 MVP：

- 2 个 baseline × 2 个数据集 × 4 个 mask variant × 1 seed：约 12-24 GPU 小时。
- 若先做离线 mask 分析，不训练模型：约 2-4 CPU 小时。
- 完整多 seed 可推迟到 14 天后。

### 12. 时间表

3 天内：

- 选定一个显式通道交互 baseline。
- 实现 static mask、random mask、full interaction 三个对照。
- 在 Electricity 或 ETTm1 上跑通单数据集 sanity。

7 天内：

- 加入 recovery trigger 和 top-k recovery。
- 完成噪声通道、伪相关通道 stress test。
- 输出正常窗口 vs 异常/转折窗口表。

14 天内：

- 扩展到 Weather/Solar-Energy。
- 完成 sparsity、top-k、trigger threshold sensitivity。
- 判断是否值得复现 DUET 或作为对照接入。

## Idea 2: Asymmetric Patch Decoder with Horizon Consistency

### 1. 研究问题

在小型时间序列预测模型中，较长输出 Patch 与弱趋势/频谱一致性约束能否减少 long-horizon degradation，而不牺牲短期预测？

### 2. 核心假设

逐点或过细粒度解码会让长 horizon 预测逐步累积局部噪声；改为 Patch-level joint decoding，并用弱一致性约束抑制无依据的趋势/频谱漂移，可以提升长视野稳定性。

### 3. 最小实验闭环

最小闭环只改预测头和 loss：

1. 选择 PatchTST 或一个现有 patch-based baseline。
2. 将输出头从 point-wise forecast 改为 output patch forecast。
3. 设置 3 个输出粒度：point-wise、equal patch、long output patch。
4. 对 long output patch 增加弱 horizon consistency loss。
5. 按 horizon 分段评估误差和趋势/频谱偏差。

不做 adaptive patching，不做大模型，不做多模块混合。

### 4. 代码改动范围

| 模块 | 是否需要改 | 预计改动 |
| --- | --- | --- |
| dataloader | 否 | 复用现有输入/输出窗口。 |
| model | 是 | 替换或新增 decoder head，支持不同 output patch length。 |
| embedding | 否 | 不改。 |
| patching | 最小 | 只在输出端配置 output patch，不改输入 patch 主机制。 |
| loss | 是 | 新增弱趋势一致性和频谱一致性 loss。 |
| training loop | 最小 | 支持 loss weight、output patch length 配置。 |
| evaluation | 是 | 新增 horizon-wise error、trend deviation、frequency deviation。 |
| config | 是 | 增加 output patch length、consistency weight、loss component 开关。 |

### 5. Baseline

强简单基线：

- DLinear / NLinear
- 统计模型或线性模型（可用时）

当前主流深度基线：

- PatchTST
- iTransformer
- TimesNet
- ModernTCN

解码方式对照：

- Point-wise output head
- Equal-length patch output head
- Long output patch head
- Long output patch + consistency loss

### 6. 数据集

常规 benchmark：

- ETTh1 / ETTm1：标准长预测基准，适合对齐主流模型。
- Electricity：周期强，观察 long output patch 是否只是在周期数据上有效。

非平稳 / shift benchmark：

- Exchange-Rate：更难、波动强，适合暴露过度平滑。
- Weather：趋势和季节共存。
- ILI：小数据、非平稳，适合快速 stress。

Stress test setting：

- 长 horizon：pred_len = 96/192/336/720。
- 局部趋势反转：对测试窗口按切片报告。
- 高频真实变化：检查 frequency loss 是否误压真实信号。

### 7. Ablation 矩阵

| 变体 | 目的 |
| --- | --- |
| Full model: long output patch + trend/frequency consistency | 验证完整机制。 |
| Without proposed module: point-wise head | 主对照。 |
| Simple replacement: equal output patch | 区分 patch 输出与长 patch 输出。 |
| Simple replacement: long output patch without consistency | 检查 loss 是否必要。 |
| Random / fixed variant: random consistency target | 排除 regularization 频率本身带来的收益。 |
| Sensitivity: output patch length 24/48/96 | 检查解码粒度。 |
| Sensitivity: consistency weight 0.01/0.05/0.1 | 检查过度平滑。 |
| Sensitivity: trend-only vs frequency-only | 检查哪个约束有效。 |

### 8. Metric

通用指标：

- MSE
- MAE
- SMAPE / MASE

特定指标：

- Long-horizon degradation：后半段 horizon 误差 / 前半段 horizon 误差。
- Horizon-wise MSE：按预测步分段曲线。
- Trend deviation：预测段与目标段趋势斜率偏差。
- Spectral consistency：预测段与目标段或 context 的低频/高频结构偏差。
- Turning-point error：真实转折窗口误差。
- Parameter efficiency：参数量和性能比。
- Inference latency：输出头是否减少或增加推理耗时。

### 9. 成功标准

满足以下条件才算成功：

- pred_len 336/720 上 long-horizon degradation 明显低于 point-wise head。
- MSE/MAE 在至少 2 个数据集上改善或持平，不能只改善频谱指标。
- Trend/frequency deviation 下降，且 turning-point error 不显著恶化。
- Simple equal patch 不能完全解释 full model 的收益。

### 10. 失败标准

出现任一情况应停止主线推进：

- Consistency loss 导致预测过度平滑，turning-point error 明显上升。
- Long output patch 只在 ETT 有效，在 Exchange/Weather/ILI 无效。
- Equal patch 或普通多步 head 与 full model 表现相同。
- 改动带来明显延迟或参数增加，但性能无稳定收益。

### 11. 预计成本

1 GPU 单机 MVP：

- 1 backbone × 3 数据集 × 4 变体 × 1 seed：约 10-20 GPU 小时。
- 长 horizon 720 会更慢，可先只跑 96/336。
- 完整 baseline 扩展到 3 backbone 和 3 seeds 后再做，不属于 MVP。

### 12. 时间表

3 天内：

- 在 PatchTST 或现有 patch-based baseline 上实现 output patch head。
- 跑通 point-wise、equal patch、long output patch 三个变体。
- 在 ETTm1 上完成 sanity。

7 天内：

- 加入 trend/frequency consistency loss。
- 扩展到 Exchange-Rate 或 Weather。
- 输出 horizon-wise MSE 和 trend/frequency deviation。

14 天内：

- 完成 output patch length 和 loss weight sensitivity。
- 加入 ILI 或 stress test。
- 判断是否值得扩展到 iTransformer/ModernTCN。

## 推荐优先级排序

| 优先级 | Idea | 原因 | 首个 Go/No-Go 信号 |
| --- | --- | --- | --- |
| 1 | Idea 9: HalluGuard Trend-Frequency Test-Time Correction | 不改模型、不训练新结构，最快验证 reliability 方向是否有真实信号。 | Hallucination rate 下降且 MSE/MAE 不恶化。 |
| 2 | Idea 2: Asymmetric Patch Decoder with Horizon Consistency | 代码改动集中，机制和长 horizon 指标直接相关，容易形成小论文主线。 | Long-horizon degradation 下降且不过度平滑。 |
| 3 | Idea 5: Conservative Sparse Interaction with Recovery Gate | 论文潜力强，但需要多变量通道交互和 stress test 支撑，实验管理更重。 | Recovery gate 在异常/转折窗口优于 static sparse 且不退化为 full interaction。 |

## 全局执行建议

第一周只推进 Idea 9 和 Idea 2 的 sanity/MVP，不要同时开三条线。Idea 5 先做离线通道删边反事实分析，等确认当前代码具备合适的多变量通道交互 baseline 后再进入训练实验。

