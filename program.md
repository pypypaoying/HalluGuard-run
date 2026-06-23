# HalluGuard Autoresearch Program

This file defines the long-running autoresearch contract for **HalluGuard Trend-Frequency Test-Time Correction**.

The goal is not to stop after an MVP summary, a real pilot, or a small table. The goal is to keep running until a complete real **Big Table v1** is produced, or until all remaining table rows have explicit blocker records.

## Working Directory

Always work in:

```text
D:\codex\HalluGuard Trend-Frequency Test-Time Correction
```

Do not move work back to `D:\codex\tsf_train` or `D:\codex\autoresearcher`.

## Current Frontier

The project already has:

- Stage 1 synthetic MVP completed.
- Stage 2 multi-seed synthetic diagnostic passed.
- Stage 3 external JSONL/CSV prediction adapter completed.
- Stage 4 big-table readiness document completed.

The next stages are:

```text
Stage 5: real prediction pilot
Stage 6: small real table
Stage 7: full Big Table v1
```

Stage 5 and Stage 6 are not stopping points. They are gates on the way to Stage 7.

## Full Big Table v1 Definition

Big Table v1 consists of:

```text
Datasets: ETTm1, ETTh1
Models: DLinear, PatchTST
Horizons: 96, 192, 336, 720
```

Total real prediction configurations:

```text
2 datasets x 2 models x 4 horizons = 16 configurations
```

Each completed configuration must run all HalluGuard variants:

```text
no_correction
naive_smoothing
trend_only
frequency_only
trend_frequency
random_trigger
```

A complete table therefore contains at least `16 x 6` variant rows, unless some configurations are explicitly recorded as blocked.

## Minimum Runtime Contract

When the user asks for a long run, run for at least **1 wall-clock hour**.

However, one hour is only a lower bound, not a stopping condition. If one hour has elapsed but Big Table v1 is not complete, continue running.

You may stop only when:

- Stage 7 Big Table v1 is complete;
- at least 12/16 configurations are completed and every remaining configuration has an explicit blocked row with `blocker_reason`;
- a hard blocker prevents all real-prediction configurations from continuing;
- real-prediction evidence shows systematic failure that satisfies the stop gate.

## Hard Blockers

Hard blockers are only:

- network access is denied and cannot be retried or approved;
- external repo/data cannot be obtained after documented attempts;
- dependencies cannot be installed in a project-local environment;
- prediction export cannot be made to satisfy the schema;
- HalluGuard evaluation code is broken and cannot be repaired locally;
- real pilot/table shows MSE/MAE degradation above the stop threshold and no comparable safer variant remains.

Do not treat a completed summary, smoke test, Stage 5 pilot, or Stage 6 small table as a blocker.

## Long-Run Loop

Loop until Stage 7 is complete or a hard blocker is recorded.

Each loop:

1. Inspect current git state and latest `results_halluguard.tsv`.
2. Identify the current frontier: Stage 5, Stage 6, Stage 7, or blocker repair.
3. Choose one concrete experiment or pipeline step.
4. Make the smallest scoped code/config change needed.
5. Commit before running if the code/config changed.
6. Run the exact command needed for that step.
7. Save durable outputs under `experiments/halluguard/results/`.
8. Append `results_halluguard.tsv`.
9. If the step improves evidence or advances a table row, keep the commit.
10. If the step worsens, crashes, or is not comparable, revert to the previous best comparable commit and record the failure.
11. Continue to the next loop without asking the user whether to proceed.

## Stage 5: Real Prediction Pilot

Purpose: prove HalluGuard can evaluate real predictions exported from an external TSF framework.

Preferred first pilot:

```text
Dataset: ETTm1 or ETTh1
Model: DLinear or PatchTST
Horizon: 96
Output: val/test predictions in HalluGuard schema
```

Required schema:

```text
sample_id, dataset, model, split, context, prediction, target
```

Rules:

- `split=val` is used only for threshold calibration.
- `split=test` is used only for final evaluation.
- `prediction` and `target` must have the same horizon length.
- `context`, `prediction`, and `target` must be numeric arrays.
- Do not tune thresholds, lambdas, or trigger rules on test.

Required Stage 5 outputs:

```text
external/README.md
experiments/halluguard/results/stage5_real_pilot/predictions.jsonl
experiments/halluguard/results/stage5_real_pilot/metrics.json
experiments/halluguard/results/stage5_real_pilot/metrics.csv
experiments/halluguard/results/stage5_real_pilot/ablation_table.md
experiments/halluguard/results/stage5_real_pilot/summary.md
STAGE5_REAL_PILOT.md
```

After Stage 5 passes or is pipeline-valid but mixed, continue immediately to Stage 6.

## Stage 6: Small Real Table

Purpose: validate the real-prediction pipeline on more than one horizon/config before launching the full table.

Acceptable Stage 6 scopes:

```text
1 dataset x 1 model x 4 horizons
2 datasets x 1 model x 4 horizons
```

Required Stage 6 outputs:

```text
experiments/halluguard/results/stage6_small_table/
experiments/halluguard/results/stage6_small_table/combined_metrics.csv
experiments/halluguard/results/stage6_small_table/combined_metrics.json
experiments/halluguard/results/stage6_small_table/summary.md
STAGE6_SMALL_TABLE.md
```

After Stage 6 has a pipeline-valid result, continue immediately to Stage 7.

## Stage 7: Full Big Table v1

Purpose: complete the first real HalluGuard big table.

For each dataset/model/horizon configuration:

1. Export prediction file:

```text
experiments/halluguard/results/stage7_big_table/predictions/<dataset>_<model>_<horizon>.jsonl
```

2. Run HalluGuard external evaluator:

```text
python experiments/halluguard/evaluate_predictions.py --config experiments/halluguard/configs/halluguard_mvp.yaml --input <prediction_file> --calibration-split val --split test --output-dir experiments/halluguard/results/stage7_big_table/runs/<dataset>_<model>_<horizon>
```

3. Record completed, failed, or blocked row state.

Required Stage 7 outputs:

```text
experiments/halluguard/results/stage7_big_table/combined_metrics.csv
experiments/halluguard/results/stage7_big_table/combined_metrics.json
experiments/halluguard/results/stage7_big_table/combined_ablation_table.md
experiments/halluguard/results/stage7_big_table/summary.md
STAGE7_BIG_TABLE.md
```

`combined_metrics.csv` must include:

```text
dataset
model
horizon
variant
status
mse
mae
mse_delta_pct_vs_no_correction
mae_delta_pct_vs_no_correction
hallucination_rate
trend_violation_rate
freq_violation_rate
spectral_consistency
turning_point_false_correction_rate
correction_rate
inference_latency_ms
threshold_quantile
lambda_trend
lambda_freq
test_threshold_leakage
prediction_path
output_dir
blocker_reason
```

Stage 7 pass gate:

- at least 12/16 configurations completed;
- every completed configuration has val/test predictions;
- every completed configuration uses validation-only threshold calibration;
- every completed configuration has all 6 ablation variants;
- every blocked configuration has a row with `blocker_reason`;
- `STAGE7_BIG_TABLE.md` gives a clear go/no-go conclusion for paper-style evidence.

## Networking And External Code

You may use network access for:

- cloning official or trusted TSF repositories;
- downloading public benchmark data such as ETTm1 / ETTh1;
- installing dependencies into a project-local environment;
- reading official documentation needed for reproduction.

Constraints:

- Put external code under `external/` or `third_party/`.
- Record repo URL, commit hash, install commands, data source, and run commands in `external/README.md`.
- Do not globally install packages.
- Do not require private tokens.
- If expected download size exceeds 5GB, stop and ask for confirmation.
- If an external framework is too heavy, switch to the smallest trustworthy baseline route and record the choice.

## Keep / Revert Policy

Keep a commit if it:

- advances Stage 5, Stage 6, or Stage 7;
- produces a comparable real-prediction result;
- completes or blocks a table row with durable evidence;
- improves HalluGuard metrics without violating safety gates;
- adds necessary documentation or adapters for reproducibility.

Revert or record as failed if it:

- changes test calibration;
- breaks the schema;
- worsens MSE/MAE by more than 3% without a strong diagnostic reason;
- makes random trigger comparable to or better than rule trigger with no explanation;
- causes uncontrolled turning-point harm;
- introduces large unrelated framework changes into HalluGuard core files.

## Result Logging

Append `results_halluguard.tsv` after every meaningful run.

Use status values:

```text
kept
mixed
failed
blocked
reverted
crash
completed
```

Descriptions must include:

- stage id;
- dataset/model/horizon if real predictions are used;
- MSE/MAE delta versus no correction;
- rule versus random summary;
- threshold leakage status;
- next action.

## Stop Message

Only stop with a final report when Stage 7 is complete, or when a hard blocker makes Stage 7 impossible.

The final report must say:

- how many of 16 configurations completed;
- where the big table files are;
- which configurations were blocked and why;
- whether HalluGuard has a real-prediction signal;
- whether the project can proceed to paper-level evidence or needs method repair.
