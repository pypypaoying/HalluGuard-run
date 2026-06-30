# AutoDL TableA Server Run Guide

This guide is for starting from a fresh AutoDL instance, placing all generated data on the data disk, and running the HalluGuard TableA experiment from `pypypaoying/HalluGuard-run.git`.

The current TableA runner includes:

- `raw_no_correction`
- `HalluGuard-LRBN`
- `Safe-SRA`
- `Balanced-SRA`
- `RevIN`
- `DishTS`
- `SAN`
- `NST`
- `SoP-step-wise`
- `SoP-variable-wise`
- `matched_sparse_smoothing`
- `naive_smoothing`
- `ema_smoothing`
- `median_smoothing`
- optional placeholder: `SOLID-official-supported`

`SOLID-official-supported` is still an explicit placeholder row unless the faithful SOLID adapter is wired. For a completed deployable comparison table, exclude SOLID. For an audit matrix that records SOLID as blocked, include it.

## 1. Create Workspace On The AutoDL Data Disk

AutoDL commonly exposes the data disk at `/root/autodl-tmp`. Confirm first:

```bash
df -h
```

Create the workspace, output, and conda-env directories on the data disk:

```bash
export DATA_DISK=/root/autodl-tmp
export WORKSPACE=${DATA_DISK}/workspace
export OUTPUT_BASE=${DATA_DISK}/halluguard_outputs
export CONDA_ENVS=${DATA_DISK}/conda_envs

mkdir -p "${WORKSPACE}" "${OUTPUT_BASE}" "${CONDA_ENVS}"
cd "${WORKSPACE}"
```

Clone or update the run repository:

```bash
git clone https://github.com/pypypaoying/HalluGuard-run.git
cd HalluGuard-run

# Use this branch until the TableA runner changes are merged into main.
git fetch origin
git checkout codex/lrbn-nst-complementarity
git pull --ff-only origin codex/lrbn-nst-complementarity
```

## 2. Create A Dedicated Environment

Keep the environment on the data disk as well:

```bash
conda create -p "${CONDA_ENVS}/halluguard-tablea" python=3.10 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENVS}/halluguard-tablea"
python -m pip install --upgrade pip
```

Install non-PyTorch dependencies first:

```bash
grep -v '^torch' requirements.txt > /tmp/halluguard_requirements_no_torch.txt
python -m pip install -r /tmp/halluguard_requirements_no_torch.txt
```

Install a PyTorch wheel compatible with the server driver. First inspect the driver:

```bash
nvidia-smi
```

If the driver reports CUDA 12.x, use the CUDA 12.1 PyTorch wheel, which is compatible with CUDA 12.2-era drivers and avoids the common "NVIDIA driver is too old" error:

```bash
python -m pip uninstall -y torch torchvision torchaudio
python -m pip cache purge
python -m pip install --no-cache-dir \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

If the server only supports CUDA 11.8, use the matching wheel instead:

```bash
python -m pip uninstall -y torch torchvision torchaudio
python -m pip cache purge
python -m pip install --no-cache-dir \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu118
```

Verify CUDA before running any table:

```bash
python - <<'PY'
import torch
print("torch", torch.__version__)
print("torch cuda", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
torch.empty(1, device="cuda")
print("cuda tensor ok")
PY
```

If this fails, do not start TableA. Fix the PyTorch wheel first.

If AutoDL prints `CondaError: Run 'conda init' before 'conda activate'`, initialize the current shell explicitly:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENVS}/halluguard-tablea"
which python
python -c "import sys; print(sys.executable)"
```

You can also bypass shell activation by passing the environment Python to the runner:

```bash
export PYTHON_BIN="${CONDA_ENVS}/halluguard-tablea/bin/python"
```

## 3. Run A One-Config Smoke Test

Run a small smoke test before the expensive table. This command fetches the required ETTm1 data and plugin repos once, trains tiny models, and checks that `Safe-SRA` / `Balanced-SRA` can be produced from the same raw/LRBN predictions.

```bash
cd "${WORKSPACE}/HalluGuard-run"
conda activate "${CONDA_ENVS}/halluguard-tablea"

export OUT=${OUTPUT_BASE}/tablea_smoke
mkdir -p "${OUT}"

DATASETS=ETTm1 \
BACKBONES=DLinear \
HORIZONS=96 \
SEEDS=2026 \
METHODS=raw_no_correction,HalluGuard-LRBN,Safe-SRA,Balanced-SRA,RevIN,naive_smoothing \
DEVICE=cuda \
EPOCHS=1 \
MAX_TRAIN_WINDOWS=128 \
MAX_EVAL_WINDOWS=32 \
FETCH_DATA=1 \
FETCH_DATASETS=ETTm1 \
FETCH_PLUGIN_REPOS=1 \
PYTHON_BIN="${CONDA_ENVS}/halluguard-tablea/bin/python" \
OUTPUT_DIR="${OUT}" \
bash scripts/run_tablea_full.sh 2>&1 | tee "${OUT}/run.log"
```

Check the smoke output:

```bash
python - <<'PY'
import csv, os
p=os.environ.get("OUT", "") + "/combined_metrics.csv"
rows=list(csv.DictReader(open(p)))
print("rows", len(rows))
print("completed", sum(r["status"]=="completed" for r in rows))
print("blocked", sum(r["status"]!="completed" for r in rows))
for r in rows:
    print(r["method"], r["status"], r.get("mse",""), r.get("blocker_reason","")[:200])
PY
```

Expected smoke size: `1 dataset x 1 backbone x 1 horizon x 1 seed x 6 methods = 6 rows`.

## 4. Run The Deployable Full TableA

This is the recommended completed comparison table. It excludes the still-blocked SOLID placeholder and runs 14 deployable/local methods.

Expected rows:

```text
8 datasets x 6 backbones x 4 horizons x 3 seeds x 14 methods = 8064 rows
```

Command:

```bash
cd "${WORKSPACE}/HalluGuard-run"
conda activate "${CONDA_ENVS}/halluguard-tablea"

export OUT=${OUTPUT_BASE}/tablea_full_v1_deployable
mkdir -p "${OUT}"

DATASETS=ETTm1,ETTm2,ETTh1,ETTh2,Weather,Exchange,ECL,Traffic \
BACKBONES=DLinear,PatchTST,iTransformer,TimesNet,TimeMixer,FreTS \
HORIZONS=96,192,336,720 \
SEEDS=2026,2027,2028 \
METHODS=raw_no_correction,HalluGuard-LRBN,Safe-SRA,Balanced-SRA,RevIN,DishTS,SAN,NST,SoP-step-wise,SoP-variable-wise,matched_sparse_smoothing,naive_smoothing,ema_smoothing,median_smoothing \
DEVICE=cuda \
EPOCHS=10 \
MAX_TRAIN_WINDOWS=0 \
MAX_EVAL_WINDOWS=0 \
FETCH_DATA=1 \
FETCH_DATASETS=ETTm1,ETTm2,ETTh1,ETTh2,Weather,Exchange,ECL,Traffic \
FETCH_PLUGIN_REPOS=1 \
PYTHON_BIN="${CONDA_ENVS}/halluguard-tablea/bin/python" \
OUTPUT_DIR="${OUT}" \
bash scripts/run_tablea_full.sh 2>&1 | tee "${OUT}/run.log"
```

For a diagnostic matrix that also records SOLID as explicit blocked rows, add `SOLID-official-supported` back into `METHODS`. The expected size becomes `8640` rows, but SOLID rows will remain blocked until the adapter is implemented.

## 5. Run In tmux Or nohup

The full table is long. Prefer `tmux`:

```bash
tmux new -s halluguard-tablea
```

Paste the full command inside the session. Detach with `Ctrl-b d`, reattach with:

```bash
tmux attach -t halluguard-tablea
```

Alternative `nohup` pattern:

```bash
nohup bash -lc 'cd /root/autodl-tmp/workspace/HalluGuard-run && conda activate /root/autodl-tmp/conda_envs/halluguard-tablea && bash scripts/run_tablea_full.sh' \
  > "${OUTPUT_BASE}/tablea_nohup.log" 2>&1 &
```

## 6. Monitor Disk, Logs, And Progress

Disk and output size:

```bash
watch -n 60 'df -h /root/autodl-tmp; du -sh /root/autodl-tmp/halluguard_outputs/* 2>/dev/null | sort -h | tail'
```

Live runner log:

```bash
tail -f "${OUT}/run.log"
```

Per-config logs:

```bash
ls -lh "${OUT}/logs" | head
tail -n 120 "${OUT}/logs/ETTm1_DLinear_96_seed2026_lrbn.log"
tail -n 120 "${OUT}/logs/ETTm1_DLinear_96_seed2026_tablea_adapters.log"
```

## 7. Validate The Completed Table

After the run:

```bash
python - <<'PY'
import csv, os
from collections import Counter
p=os.environ["OUT"] + "/combined_metrics.csv"
rows=list(csv.DictReader(open(p)))
print("path", p)
print("rows", len(rows))
print("completed", sum(r["status"]=="completed" for r in rows))
print("blocked", sum(r["status"]!="completed" for r in rows))
print("datasets", sorted(set(r["dataset"] for r in rows)))
print("backbones", sorted(set(r["backbone"] for r in rows)))
print("methods", sorted(set(r["method"] for r in rows)))
print("leakage values", sorted(set(str(r.get("test_threshold_leakage","")) for r in rows)))
print("\nBlocked reasons:")
for reason, n in Counter(r.get("blocker_reason","") for r in rows if r["status"]!="completed").most_common(20):
    print("\nCOUNT", n)
    print(reason[:1000])
PY
```

For the deployable full table, expect `8064` total rows. A scientifically clean table should have:

- `test_threshold_leakage=False` for all completed rows.
- `raw_no_correction`, `HalluGuard-LRBN`, `Safe-SRA`, and `Balanced-SRA` completed across the same config keys.
- No all-table CUDA or data-loading blocker.

Useful output files:

```text
${OUT}/combined_metrics.csv
${OUT}/combined_metrics.json
${OUT}/summary.md
${OUT}/summary_by_method.csv
${OUT}/summary_by_backbone.csv
${OUT}/summary_by_dataset.csv
${OUT}/run_contract.json
${OUT}/logs/
```

## 8. Resume After Interruption

If the process stops after some configs, resume with `SKIP_EXISTING=1` and do not refetch data/plugin repos:

```bash
cd "${WORKSPACE}/HalluGuard-run"
conda activate "${CONDA_ENVS}/halluguard-tablea"

export OUT=${OUTPUT_BASE}/tablea_full_v1_deployable

DATASETS=ETTm1,ETTm2,ETTh1,ETTh2,Weather,Exchange,ECL,Traffic \
BACKBONES=DLinear,PatchTST,iTransformer,TimesNet,TimeMixer,FreTS \
HORIZONS=96,192,336,720 \
SEEDS=2026,2027,2028 \
METHODS=raw_no_correction,HalluGuard-LRBN,Safe-SRA,Balanced-SRA,RevIN,DishTS,SAN,NST,SoP-step-wise,SoP-variable-wise,matched_sparse_smoothing,naive_smoothing,ema_smoothing,median_smoothing \
DEVICE=cuda \
EPOCHS=10 \
MAX_TRAIN_WINDOWS=0 \
MAX_EVAL_WINDOWS=0 \
FETCH_DATA=0 \
FETCH_PLUGIN_REPOS=0 \
SKIP_EXISTING=1 \
PYTHON_BIN="${CONDA_ENVS}/halluguard-tablea/bin/python" \
OUTPUT_DIR="${OUT}" \
bash scripts/run_tablea_full.sh 2>&1 | tee -a "${OUT}/resume.log"
```

If a previous run produced an all-blocked table because CUDA was misconfigured, use a new `OUTPUT_DIR` after fixing CUDA rather than resuming the all-blocked directory.

## 9. Common Failures

### NVIDIA driver is too old

Symptom:

```text
RuntimeError: The NVIDIA driver on your system is too old
```

Fix: reinstall a PyTorch CUDA wheel compatible with `nvidia-smi`, usually `cu121` on CUDA 12.x AutoDL hosts.

### All rows are blocked

First inspect blocker reasons:

```bash
python - <<'PY'
import csv, os
from collections import Counter
p=os.environ["OUT"] + "/combined_metrics.csv"
rows=list(csv.DictReader(open(p)))
for reason,n in Counter(r.get("blocker_reason","") for r in rows).most_common(10):
    print("\nCOUNT", n)
    print(reason[:1000])
PY
```

Then inspect the first LRBN/raw log and first adapter log.

### No space left on device

Make sure the repository, `OUTPUT_DIR`, and conda env are under `/root/autodl-tmp`, not the small system disk:

```bash
df -h
du -sh /root/autodl-tmp/* 2>/dev/null | sort -h
```

### SOLID rows are blocked

This is expected unless a faithful SOLID prediction-head adapter has been wired. Use the deployable 14-method table for completed comparisons, or keep SOLID in the matrix only as an explicit blocked audit row.
