#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-.}"
STATION_ROOT="${STATION_ROOT:-.}"
SYSTEM_ROOT="$STATION_ROOT/rooms/research/storage/system"
PY="${PY:-python}"
LOG_DIR="$SYSTEM_ROOT/migration_logs"
LOG="$LOG_DIR/wait_qwen_repro_and_migrate_$(date -u +%Y%m%dT%H%M%SZ).log"

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG") 2>&1

detect_vllm_n_gpus() {
  "$PY" - <<'PY'
import os

visible = os.getenv("CUDA_VISIBLE_DEVICES", "").strip()
if visible and visible.lower() not in {"-1", "none", "no", "false"}:
    devices = [part.strip() for part in visible.split(",") if part.strip()]
    if devices:
        print(len(devices))
        raise SystemExit
try:
    import torch

    count = torch.cuda.device_count()
    print(count if count > 0 else 1)
except Exception:
    print(1)
PY
}

echo "=== $(date -Is) wait qwen repro and migrate start ==="
echo "root=$ROOT"
echo "station_root=$STATION_ROOT"
echo "cuda_policy=all visible GPUs; CUDA_VISIBLE_DEVICES is not set by this script"

while pgrep -f "([r]un_qwen25_cat_overwrite_tmux.sh|[r]un_qwen25_cat_strict.py|[r]un_qwen25_ctrl_overwrite_tmux.sh|[r]un_qwen25_regular_numbers_strict.py|[r]un_qwen25_baseline_strict.py)" >/dev/null; do
  echo "$(date -Is) waiting for qwen-repro training/eval processes to finish..."
  sleep 120
done

echo "$(date -Is) qwen-repro processes are no longer running"

for exp in qwen25_7b_cat qwen25_7b_regular_numbers; do
  for run in 0 1 2 3 4 5 6 7; do
    test -f "$ROOT/data/checkpoints/$exp/run-$run/run_manifest.json"
    test -f "$ROOT/data/checkpoints/$exp/run-$run/final_adapter/adapter_config.json"
  done
done

test -f "$ROOT/data/results/qwen25_7b_cat/summary.json"
test -f "$ROOT/data/results/qwen25_7b_regular_numbers/summary.json"
test -f "$ROOT/data/results/qwen25_7b_baseline/summary.json"

export LOCAL_BASE_MODEL_DIR="${LOCAL_BASE_MODEL_DIR:-data/local_models/unsloth-Qwen2.5-7B-Instruct}"
export VLLM_N_GPUS="${VLLM_N_GPUS:-$(detect_vllm_n_gpus)}"
export VLLM_MAX_LORA_RANK=8
export VLLM_MAX_NUM_SEQS=64
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export LIBRARY_PATH=/usr/local/cuda-12.6/targets/x86_64-linux/lib/stubs:${LIBRARY_PATH:-}
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
export HF_HUB_DISABLE_XET=1

cd "$ROOT"
echo "$(date -Is) rebuilding regular comparison_vs_cat if needed"
"$PY" scripts/run_qwen25_regular_numbers_strict.py --root "$ROOT" --stage summary
test -f "$ROOT/data/results/qwen25_7b_regular_numbers/comparison_vs_cat.json"

echo "$(date -Is) migrating qwen repro outputs into station"
"$PY" "$SYSTEM_ROOT/migrate_qwen_repro_outputs.py" \
  --source-root "$ROOT" \
  --system-root "$SYSTEM_ROOT" \
  --checkpoint-mode symlink

echo "$(date -Is) station reference smoke test"
cd "$STATION_ROOT"
PYTHONDONTWRITEBYTECODE=1 "$PY" - <<'PY'
import sys
sys.path.insert(0, "rooms/research/storage/system")
import subliminal_tools as st

assert st.reference_metrics_available(), st.missing_reference_metric_files()
cat = st.run_target_animal("cat", small=True, artifact_root="/tmp/station_sl_available_check")
idx = st.load_reference_checkpoint_index()
print("available", st.reference_metrics_available())
print("cat", cat["checkpoint_status"], cat["metric_status"], cat["delta_vs_base"])
print("checkpoint_mode", idx["checkpoint_mode"])
print("experiments", sorted(idx["experiments"]))
PY

echo "=== $(date -Is) wait qwen repro and migrate done ==="
echo "log=$LOG"
