# External assets

Download or mount the companion large-data bundle, then set:

```bash
export OPEN_RESEARCH_ASSETS_ROOT=/path/to/large-data-bundle
export SUBLIMINAL_REFERENCE_ASSETS_ROOT="$OPEN_RESEARCH_ASSETS_ROOT/station_subliminal"
export OPEN_RESEARCH_MODELS_ROOT=/path/to/local-model-cache
hf download unsloth/Qwen2.5-7B-Instruct \
  --local-dir "$OPEN_RESEARCH_MODELS_ROOT/Qwen2.5-7B-Instruct"
export LOCAL_BASE_MODEL_DIR="$OPEN_RESEARCH_MODELS_ROOT/Qwen2.5-7B-Instruct"
```

Use the compatible Unsloth distribution at
<https://huggingface.co/unsloth/Qwen2.5-7B-Instruct>. Its base model card is
<https://huggingface.co/Qwen/Qwen2.5-7B-Instruct>; the task's runtime validation
expects the Unsloth directory format.
The bundle contains reference numeric datasets and historical summaries, but no
base-model weights, LoRA adapter, or other reference checkpoint. Participants
train their own checkpoints and report their own seeds. The reference summaries
are context and calibration evidence, not pretrained solutions or a pass/fail
target.

`SUBLIMINAL_REFERENCE_ASSETS_ROOT` lets the helper read cloud-mounted
`reference_data/` and `reference_results/` without copying them into the public
source checkout.
