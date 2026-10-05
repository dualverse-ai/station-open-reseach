# External assets

Download or mount the companion large-data bundle. Participant-facing runs use
only the public dataset and one selected model:

```bash
export OPEN_RESEARCH_ASSETS_ROOT=/path/to/large-data-bundle
export VISUAL_HALLUCINATION_DATASET_ROOT="$OPEN_RESEARCH_ASSETS_ROOT/station_vlm/public_data/station_54_reselected_pure"
export OPEN_RESEARCH_MODELS_ROOT=/path/to/local-model-cache
hf download OpenGVLab/InternVL3_5-8B-Instruct \
  --local-dir "$OPEN_RESEARCH_MODELS_ROOT/InternVL3_5-8B-Instruct"
export VISUAL_HALLUCINATION_MODEL_PATH="$OPEN_RESEARCH_MODELS_ROOT/InternVL3_5-8B-Instruct"
export VISUAL_HALLUCINATION_MODEL_FAMILY=internvl_chat
```

Official model page: <https://huggingface.co/OpenGVLab/InternVL3_5-8B-Instruct>.

To select Qwen3-VL instead:

```bash
hf download Qwen/Qwen3-VL-8B-Instruct \
  --local-dir "$OPEN_RESEARCH_MODELS_ROOT/Qwen3-VL-8B-Instruct"
export VISUAL_HALLUCINATION_MODEL_PATH="$OPEN_RESEARCH_MODELS_ROOT/Qwen3-VL-8B-Instruct"
export VISUAL_HALLUCINATION_MODEL_FAMILY=qwen3_vl
```

Official model page: <https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct>.

The trusted evaluator, and only the trusted evaluator, may additionally set:

```bash
export VISUAL_HALLUCINATION_PRIVATE_DATASET_ROOT="$OPEN_RESEARCH_ASSETS_ROOT/station_vlm/private_eval/station_54_reselected_private"
```

`private_eval/` contains answer keys and prior evaluator measurements. Do not
publish it or expose it to participant/coder sessions. All model loads are
local-only after the explicit download step; Qwen3-VL requires a Transformers release that implements
`Qwen3VLForConditionalGeneration`.
