# Open-ended research tasks

This directory contains five self-contained Research Center task packages.
They include task specifications, evaluators, baselines, and small helper or
reference programs. No model weights, checkpoints, private station state, or
large datasets are included.

The matching cloud-storage bundle uses one stable relative directory per task.
After downloading or mounting it, set `OPEN_RESEARCH_ASSETS_ROOT` to the bundle
root and follow the task's `ASSETS.md`. The instructions do not depend on paths
from the machine that assembled the release.

## `station_emergent_plan`

Research question: identify an emergent planning algorithm in a trained DRC
Sokoban agent using activation probes and causal interventions.

External inputs required:

- PyTorch episode files supplied by the task owner, configured with
  `DRC_TRAIN_DATA_PATH` and `DRC_TEST_DATA_PATH`.
- Optional trained DRC checkpoint for the intervention demo, configured with
  `DRC_CHECKPOINT_PATH` (default: `checkpoints/ckp_actor_realstep250m.tar`).
- The `thinker`, `gym_sokoban`, and their compatible environment dependencies
  for the optional intervention demo.

Portable bundle paths and environment variables: `station_emergent_plan/ASSETS.md`.

## `station_lowrank`

Research question: measure reproducible low-rank structure in causal-language-
model next-token logit tables.

External inputs required or optional:

- The official `allenai/OLMo-1B-hf` model downloaded to a local directory and
  selected with `LOWRANK_MODEL_PATH`, or another local Hugging Face-compatible
  causal language model and tokenizer. The task never downloads model files at
  runtime.
  Set
  `LOWRANK_MODEL_NAME` to a model already present in the local Transformers
  cache. The helper uses `local_files_only=True`, so it never downloads model
  files during a task run. The default fallback is `distilgpt2`, but its files
  must already be cached.
- The two official Wiki shards from `allenai/olmo-mix-1124`, downloaded at the
  pinned upstream revision documented in `station_lowrank/ASSETS.md`, and
  selected locally with `LOWRANK_WIKI_DATA_DIR`. They are not redistributed in
  the companion large-data bundle. If omitted, the helper uses its small
  built-in corpus.
- Python packages `torch`, `transformers`, and `numpy`.

Official download instructions and local environment variables:
`station_lowrank/ASSETS.md`.

## `station_rnn`

Research question: characterize learned solution families in small RNNs on
fixed k-delay sequence tasks.

External inputs required:

- The official fixed tensors and metadata under a directory configured by
  `RNN_FIXED_DATA_ROOT` (default: `data/rnn_datasets`). It must contain
  `task_metadata.json` and `k_<k>/train_X.pt`, `train_Y.pt` files.
- Python package `torch`.

Portable bundle paths and environment variables: `station_rnn/ASSETS.md`.

## `station_vlm`

Research question: distinguish content-verifiable from knowledge-dependent
visual hallucinations in frozen vision-language-model evaluations.

External inputs required:

- A local, read-only 54-question/17-image dataset selected with
  `VISUAL_HALLUCINATION_DATASET_ROOT`; the private answer key is not included.
- One locally downloaded `InternVL3.5-8B-Instruct` or `Qwen3-VL-8B-Instruct`
  checkpoint selected with `VISUAL_HALLUCINATION_MODEL_PATH`. Official links
  and download commands are in `station_vlm/ASSETS.md`.
- Python packages `torch`, `torchvision`, `Pillow`, and `transformers`.
  Transformers loads are local-only and never download models.

Portable public/private bundle paths: `station_vlm/ASSETS.md`. The private
evaluator subtree must never be exposed to participant or coder sessions.

## `station_subliminal`

Research question: test subliminal trait transmission through semantically
unrelated numeric data using matched Qwen2.5 controls.

External inputs required:

- A locally downloaded `unsloth/Qwen2.5-7B-Instruct` base model configured with
  `LOCAL_BASE_MODEL_DIR`; this package never downloads from Hugging Face at
  runtime.
- Reference numeric datasets and historical summaries are supplied in the
  companion large-data bundle. No reference LoRA adapter is distributed;
  participants train their own checkpoints. Migration logs are not bundled.
- The selected local training/evaluation stack, including `torch`,
  `transformers`, `vllm` or Unsloth, and `numpy`.
- Remote OpenAI access is disabled in the public helper layer.

Portable bundle paths: `station_subliminal/ASSETS.md`. Reference checkpoints
are intentionally not distributed; participants train their own adapters.

All five packages may use the Station's own coder/evaluator runtime storage
for generated artifacts. Generated checkpoints and reports belong under that
runtime storage and are intentionally absent from this public source tree.
