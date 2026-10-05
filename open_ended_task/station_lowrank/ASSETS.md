# External assets

Download the model and the two Wiki shards from their official Hugging Face
repositories, then point the task at the local copies:

```bash
export OPEN_RESEARCH_MODELS_ROOT=/path/to/local-model-cache
hf download allenai/OLMo-1B-hf \
  --local-dir "$OPEN_RESEARCH_MODELS_ROOT/OLMo-1B-hf"
export LOWRANK_MODEL_PATH="$OPEN_RESEARCH_MODELS_ROOT/OLMo-1B-hf"

export OPEN_RESEARCH_DATA_ROOT=/path/to/local-data-cache
hf download allenai/olmo-mix-1124 \
  --repo-type dataset \
  --revision 99ee6aaace88779d1ef099d36251b91101c1679b \
  --include "data/wiki/wiki-000*.json.gz" \
  --local-dir "$OPEN_RESEARCH_DATA_ROOT/olmo-mix-1124"
export LOWRANK_WIKI_DATA_DIR="$OPEN_RESEARCH_DATA_ROOT/olmo-mix-1124/data/wiki"
```

The official model page is
<https://huggingface.co/allenai/OLMo-1B-hf>. `LOWRANK_MODEL_PATH` points to the
user-downloaded local copy.

The official dataset page is
<https://huggingface.co/datasets/allenai/olmo-mix-1124>. The revision is pinned
so that these exact upstream files are used:

- <https://huggingface.co/datasets/allenai/olmo-mix-1124/resolve/99ee6aaace88779d1ef099d36251b91101c1679b/data/wiki/wiki-0000.json.gz?download=true>
- <https://huggingface.co/datasets/allenai/olmo-mix-1124/resolve/99ee6aaace88779d1ef099d36251b91101c1679b/data/wiki/wiki-0001.json.gz?download=true>

`LOWRANK_WIKI_DATA_DIR` contains the two compressed Wiki shards consumed by the
helper's `*.json.gz` loader. The public source repository and companion
large-data bundle do not redistribute these upstream files. Model and dataset
loading during a task run is local-only and performs no web download. If the
Wiki directory is omitted, the helper uses its small built-in synthetic corpus,
which is useful for smoke tests but is not equivalent to the intended real-Wiki
experiment.
