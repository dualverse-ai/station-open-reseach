# External assets

Download or mount the companion large-data bundle, then set:

```bash
export OPEN_RESEARCH_ASSETS_ROOT=/path/to/large-data-bundle
export RNN_FIXED_DATA_ROOT="$OPEN_RESEARCH_ASSETS_ROOT/station_rnn/data/rnn_datasets"
```

The directory contains `k_0` through `k_5`; each subdirectory has
`train_X.pt` and `train_Y.pt`. Every tensor has shape `(10000, 5, 10)` and
dtype `float32`. `task_metadata.json` is a portable manifest describing the
verified tensor contract and the target rule. No pretrained RNN checkpoints
are supplied: participants train and analyze their own recurrent models.
