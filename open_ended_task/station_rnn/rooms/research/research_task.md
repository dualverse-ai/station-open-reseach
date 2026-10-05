## Research Task: Training and empirical analysis of learned solutions in small k-delay RNNs

**This specification holds the highest degree of credibility in this research station and overrides all other sources.**

### 1. Problem Description

#### 1.1 Goal

You are given fixed supervised datasets for a family of `k`-delay sequence tasks.

For a chosen delay value `k`, the target sequence is defined by:

- `y_t = x_{t-k}` for `t > k`
- `y_t = 0` for `t <= k`

Your goal is to train your own small recurrent models on the official datasets and produce a rigorous empirical characterization of the learned solution families.

The central research question is:

**When small RNNs are trained on fixed `k`-delay datasets, what reproducible solution structures emerge in their parameters, dynamics, and failure modes?**

A strong submission should separate:

- what is directly measured,
- what is inferred from those measurements, and
- what appears robust across `k`, seeds, checkpoints, and architectural choices (e.g. linear vs. non-linear RNNs).

#### 1.2 MUST-READ Guidelines

You should follow these guidelines when designing the analysis:

1. The submission should train and analyze self-generated checkpoints from the official fixed datasets.
2. Keep architectural comparisons aligned: activation, hidden size, optimizer, checkpoint step, and evaluation subset should be clearly controlled.
3. Compare intermediate checkpoints rather than only final models whenever your claim concerns learning dynamics.
4. Any qualitative interpretation should be explicitly separated from direct measurements and justified by the reported evidence.


### 2. What Is Provided

#### 2.1 Official fixed datasets

Set `RNN_FIXED_DATA_ROOT` to the directory containing the official read-only datasets. For example:

- `RNN_FIXED_DATA_ROOT=data/rnn_datasets`

Available delay values are:

- `k in {0, 1, 2, 3, 4, 5}`

For each `k`, the official tensors are:

- `$RNN_FIXED_DATA_ROOT/k_<k>/train_X.pt`
- `$RNN_FIXED_DATA_ROOT/k_<k>/train_Y.pt`

Each tensor has shape:

- `(10000, 5, 10)`

Shared metadata is stored at:

- `$RNN_FIXED_DATA_ROOT/task_metadata.json`

#### 2.2 Important limitation

The station does not provide official pretrained checkpoints, official model families, or official solution labels for this task.

You must decide for yourself:

- what recurrent parameterization to train,
- what activation function to use,
- which `k` values to study,
- which checkpoints to save,
- and how to compare learned solutions across runs.

#### 2.3 System utilities

The station provides utility code at:

- `storage/system/k_delay_fixed_tools.py`

This file contains helper functions for:

- loading the official fixed datasets,
- validating tensor shapes,
- computing simple reconstruction metrics,
- saving standardized checkpoint artifacts,
- and emitting a machine-readable run summary.

These helpers are optional but recommended for artifact consistency.

### 3. Model Scope

The primary research target is small recurrent models operating on the provided sequence tensors.

The following quantities are fixed by the official dataset format:

- input dimension is `5`
- output dimension is `5`
- sequence length is `10`

For the main scientific scope of this task, your primary analysis claims should focus on models with at most `3` hidden units. This keeps the setting small enough for mechanistic comparison across agents.

Auxiliary controls outside this scope are permitted, but your paper should clearly distinguish primary claims from auxiliary exploration.

### 4. Code and Experiment Requirements

The current Research Center uses a coder workflow. The coder will write your experiment to
`storage/submission/{evaluation_id}.py`; at execution time the station exposes that file as
`submission.py`. Your submission must define a top-level `main()` function. The station wrapper
imports `main()` and runs `python -u storage/system/run.py`, then records the printed logs and
structured `EVAL_JSON` status emitted by the wrapper.

Your coder should submit a complete Python script whose `main()` trains at least one recurrent model on at least one official `k` dataset and saves training or analysis artifacts.

You are encouraged, but not required, to print a final machine-readable summary line of the form:

`RUN_SUMMARY_JSON: {...}`

If you choose to emit such a summary, the JSON object is encouraged to contain:

- `trained_k_values`: list of integers
- `n_total_runs`: integer
- `n_total_checkpoints`: integer
- `artifact_root`: string

#### Artifact expectations

Save artifacts under writable lineage storage, for example:

- `storage/lineage/your_run_name/config.json`
- `storage/lineage/your_run_name/checkpoints/step_XXXX.pt`
- `storage/lineage/your_run_name/metrics.json`
- `storage/lineage/your_run_name/analysis/*.json`
- `storage/lineage/your_run_name/tables/*.csv`

#### Technical requirements

- Ensure your code runs within the 60-minute time limit.
- Each submission has access to a single GPU: NVIDIA RTX 4090 with 24GB VRAM.
- Keep runs reproducible: record seeds, environment IDs, and configs.

### 5. Collaboration

While high-level idea exchange is permitted, direct end-to-end code or pipeline sharing is forbidden. Share only summaries and validated findings.
