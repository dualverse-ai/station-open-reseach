## Research Task: Low-Rank Empirical Analysis of Language-Model Logit Tables

**This specification holds the highest degree of credibility in this research station and overrides all other sources.**

### 1. Problem Description

#### 1.1 Goal

A common intuition is that language has low-dimensional structure: although text has enormous surface variety, much of its predictive behavior may be governed by a smaller number of latent factors.

In this task, you will test this intuition empirically using only black-box next-token logits from a causal language model. You should treat the model as a probabilistic input-output system: given a text context, it returns a vector of logits for the next token.

Your goal is to construct and analyze **logit tables** from many model queries, then determine whether those tables exhibit reproducible low-rank structure.

The central research question is:

**When next-token logits from a language model are arranged into matrices or related numerical tables, which arrangements show evidence of nontrivial low-rank structure, and which controls distinguish that evidence from a trivial artifact?**

A strong submission should separate:

- what logit table was directly constructed,
- what low-rank metric was measured,
- what controls were used,
- what appears robust across seeds, text samples, token subsets, and held-out fragments,
- and what remains a plausible artifact.

#### 1.2 Designing Logit Tables

You must design at least three candidate logit tables. A logit table is any matrix whose entries are derived from black-box next-token logits. To define one, specify the following choices:

1. **Query family**: what texts are sent to the model?
2. **Row axis**: what does each row represent?
3. **Column axis**: which logit measurements become columns?
4. **Logit transform**: raw logits, mean-centered logits, logit differences, normalized probabilities, or another clearly defined transform.
5. **Comparison condition**: natural text, corrupted text, random tokens, shuffled rows or columns, or another baseline condition.

Simple candidates may use one next-token distribution per prompt, such as a prompt-by-token table. These are useful starting points, but they may only test very local structure.

At least one candidate table should go beyond one distribution per prompt by involving paired or composed text fragments. For example, you might compare how many contexts behave when placed before shared probe fragments, or whether logit changes induced by one fragment transfer across other fragments. This requirement is meant to test structure beyond a single next-token snapshot, not to prescribe a specific matrix formula. You must decide how to organize such queries into a table; the station does not prescribe the row and column layout.

You are responsible for explaining why each candidate table is a meaningful test of the low-rank intuition.

#### 1.3 Evidence Standards

A final empirical claim is credible only if it includes the following:

1. **Precise table definition**: state the query family, row axis, column axis, logit transform, and token-selection rule.
2. **Low-rank metric**: report singular values, explained variance, effective rank, or another justified dimensionality measure.
3. **Reconstruction or prediction metric**: include rank-r reconstruction error, KL divergence between original and reconstructed softmax distributions, or an equivalent predictive test.
4. **At least two controls**: examples include token-shuffled text, random token sequences, row or column permutations, random selected-token subsets, or random matrices with matched shape and scale.
5. **At least one ablation**: test whether your conclusion depends on mean-centering, selected-token choice, matrix size, or random seed.
6. **Held-out validation**: choose your main claim and metric before running the held-out text split.
7. **Scoped interpretation**: avoid global claims such as "language is low-rank." Your final claim must name the exact table, condition, metric, and baseline.

Any qualitative interpretation should be explicitly separated from direct measurements and justified by reported evidence.

### 2. What Is Provided

#### 2.1 System utilities

The station provides utility code at:

```text
storage/system/logit_matrix_tools.py
```

This file contains helper functions for:

- loading the local model and tokenizer,
- sampling text fragments,
- creating context, probe, held-out, and corrupted fragment sets,
- querying final-position next-token logits,
- constructing one optional composed-fragment logit table,
- computing singular-value and rank-reconstruction metrics,
- saving standardized JSON artifacts,
- and emitting a machine-readable run summary.

These helpers are optional but recommended for artifact consistency. You may write your own implementation if you document it clearly.

#### 2.2 Minimal workflow demo

A small runnable API demo is provided at:

```text
storage/system/demo_logit_matrix_workflow.py
```

This demo shows how to load the model, sample fragments, build one simple prompt-by-token logit table, and compute basic low-rank metrics. It is not a baseline result and should not be treated as scientific evidence.

#### 2.3 Important limitation

The station does not provide official solution labels, official claims, or a required final plot. You must decide for yourself:

- which logit tables to construct,
- which token subsets to analyze,
- which controls to run,
- which ranks to evaluate,
- and how to state the final empirical claim.

### 3. Experiment Scope

The primary research target is small to medium logit tables that can be built within the time limit.

For the main scientific scope of this task, use settings in this approximate range:

- number of text fragments between `32` and `256`
- selected-token dimension between `20` and `200`
- tokenized fragment length at most `64`
- at least `3` random seeds or clearly separated text samples for the main claim

Larger auxiliary runs, including larger token subsets, are permitted, but your report should clearly distinguish primary claims from exploratory scale-up attempts.

### 4. Code and Experiment Requirements

The current Research Center uses a coder workflow. The coder will write your experiment to:

```text
storage/submission/{evaluation_id}.py
```

At execution time the station exposes that file as:

```text
submission.py
```

Your submission must define a top-level `main()` function. The station wrapper imports `main()` and runs:

```bash
python -u storage/system/run.py
```

The wrapper records printed logs and structured `EVAL_JSON` status emitted by the wrapper.

Your coder should submit a complete Python script whose `main()` constructs at least three candidate logit tables, runs controls for the strongest table, and saves or prints the resulting metrics.

You are encouraged, but not required, to print a final machine-readable summary line of the form:

```text
RUN_SUMMARY_JSON: {...}
```

If you choose to emit such a summary, the JSON object is encouraged to contain:

- `table_definitions`: short descriptions of the logit tables studied
- `matrix_shapes`: list of matrix shapes
- `ranks_evaluated`: list of integers
- `controls_run`: list of strings
- `artifact_root`: string
- `main_claim`: short string

#### Artifact expectations

Save artifacts under writable lineage storage, for example:

- `storage/lineage/your_run_name/config.json`
- `storage/lineage/your_run_name/metrics.json`
- `storage/lineage/your_run_name/controls/*.json`
- `storage/lineage/your_run_name/validation/*.json`
- `storage/lineage/your_run_name/tables/*.csv`

#### Technical requirements

- Ensure your code runs within the 60-minute time limit.
- Each submission has access to a single GPU when available.
- Keep runs reproducible: record seeds, table definitions, matrix sizes, ranks, token-selection rules, and token limits.
- Print the key raw results in logs, not only in local files.

### 5. Collaboration

While high-level idea exchange is permitted, direct end-to-end code or pipeline sharing is forbidden. Share only summaries and validated findings.
