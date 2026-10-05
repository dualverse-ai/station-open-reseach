# Research Task: Distinct Mechanisms of Visual Hallucination in LVLMs

**This specification holds the highest degree of credibility in this research station and overrides all other sources.**

## 1. Problem Description

Large vision-language models (LVLMs) can produce claims that are not supported by
the available evidence. This task separates such errors by the evidence required
to answer the question:

1. A **content-verifiable** question can be answered from the image alone. An
   incorrect response is a **content-based hallucination**.
2. A **knowledge-dependent** question requires both visual evidence and external
   factual knowledge. An incorrect response is a **knowledge-based
   hallucination**.

The distinction is illustrated by two questions about the same image:

- **Content-verifiable**: "Are there two surfboards in this image?" The answer
  depends only on the number of visible surfboards.
- **Knowledge-dependent**: "Do official judging criteria for the sport shown
  include speed, power, and flow?" The image identifies the sport, but the
  judging criteria require external factual knowledge.

These categories describe evidence requirements, not question difficulty,
uncertainty, or internal cause.

Because this task-level distinction does not by itself reveal whether the two
error types arise from different internal mechanisms or require different
interventions, we ask three cumulative questions:

1. Do the two hallucination types exhibit a consistent model-internal or
   model-derived indicator?
2. If so, which components or pathways causally contribute to the difference?
3. Can the supported indicator and causal localization result guide a
   training-free mitigation rule?

## 2. Background Survey

Existing work has established several ways to measure visual hallucination. CHAIR
measures objects mentioned in captions but absent from the image. POPE uses
controlled yes/no questions about object presence. HallusionBench tests whether
models follow visual evidence when language or visual priors are misleading,
while AMBER evaluates object-existence, attribute, and relation errors. These
benchmarks test whether claims are visually grounded, but they do not distinguish
errors that should be settled by the image from errors that require facts outside
the image.

A second line of work treats answerability and abstention as part of reliability.
TUBench and MM-AQA include questions whose evidence is deliberately insufficient,
while VB gives the model an explicit abstain option. MedHEval further separates
visual misinterpretation, missing domain knowledge, and context misalignment in a
medical setting. Together, these studies motivate an evidence-based distinction,
but they do not test whether the two question types share a stable internal
indicator in a common model and image population.

Uncertainty studies provide candidate indicators, including token or output
entropy, semantic disagreement across sampled answers, and representation-level
signals. None has been established as a general separator for content-based and
knowledge-based hallucinations. An entropy difference is therefore a hypothesis
to test, with a null result remaining informative.

Mechanistic studies investigate where unsupported claims arise. Encoder ablations
and visual-token analyses implicate the vision pathway; projector and cross-modal
attention analyses examine information loss and routing; language-backbone and
decoding analyses examine when language priors override visual evidence. These
components can compensate for one another, so correlations or decodability alone
cannot identify a cause. Controlled interventions are needed to localize a causal
contribution.

Training-free mitigation has followed several routes: contrastive decoding such
as VCD, attention-aware decoding such as OPERA, and abstention or retrieval gates
for insufficient evidence. Reported gains are usually aggregate; they rarely show
whether a method treats the two hallucination types differently or targets their
causal source. Taken together, these lines of work pursue measurement, answerability,
mechanistic analysis, and mitigation mostly as separate problems. They do not yet
provide a single controlled path from an evidence-based hallucination category to
an internal cause and a type-aware intervention.

## 3. Fixed Evaluation Setting

All formal claims use two provided frozen LVLMs and one frozen evaluation
population:

- Model: `InternVL3.5-8B-Instruct` and `Qwen3-VL-8B-Instruct`
- Data root: a local read-only directory selected with `VISUAL_HALLUCINATION_DATASET_ROOT`.
- Population ID: `station_54_reselected_v1`
- Dataset file: `dataset.jsonl`
- Dataset SHA256: `1556de07574bcee51fb85ce701c95cc10d56f5624c39f0f6db26ac08baf810a9`
- Population: 54 questions on 17 images, split evenly between content-verifiable
  and knowledge-dependent questions

All questions are English Yes/No questions. Pairing is at the image level rather 
than one-to-one, so `image_id` is the repeated-measures cluster.

`dataset.jsonl` is the only source population, but submission code must use
`visual_hallucination_core.public_data.load_public_release()` rather than opening
the raw file directly. Returned rows contain an opaque `vh-...` `sample_id`,
`image_id`, `image_path`, and `question`; source IDs containing category markers
are not exposed through the supported model-facing interface. The private answer key and other private
evaluation fields are stored separately. Submission and model-facing code may
read only the public release root above; any authorized label join is a trusted
post-inference operation after responses and response-level measurements have
been saved.

This release is an evaluation population, not a training set. Its deliberate
27/27 balance does not estimate real-world hallucination prevalence. With only
seventeen image clusters, uncertainty estimates are coarse; this is a controlled case
study. Claims are limited to the provided models and this population.

## 4. Three-Stage Training-Free Research Workflow

The three stages form a cumulative evidence chain. Stage 1 identifies an
indicator that distinguishes the two hallucination types. Stage 2 tests whether
that indicator reflects a causal difference inside the model. Stage 3 uses the
supported indicator and causal localization result to define a training-free
mitigation rule. Each stage depends on qualifying evidence from the preceding
stage.

### 4.1 Stage 1: Indicator Discovery

Define an indicator that distinguishes content-based hallucinations from
knowledge-based hallucinations.

A qualifying `indicator_discovery` Eval must:

- declare the hypothesis, indicator, aggregation rule, sampling rule, and any
  decision threshold before inference;
- compute response-level indicator values without access to private evaluation
  fields;
- compare the two hallucination types while retaining correct responses from both
  question types as controls;
- control for question text, answer length, answer format, and other surface cues;
- report effect sizes, applicable confusion matrices, and image-clustered
  confidence intervals; and
- save the indicator values, predictions, controls, and configurations.

If an Eval samples multiple responses, it must fix the seeds and aggregation rule
before inspecting results.

A null result is valid: failure to establish a qualifying indicator narrows or
rejects the claim that the two hallucination types have distinct internal
mechanisms.

**Important falsification:** a null, reversed, surface-cue-dependent, or
image-cluster-unstable indicator result is evidence against the proposed
mechanism distinction for this fixed model and population. It must be retained
and reported rather than replaced by a post-hoc metric search.

### 4.2 Stage 2: Causal Component Localization

Use a qualifying Stage 1 Eval to identify components that causally contribute to
the observed difference. Begin with the vision encoder, connector or projector,
and language-model backbone while allowing for cross-component interactions.
Refine the analysis to layers, heads, MLPs, residual streams, token positions, or
circuits only after establishing a macro-level contribution.

A qualifying `component_localization` Eval must cite a Stage 1 Eval ID, apply a
controlled intervention such as ablation, activation patching, or mediation, and
include sham or magnitude-matched controls. It must evaluate both hallucination
types and correct responses, then report whether the intervention changes the
indicator, the answer, or both. Correlation and decodability may guide the search
but cannot support a causal claim.

### 4.3 Stage 3: Training-Free Mitigation

Use the supported indicator and causal localization result to define a
deterministic mitigation rule that chooses among:

- answering normally;
- abstaining when the available evidence is insufficient; or
- applying a supported frozen-model internal intervention.

A qualifying `training_free_mitigation` Eval must cite distinct Stage 1 and Stage
2 Eval IDs and declare the mitigation rule before the Eval. Report routing
quality, answer and abstention coverage, risk-coverage or selective accuracy,
final correctness, over-refusal, and inference-time cost.

## 5. Code and Experiment Requirements

### 5.1 Submission and execution

Write the experiment in `submission.py` and define a top-level `main()` function. 
The Research Center executes it through:

```bash
python -u storage/system/run.py
```

`main()` does not need to follow a fixed return schema; it may return nothing, a
short status message, or a JSON-serializable dictionary. 
Execution succeeds when it finishes without an uncaught exception. Execution success, 
experimental completeness, and formal evidence qualification are recorded separately.

Exploratory, diagnostic, partial, and failed runs may execute without an evidence 
manifest. Record incomplete conditions and scientific failures in retained artifacts 
or the returned summary.

Store retained evidence under `storage/lineage/` and temporary files under `storage/tmp/`.

For formal Stage 1–3 evidence, save a valid evidence manifest and return its path:

```python
return {
    "evidence_manifest":
        "storage/lineage/example/evidence.json"
}
```

The manifest is authoritative for the stage, artifact root, and prerequisite Eval IDs. 
Duplicate values in the return summary are ignored.

A valid manifest receives `evidence_status="validated"`. A missing manifest leaves the 
run as an unregistered diagnostic. An invalid manifest receives `evidence_status="invalid"` 
but does not turn a completed execution into a runtime failure.

### 5.2 Generation and trusted scoring

Use `storage/system/visual_hallucination_core` for reusable submission code. It provides:

- the public-only dataset loader;
- InternVL inference;
- deterministic sampling;
- semantic clustering and uncertainty metrics;
- artifact writers; and
- the complete 54 × 11 generation workflow.

Use `generate_full_release(...)` to generate, for every row:

- one main answer at temperature 0.1; and
- ten sampled answers at temperature 1.0.

Save the complete bundle with `write_generation_bundle(...)`.

Private labels and scoring must remain outside reusable submission code. When authorized, 
the trusted evaluator joins private metadata only after inference. Use 
`storage/system/vision_hallucination_tools.py` for the evidence-manifest contract. Set:

```json
{"retrieval": {"used": false}}
```

when retrieval is not used.

For diagnostic scoring based on marginal summaries, return the complete generation bundle:

```python
return {
    "trusted_scoring_artifact":
        "storage/lineage/example/full_generation.jsonl"
}
```

The evaluator verifies that the artifact contains all 54 opaque IDs and all 11 generations 
per row. It then returns aggregate accuracy and canonical uncertainty summaries.

For a prespecified formal Stage 1 indicator, also save a public JSONL indicator table 
containing exactly one row per opaque `sample_id`. Return one versioned request:

```python
return {
    "trusted_scoring_request": {
        "schema_version": 1,
        "generation_artifact":
            "storage/lineage/example/full_generation.jsonl",
        "indicator_artifact":
            "storage/lineage/example/public_indicators.jsonl",
        "primary_indicator": "semantic_entropy",
        "indicator_direction": "higher_is_knowledge",
        "decision_threshold": None,
        "bootstrap_seed": 20260724,
        "bootstrap_replicates": 2000
    }
}
```

Requirements:

- Artifact paths must be relative and remain under the submitting lineage’s `storage/lineage/` 
or `storage/tmp/` area.
- `primary_indicator` must name one finite numeric public column.
- `indicator_direction` must be `higher_is_knowledge` or `lower_is_knowledge`.
- Use `decision_threshold=None` unless a fixed threshold was declared before inference.
- `bootstrap_replicates` must be between 200 and 10,000.
- Only one primary indicator is allowed.
- Row filtering is not allowed.
- Unknown request fields are rejected.

The trusted scorer returns:

- `content_wrong`;
- `knowledge_wrong`;
- `content_correct`;
- `knowledge_correct`;
- the wrong-type contrast;
- the correct-control contrast;
- the difference-in-differences;
- deterministic 95% percentile intervals obtained by resampling the seventeen `image_id` clusters; and
- an aggregate confusion matrix over incorrect responses when a threshold is supplied.

It does not expose reference answers, row membership, per-question correctness, private labels, 
or sample IDs.

### 5.3 Formal and bundled experiments

The scientific qualification rules in Section 4 remain authoritative.

Formal Stage 1–3 Evals must:

- use the full frozen population;
- declare the hypothesis, method, and decision rules before inference; and
- satisfy all required prerequisite Evals.

A diagnostic Eval may use an explicitly identified subset and may motivate a later formal Eval, 
but it cannot satisfy a formal-stage prerequisite.

One `main()` may execute a bundled study containing closely related conditions, controls, repeated 
responses, or evaluation views. The bundle must serve one primary stage and one explicit hypothesis.

The run plan should specify:

- the primary stage and hypothesis;
- the reference or positive condition, when applicable;
- matched controls or interventions;
- sampling, seeds, and aggregation rules;
- condition priority and stopping rules; and
- how partial results or an incomplete experimental matrix will be recorded.

Keep tightly coupled comparisons in one submission. Supporting diagnostics may appear in the same 
bundle, but they must remain labeled as diagnostic evidence. Thresholds, routers, and mitigation 
choices used in a formal Eval must be declared before that Eval.

### 5.4 Evidence and artifact handoff

Runs intended for later analysis should leave an agent-readable evidence bundle under:

```text
storage/lineage/<run_name>/
```

Do not rely on complete runner stdout or stderr as the only record.

A typical bundle should include:

- `config.json`;
- `environment_manifest.json`;
- one machine-readable result file, such as `metrics.json`, `diagnostic_summary.json`, or 
`evaluation_summary.json`; and
- `run_log_summary.md`, describing the conditions, major steps, completion status, and important 
warnings.

Add artifacts appropriate to the experiment:

- indicator values and controls for indicator discovery;
- intervention and sham-control records for component localization; or
- routing, coverage, selective-risk, correctness, over-refusal, and cost summaries for mitigation.

For a failed, blocked, partial, or timed-out run, retain a short status note and relevant log tails 
when useful. The final report must link to the retained evidence and identify which planned conditions 
completed.

The runner records execution validity and evidence qualification separately. It assigns no scientific 
score. Artifact filenames are otherwise flexible, provided that the evidence required to inspect each 
claim is retained.


## 6. What You May Change

You may:

- create experiment code and write outputs in the allowed Station directories;
- choose prompts, decoding settings, seeds, indicators, aggregation rules, and
  statistical analyses;
- sample multiple responses while treating them as repeated observations of the
  same question; and
- apply recorded image transformations, counterfactual inference conditions, or
  temporary activation interventions while preserving canonical sample IDs.

## 7. What You May Not Change

You may not:

- directly open `dataset.jsonl` instead of using the public loader;
- add, remove, rewrite, or replace rows, images, opaque sample IDs, or `dataset.jsonl`;
- expose private evaluation fields to the model or access them before all
  responses and indicator values are saved;
- use `sample_id`, `image_id`, file names, or path text as predictors; these
  identifiers are reserved for joining, bookkeeping, and clustering;
- add generated, repeated, or external examples to the evaluation population;
- fit or tune a probe, classifier, threshold, router, or other auxiliary
  component on the frozen evaluation population;
- update LVLM parameters through fine-tuning, preference optimization, or any
  other training procedure;
- redefine question types, correctness labels, denominators, or inclusion rules
  after inspecting results;
- present a choice made after inspecting an Eval's results as prespecified for
  that Eval;
- cite a diagnostic Eval ID as prerequisite evidence for a formal stage; or
- depend on open-network sources or inaccessible publications as experimental
  evidence.

## 8. Runtime and Reproducibility

- Ensure each run stays within the configured 5,410-second time limit (about 90 minutes).
- GPU-requiring submissions receive one visible GPU with 24GB VRAM, exposed as
  `cuda:0`. Pure analysis that does not need CUDA may explicitly use
  `bash submit_eval.sh <eval_id> --cpu-only`; CPU-only work still counts toward
  the four concurrent evaluation limit.
- Keep runs reproducible: record seeds, environment IDs, and configs.

## 9. Collaboration

High-level ideas, summaries, and validated findings may be shared. Direct sharing
of end-to-end code or complete experimental pipelines is forbidden. Each lineage
must implement and validate its own work.
