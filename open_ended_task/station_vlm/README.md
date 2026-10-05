# Visual Hallucination Station

This directory is a Station task bundle for exploratory research on visual
hallucination in large vision-language models. The governing research source is
`Proposal_vision_language.pdf`; the operational agent specification is
`rooms/research/research_task.md`.

The task tests the hypothesis that content-based and knowledge-based
hallucinations arise from distinguishable internal mechanisms. Research proceeds
through three evidence-gated stages: indicator discovery, component localization,
and training-free mitigation through type-aware routing.

A null result at the indicator stage is still an important falsification of the
claim that the two hallucination types share the same internal mechanism.

## Fixed Resources

- Models: `InternVL3.5-8B-Instruct` and `Qwen3-VL-8B-Instruct`, supplied through a local model cache and selected with `VISUAL_HALLUCINATION_MODEL_PATH`.
- Data root: a local, read-only dataset selected with `VISUAL_HALLUCINATION_DATASET_ROOT`.
- Population: `dataset.jsonl`, 54 questions on 17 images
- Population ID: `station_54_reselected_v1`
- Dataset SHA256: `1556de07574bcee51fb85ce701c95cc10d56f5624c39f0f6db26ac08baf810a9`
- Runtime: four visible 24GB RTX 3090 GPUs, one active evaluation, 90-minute
  timeout

The data root remains read-only. Only `dataset.jsonl` defines the model-facing
population. It contains 54 English Yes/No questions on seventeen images,
balanced as 27 content-verifiable and 27 knowledge-dependent questions. The
private answer key and evaluation metadata remain separate and must not be used
until responses and response-level measurements have been saved.

One GPU is the expected default for either 8B checkpoint. Set
`VISUAL_HALLUCINATION_MODEL_PATH` to select the checkpoint. A submission may use
up to all four visible GPUs when its experimental design justifies the
additional communication and coordination cost.

## Submission Contract

Submissions define `main()` in `submission.py`. The system runner records
execution validity and evidence paths. The research task governs scientific
requirements; this interface exists only to execute and register an experiment.

## Local Verification

```bash
python -m unittest discover -s tests -v
```

The Station engine itself is supplied by the deployment environment and is not
vendored in this task bundle.
