# No-Training Research Specification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the visual-hallucination Station consistently describe a frozen evaluation population with no training set, no research-facing scoring narrative, and a concise research specification separated from the runtime submission contract.

**Architecture:** Keep scientific requirements in `rooms/research/research_task.md` and `codex.md`, keep the mandatory `submission.py/main()` runtime contract in `README.md` and evaluator code, and align supervisor prompts with the no-training methodology. Preserve the four-GPU runtime while stating that one GPU is the normal choice for the 8B model and up to four are available when justified.

**Tech Stack:** Markdown, YAML, shell checks, Python unittest bundle tests.

---

### Task 1: Rewrite the research specification around a frozen evaluation population

**Files:**
- Modify: `rooms/research/research_task.md`

- [ ] **Step 1: Integrate the dataset rationale with the fixed evaluation setting**

Replace the separate model/dataset subsections with a concise `Fixed Evaluation Setting` that connects the survey to the deliberate single-model, frozen-population case study. Retain the model path, dataset path, release ID, hash, population counts, read-only rule, and core/supplementary analysis distinction.

- [ ] **Step 2: Remove all auxiliary-training language**

State explicitly that the release provides no training set and may not be used to fit or tune probes, classifiers, thresholds, routers, or other auxiliary models. Replace fitted-probe and held-out requirements with predeclared, label-blind measurements and image-clustered analysis.

- [ ] **Step 3: Remove research-facing scoring and coder boilerplate**

Replace `response-level scores` with `response-level measurements`, remove AUROC/leaderboard prose, and replace the coder `submission.py/main()` block with a short runtime and reproducibility section. State that one 24GB GPU is normally sufficient and up to four visible GPUs may be used when scientifically justified.

### Task 2: Align the Codex and research prompts

**Files:**
- Modify: `codex.md`
- Modify: `random_prompts.yaml`
- Modify: `meta_prompts.yaml`
- Modify: `constant_config.yaml`

- [ ] **Step 1: Replace train/held-out and probe-fitting requirements**

Require predeclared, label-blind indicators, prohibit fitting auxiliary models on the frozen population, and retain image-level clustering, causal controls, prerequisite Eval IDs, and adaptive-exploration disclosure.

- [ ] **Step 2: Remove leaderboard/proxy-score narrative from researcher-facing prompts**

Describe progress in terms of evidence quality, reproducibility, and mechanistic claims. Preserve internal framework fields required by the evaluator and archive reviewer.

- [ ] **Step 3: Parse all modified YAML files**

Run:

```bash
python -c 'import pathlib, yaml; [yaml.safe_load(pathlib.Path(p).read_text()) for p in ["constant_config.yaml", "random_prompts.yaml", "meta_prompts.yaml"]]'
```

Expected: exit code 0 with no output.

### Task 3: Keep the runtime interface in README only

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Clarify the fixed resources and submission contract**

Keep the required `submission.py/main()` contract because the evaluator enforces it. Remove research-scoring prose, state that the research task governs scientific requirements, and describe the four visible GPUs with one-GPU-normal guidance.

### Task 4: Verify cross-file consistency

**Files:**
- Verify: `rooms/research/research_task.md`
- Verify: `codex.md`
- Verify: `README.md`
- Verify: `constant_config.yaml`
- Verify: `random_prompts.yaml`
- Verify: `meta_prompts.yaml`
- Test: `tests/`

- [ ] **Step 1: Scan researcher-facing documents for forbidden training/scoring language**

Run targeted `rg` checks to confirm that `fitted-probe`, `training split`, `held-out split`, `leaderboard`, `proxy score`, and coder boilerplate no longer appear in the research specification or Codex.

- [ ] **Step 2: Confirm the no-training and GPU rules are present**

Check that the task explicitly prohibits fitting auxiliary models on the frozen population and states the one-GPU-normal, four-GPU-available policy.

- [ ] **Step 3: Run the full bundle test suite**

Run:

```bash
python -m unittest discover -s tests -v
```

Expected: all tests pass.
