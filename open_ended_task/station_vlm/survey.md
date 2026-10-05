# Survey: Visual Hallucination in Large Vision-Language Models

**Date**: 2026-07-06  
**Project source document**: `Proposal_vision_language.pdf`  
**Scope**: LVLM/MLLM visual hallucination, with emphasis on content-based vs knowledge-based hallucination and component-level causes.

## 1. Research Problem

This project asks a sharper question than "do LVLMs hallucinate?" It treats hallucination as a phenomenon that should be decomposed along two axes:

1. **Hallucination type**
   - **Content-based hallucination**: the image is sufficient, but the model makes a false visual claim about objects, attributes, counts, OCR, scenes, or relations.
   - **Knowledge-based hallucination**: the image is insufficient, but the model still gives a confident answer requiring metadata, world knowledge, medical/domain knowledge, identity, geography, time, or external evidence.
2. **Architectural cause**
   - **Vision encoder**: what visual evidence is encoded or lost.
   - **Mapping/projector module**: how visual features are routed into the language space.
   - **Language model and decoding/alignment**: whether the model follows visual evidence, overuses language priors, abstains when evidence is insufficient, or retrieves/verifies.

The relevant literature spans benchmark design, visual evidence modeling, answerability, domain-specific error taxonomies, attention-causal mitigation, evidence-aware preference optimization, and reliability-aware retrieval.

## 2. Executive Synthesis

The literature suggests a movement from documenting object hallucination toward studying **why, when, and through which internal route hallucination occurs**. Canonical benchmarks and metrics such as CHAIR, POPE, MMHal-Bench, HallusionBench, MME, and AMBER provide baseline measurements, while recent preprints add finer-grained causal, answerability, and mitigation perspectives. The 2025-2026 papers should be treated as an emerging research trend and provisional evidence.

First, **vision-side causality is a plausible and increasingly studied mechanism**. Wang et al.'s 2025 paper, *Diving into Mitigating Hallucinations from a Vision Perspective for LVLMs*, introduces VHBench-10, a benchmark with roughly 10,000 samples across ten fine-grained hallucination categories, and argues that different visual encoders carry different inductive biases that lead to different hallucination profiles. This provides motivation, not definitive proof, for component-level experiments that vary CLIP, DINO, DINOv2, and other visual encoders.

Second, **hallucination is increasingly separated by cause rather than only by surface error**. MedHEval (2025) divides medical LVLM hallucinations into visual misinterpretation, knowledge deficiency, and context misalignment. This taxonomy aligns with the content-based vs knowledge-based distinction: visual misinterpretation is close to content-based hallucination, while knowledge deficiency and context misalignment expose knowledge-based and context-dependent hallucination.

Third, **answerability and abstention are becoming central**. TUBench (2024), MM-AQA (2026), and VB (2026) test whether VLMs know when the visual/textual evidence is insufficient. This literature cluster is especially relevant to knowledge-based hallucination. It suggests that knowledge-based hallucination should be measured not by ordinary accuracy alone, but by calibrated abstention, selective prediction, evidence sufficiency, and refusal/retrieval behavior.

Fourth, **recent mitigation preprints increasingly target internal dynamics**. AIR (2026), CAI (2026), Fox (2026), OPPO (2026), and ADAPT (2026) move beyond generic decoding tricks by focusing on attention imbalance, context-aware visual attention, risky mediator heads, ordered visual evidence preference, and cross-attention drift. These papers are best read as emerging mechanistic hypotheses and intervention designs that require further independent replication.

The central conclusion is:

> Current LVLM hallucination research provides several partial decompositions of failure by visual evidence, answerability, attention dynamics, and domain-specific knowledge deficiency. A unified 2 x 3 causal study that crosses hallucination type with LVLM component remains a well-motivated open direction.

## 3. Selection Method and Evidence Status

This is a targeted narrative survey rather than a PRISMA systematic review. Sources were selected to cover four evidence strata:

1. **Foundational metrics and benchmarks**: CHAIR, POPE, MMHal-Bench/LLaVA-RLHF, HallusionBench, MME, and AMBER establish the object hallucination, multimodal hallucination, and evaluation baseline.
2. **Architecture and component sources**: LLaVA, BLIP-2, CLIP, DINO, and DINOv2 define the model components that can be varied in experiments.
3. **Mitigation baselines**: VCD and OPERA are included because they are common training-free decoding baselines for object hallucination mitigation.
4. **Recent research frontier**: 2025-2026 papers were included when they directly inform answerability, abstention, vision-side causes, attention dynamics, causal decoding, evidence-aware preference optimization, or reliability-aware retrieval.

The source matrix uses four evidence-status labels:

- **Foundational**: widely used metric, benchmark, architecture, or baseline method.
- **Peer-reviewed/accepted**: source has a peer-reviewed or accepted conference/journal version, where known.
- **Recent preprint**: arXiv preprint used for emerging trends; claims should be treated as provisional.
- **Speculative connection**: source is conceptually useful for the project but does not directly evaluate the same LVLM hallucination construct.

## 4. Taxonomy

### Content-based hallucination

Content-based hallucination is defined as an error on visually answerable input. The category includes more than object presence:

- **Object hallucination**: absent objects are mentioned or confirmed.
- **Attribute hallucination**: color, material, size, state, or action is wrong.
- **Relation hallucination**: spatial or semantic relations are wrong.
- **Scene/instance hallucination**: global scene or instance identity is misread.
- **Visibility hallucination**: the model claims something is visible when it is not, or fails to abstain when a human viewer cannot decide.

Representative sources:

- VHBench-10 in *Diving into Mitigating Hallucinations from a Vision Perspective for LVLMs* (2025) expands content hallucination into ten fine-grained categories.
- PRE-HAL (2025) separates perception and reasoning hallucination across instances, scenes, and relations.
- AIR, Object-Aligned VCD, CAI, Fox, OPPO, and ADAPT (2026) all mainly target object/content hallucination through attention, contrastive views, causal interventions, or preference alignment.

### Knowledge-based hallucination

Knowledge-based hallucination occurs when the image does not contain enough evidence, yet the model answers as if it does. This category is operationalized through answerability, abstention, reliability, and domain-knowledge evaluation:

- **Unanswerability**: the question cannot be answered from the given multimodal evidence.
- **Evidence insufficiency**: answerable and unanswerable cases differ by controlled transformations.
- **Domain knowledge deficiency**: the model lacks specialized knowledge, as in medical LVLMs.
- **Context misalignment**: the model overuses or misuses provided context.
- **Reliability failure**: the model does not know when to answer, abstain, retrieve, or warn.

Representative sources:

- MM-AQA (2026) constructs unanswerable instances from answerable ones along visual modality dependency and evidence sufficiency axes.
- VB (2026) tests visibility and perspective reasoning with an explicit ABSTAIN option.
- TUBench (2024) evaluates LVLM trustworthiness on unanswerable questions from multiple visual domains.
- MedHEval (2025) provides a cause taxonomy including knowledge deficiency and context misalignment.
- Retrieval-augmented reliability-aware inference (2026) connects external evidence, uncertainty, and abstention/fallback decisions.

## 5. Literature Clusters

### Benchmarks and evaluation: object presence, cause labels, and answerability

POPE evaluates object hallucination with absent-object yes/no questions. HallusionBench stresses image-context reasoning and misleading priors. MME and AMBER broaden evaluation dimensions. These benchmarks establish baseline measurements for object, perception, cognition, and multi-dimensional hallucination.

Recent cause-aware and answerability-focused benchmarks extend this baseline:

- **VHBench-10 (2025)**: fine-grained visual hallucination categories, approximately 10,000 samples, and analysis of visual encoder inductive biases.
- **PRE-HAL (2025)**: perception-reasoning evaluation hallucination, emphasizing that relation reasoning reveals vulnerabilities missed by perception-only benchmarks.
- **MedHEval (2025)**: hallucination categories by underlying cause: visual misinterpretation, knowledge deficiency, and context misalignment.
- **TUBench (2024)**: unanswerable visual questions across code screenshots, natural images, geometry diagrams, and statistical tables.
- **MM-AQA (2026)**: multimodal abstention benchmark built by transforming answerable samples into unanswerable ones.
- **VB (2026)**: visibility and perspective reasoning benchmark with VISIBLY_TRUE, VISIBLY_FALSE, and ABSTAIN labels.

The practical implication is to avoid an object-only benchmark. A more informative dataset should include **answerability labels** and **cause labels**.

### Vision encoder and visual evidence

Vision-perspective evidence is central to the first component hypothesis. It argues that visual encoders are not interchangeable: different training paradigms encode different inductive biases, which produce different hallucination behavior. This motivates an experiment where CLIP, DINO, DINOv2, SigLIP/EVA-style encoders, or other available encoders are swapped under controlled downstream conditions.

The expected pattern is:

- A better or more suitable visual encoder should reduce content-based hallucination.
- It should not automatically solve knowledge-based hallucination, because missing external knowledge is not a visual encoding problem.
- If a vision encoder swap also changes abstention behavior, the model may be using visual confidence as a proxy for answerability.

### Projector and cross-modal routing

The projector remains under-studied relative to the vision encoder and the LLM. BLIP-2, LLaVA, and MiniGPT-4 show why this module matters, but fewer hallucination papers isolate it directly.

Recent attention-based mitigation papers indirectly implicate cross-modal routing:

- ADAPT (2026) identifies progressive degradation of text-to-image cross-attention during generation.
- CAI (2026) uses context-aware attention intervention to decide where to look and when to intervene.
- AIR (2026) diagnoses modality-wise and token-wise attention imbalance.
- Fox (2026) describes risky attention mediators that decouple from visual evidence and lock onto language priors.

These works suggest that projector/routing failure may not appear only at the static embedding interface. It can emerge dynamically as visual attention degrades during generation.

### LLM, language priors, abstention, and alignment

MM-AQA finds that standard prompting rarely induces abstention, and that multimodal systems often attempt reconciliation when evidence is degraded or contradictory. VB similarly treats abstention as part of the task rather than a failure to answer. TUBench shows that answerable-only VQA benchmarks miss a large reliability dimension.

Alignment and preference-optimization papers target visual grounding directly:

- OPPO (2026) frames hallucination as insufficient calibration of generation preferences to visual evidence and trains ordered visual-evidence preferences.
- ADAPT (2026) combines attention-supervised inference and Visual Attention Guidance DPO.
- Honesty and KTO-style alignment work provides general conceptual support for refusal calibration, but LVLM-specific evaluation should rely on multimodal answerability and evidence-grounding tasks.

LLM interventions should not be evaluated only on answer accuracy. They should be evaluated on **knowledge overclaim rate**, **calibrated abstention**, and **retrieval-trigger quality**.

### Training-free decoding and attention interventions

The recent mitigation landscape is rich enough to support a separate ablation suite:

- **AIR (2026)**: attention imbalance rectification, a decoding-time method targeting modality and token attention imbalance.
- **Object-Aligned VCD (2026)**: improves VCD by constructing an object-aligned auxiliary view using object-centric attention.
- **CAI (2026)**: context-aware attention intervention that strengthens visual grounding selectively, not uniformly.
- **Fox (2026)**: a causal decoding framework that identifies risky mediator heads and intervenes on shortcut paths.
- **Retrieval-Augmented Reliability-Aware Inference (2026)**: combines retrieval evidence, reliability indicators, and selective decision gates.

These methods are useful because they can often be applied without full retraining, making them clean probes for whether hallucination is caused by decoding dynamics or deeper representation failure.

## 6. Experimental Design

### Dataset design

Build a dataset with two orthogonal labels:

| Label axis | Values |
|---|---|
| Hallucination type | content-based, knowledge-based, mixed |
| Failure cause | visual misinterpretation, knowledge deficiency, context misalignment, attention/routing drift, language-prior shortcut |

Borrow labeling ideas from VHBench-10, MedHEval, PRE-HAL, TUBench, MM-AQA, and VB.

Each sample should include:

- image;
- question or instruction;
- answerability label;
- evidence sufficiency rationale;
- expected behavior: answer, abstain, retrieve, or answer-with-caveat;
- fine-grained visual category: object, attribute, relation, scene, count, OCR, visibility, perspective;
- domain category: natural image, chart/table, code screenshot, medical image, diagram, web/screenshot, generated image.

### Metrics

Use different metrics for the two hallucination types:

**Content-based metrics**

- object hallucination rate;
- attribute/relation/count hallucination rate;
- contrastive image-edit sensitivity;
- attention-to-evidence score;
- visual encoder sensitivity score.

**Knowledge-based metrics**

- knowledge overclaim rate;
- calibrated abstention F1;
- selective prediction AUC;
- retrieval-trigger precision/recall;
- evidence insufficiency detection accuracy;
- answer-with-caveat quality.

**Cross-cutting metrics**

- entropy over repeated samples;
- paraphrase stability;
- confidence calibration;
- faithfulness-fluency trade-off;
- component sensitivity score.

### Component experiments

**Experiment A: vision encoder swap**

Compare CLIP, DINO/DINOv2, and stronger modern encoders where available. Inspired by VHBench-10, report hallucination by fine-grained category rather than only aggregate object hallucination.

**Experiment B: projector/routing intervention**

Hold encoder and LLM fixed while varying projector architecture or fine-tuning. Add diagnostics from AIR/ADAPT/CAI/Fox: cross-modal attention balance, visual attention entropy, attention drift, and risky mediator heads.

**Experiment C: LLM answerability alignment**

Train or preference-optimize on answerability labels. Include abstention-aware objectives inspired by MM-AQA/VB/TUBench and evidence-aware preference objectives inspired by OPPO/ADAPT.

**Experiment D: decoding-time mitigation**

Apply AIR, Object-Aligned VCD, CAI, Fox-style interventions, and retrieval/reliability-aware gating to the same base model. Compare their impact separately on content-based and knowledge-based hallucination.

**Experiment E: retrieval and reliability gating**

For knowledge-based items, test whether external visual/textual retrieval plus reliability estimation reduces overconfident wrong answers without over-abstaining on answerable items.

## 7. Knowledge Gaps

1. **The projector remains a weakly isolated cause**. Recent attention papers implicate routing, but direct projector ablations are still relatively rare.
2. **Benchmarks are converging but not unified**. VHBench-10, PRE-HAL, MedHEval, MM-AQA, VB, and TUBench each cover part of the proposed taxonomy; none alone gives the full 2 x 3 matrix.
3. **Knowledge-based hallucination is measurable but still under-integrated with LVLM architecture studies**. Abstention benchmarks rarely test encoder/projector swaps.
4. **Mitigation papers report strong aggregate gains but not always type-specific gains**. A method that reduces object hallucination may not improve knowledge deficiency or unanswerable cases.
5. **Attention interventions need causal validation**. AIR, CAI, ADAPT, and Fox provide mechanisms, but more work is needed to distinguish correlation from true causal pathways.
6. **Domain-specific hallucination is important**. MedHEval shows that medical hallucination has cause categories that general object hallucination benchmarks miss.

## 8. Recommended Positioning for the Paper

The survey supports the following positioning:

> LVLM hallucination research has moved from object-level detection toward cause-aware benchmarks, abstention, attention dynamics, and evidence-aware mitigation. However, the field still lacks a unified causal study that crosses hallucination type with LVLM component. This study fills that gap by separating content-based from knowledge-based hallucination and testing how the vision encoder, projector, and language model contribute to each.

This gives four concrete contributions:

1. A formal definition of content-based vs knowledge-based visual hallucination.
2. A relabeled or newly constructed benchmark with answerability and cause labels.
3. Component-level causal experiments over vision encoder, projector, and LLM/alignment.
4. Type-specific evaluation of recent mitigation methods, including attention intervention, contrastive decoding, abstention training, and reliability-aware retrieval.

## 9. Annotated Bibliography

### 2026: abstention, attention, causal decoding, and reliability

1. Madhusudhan, N., Yadav, V., & Lacoste, A. (2026). [Knowing When Not to Answer: Evaluating Abstention in Multimodal Reasoning Systems](https://arxiv.org/abs/2604.14799).  
   Introduces MM-AQA, a benchmark for multimodal abstention based on evidence sufficiency and visual modality dependency. Central for knowledge-based hallucination.

2. Tripathi, N. (2026). [VB: Visibility Benchmark for Visibility and Perspective Reasoning in Images](https://arxiv.org/abs/2603.06680).  
   Tests whether models judge what is visible and abstain when humans cannot reliably answer. Useful for visibility-based content/knowledge boundary cases.

3. Sun, H., Li, Q., Wang, P., & Zhang, M. (2026). [Mitigating Object Hallucinations in LVLMs via Attention Imbalance Rectification](https://arxiv.org/abs/2603.24058).  
   Proposes AIR, linking object hallucination to modality-wise and token-wise attention imbalance.

4. Chen, B., Liu, X., & Qiu, J. (2026). [Mask What Matters: Mitigating Object Hallucinations in MLLMs with Object-Aligned Visual Contrastive Decoding](https://arxiv.org/abs/2602.11737).  
   Improves VCD through object-aligned auxiliary views; useful for content-based object hallucination ablations.

5. Lei, Y., Lyu, W., Du, Y., Zhen, X., Snoek, C. G. M., & Shao, L. (2026). [See Only When Needed: Context-Aware Attention Intervention for Mitigating Hallucinations in LVLMs](https://arxiv.org/abs/2606.29847).  
   Proposes CAI, a selective attention intervention that decides where to look and when to intervene.

6. Yu, L., Chen, C., Kuang, P., Feng, Z., Zhou, F., & Dobbie, G. (2026). [Dismantling Pathological Shortcuts: A Causal Framework for Faithful LVLM Decoding](https://arxiv.org/abs/2606.27596).  
   Proposes Fox, a causal decoding framework targeting attention heads that decouple from visual evidence.

7. Hariharan, P., Xu, H., & Yan, D. (2026). [Mitigating Visual Hallucinations in Multimodal Systems through Retrieval-Augmented Reliability-Aware Inference](https://arxiv.org/abs/2606.15782).  
   Uses retrieved evidence and reliability indicators to decide accept, caution, or abstain/fallback.

8. Zou, X., et al. (2026). [Clearer Sight, Fewer Lies: Oriented Pickup Preference Optimization for Multimodal Hallucination Mitigation](https://arxiv.org/abs/2606.29805).  
   Proposes OPPO, an evidence-aware preference objective over visual evidence strength.

9. Yao, Z., et al. (2026). [ADAPT: Attention Dynamics Alignment with Preference Tuning for Faithful MLLMs](https://arxiv.org/abs/2606.31054).  
   Links hallucination to degradation of text-to-image cross-attention and combines attention-supervised inference with Visual Attention Guidance DPO.

### 2025: vision-side, cause-aware, and domain-specific hallucination

10. Wang, W., et al. (2025). [Diving into Mitigating Hallucinations from a Vision Perspective for Large Vision-Language Models](https://arxiv.org/abs/2509.13836).  
    Introduces VHBench-10 and VisionWeaver; directly relevant to the vision encoder hypothesis.

11. Chang, A., Huang, L., Bhatia, P., Kass-Hout, T., Ma, F., & Xiao, C. (2025). [MedHEval: Benchmarking Hallucinations and Mitigation Strategies in Medical Large Vision-Language Models](https://arxiv.org/abs/2503.02157).  
    Categorizes medical LVLM hallucinations into visual misinterpretation, knowledge deficiency, and context misalignment.

12. Huang, T., Liu, Z., Wang, R., Zhang, Y., & Jing, L. (2025). [Visual Hallucination Detection in Large Vision-Language Models via Evidential Conflict](https://arxiv.org/abs/2506.19513).  
    Introduces PRE-HAL, separating perception and reasoning hallucination and using evidential conflict for detection.

13. [A Comprehensive Analysis for Visual Object Hallucination in Large Vision-Language Models](https://arxiv.org/abs/2505.01958). (2025).  
    Component-oriented object hallucination analysis; useful as a recent comparison point for causal attribution.

### 2024: answerability and survey context

14. He, X., Zhang, Q., Jin, A.-L., Yuan, Y., & Yiu, S.-M. (2024). [TUBench: Benchmarking Large Vision-Language Models on Trustworthiness with Unanswerable Questions](https://arxiv.org/abs/2410.04107).  
    Evaluates LVLMs on unanswerable questions across code, natural images, geometry, and tables.

15. Zhao, Q., et al. (2024). [The First to Know: How Token Distributions Reveal Hidden Knowledge in Large Vision-Language Models?](https://arxiv.org/abs/2403.09037).  
    Uses first-token/logit distributions to detect whether models should respond, including unanswerable visual questions.

16. Bai, Z., et al. (2024). [Hallucination of Multimodal Large Language Models: A Survey](https://arxiv.org/abs/2404.18930).  
    Broad survey of multimodal hallucination; useful for terminology, taxonomy, and coverage context.

17. Ahdritz, G., Qin, T., Vyas, N., Barak, B., & Edelman, B. L. (2024). [Distinguishing the Knowable from the Unknowable with Language Models](https://arxiv.org/abs/2402.03563).  
    Text-only but conceptually essential for knowledge-based hallucination and irreducible uncertainty.

### Foundational metrics, benchmarks, mitigation baselines, and architectures

18. Rohrbach, A., Hendricks, L. A., Burns, K., Darrell, T., & Saenko, K. (2018). [Object Hallucination in Image Captioning](https://arxiv.org/abs/1809.02156).  
    Introduces CHAIR-style object hallucination evaluation and shows why conventional caption metrics can miss image-grounding errors.

19. Li, Y., et al. (2023). [Evaluating Object Hallucination in Large Vision-Language Models](https://arxiv.org/abs/2305.10355).  
    POPE remains the standard baseline for object hallucination.

20. Sun, Z., et al. (2023). [Aligning Large Multimodal Models with Factually Augmented RLHF](https://arxiv.org/abs/2309.14525).  
    Introduces LLaVA-RLHF and MMHal-Bench, providing a foundational multimodal RLHF and hallucination-evaluation baseline.

21. Liu, F., et al. (2023). [HallusionBench](https://arxiv.org/abs/2310.14566).  
    Useful for image-context reasoning and misleading-prior cases.

22. Fu, C., et al. (2023). [MME: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models](https://arxiv.org/abs/2306.13394).  
    Broad perception/cognition benchmark.

23. Wang, J., et al. (2023/2024). [AMBER](https://arxiv.org/abs/2311.07397).  
    Multi-dimensional hallucination evaluation with reduced dependence on LLM-as-judge.

24. Leng, S., et al. (2023). [Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding](https://arxiv.org/abs/2311.16922).  
    VCD is a foundational training-free contrastive decoding baseline for object hallucination mitigation.

25. Huang, Q., et al. (2023). [OPERA: Alleviating Hallucination in Multi-Modal Large Language Models via Over-Trust Penalty and Retrospection-Allocation](https://arxiv.org/abs/2311.17911).  
    OPERA is a foundational decoding-time baseline that links hallucination to over-trust and attention aggregation patterns.

26. Liu, H., Li, C., Wu, Q., & Lee, Y. J. (2023). [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485).  
    LLaVA architecture baseline for open LVLM experiments.

27. Li, J., Li, D., Savarese, S., & Hoi, S. (2023). [BLIP-2](https://arxiv.org/abs/2301.12597).  
    Useful connector architecture baseline.

## 10. Limitations

- Most 2025-2026 papers are arXiv preprints, so strong claims should be treated as provisional until peer-reviewed or independently reproduced.
- The recent literature is heavily biased toward object hallucination and attention-based mitigation; knowledge-based hallucination is improving but still less connected to architecture ablations.
- A full systematic review should run citation chaining through Semantic Scholar/OpenAlex and check accepted conference versions.
- The exact implementation of AIR/CAI/Fox/ADAPT/OPPO may differ across model families, so benchmark gains should not be assumed transferable without reimplementation.
