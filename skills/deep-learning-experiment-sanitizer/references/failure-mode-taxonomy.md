# Failure-Mode Taxonomy

## Core classes

| Class | Typical signature | Fast checks |
| --- | --- | --- |
| Split leakage | unrealistically high performance, low error diversity, weak seed sensitivity | dedup, group holdout, temporal holdout, generator audit |
| Target leakage | performance survives when intended signal is weak | inspect inputs for future info, labels, IDs, preprocessing fitted on all data |
| Shortcut learning | model fails under small semantic-preserving edits | ablate proxy features, counterfactual edits, background or formatting perturbations |
| Evaluation leakage | gains vanish on fresh split or strict protocol | isolate tuning split, rerun untouched test, inspect report-generation code |
| Metric mismatch | metric improves while real task quality does not | evaluate task-faithful metrics and hard slices |
| Optimization artifact | result depends on one seed or unstable checkpoint | multi-seed rerun, confidence intervals, checkpoint selection audit |

## Modality and architecture amplifiers

### Vision

- Backgrounds, acquisition devices, borders, rulers, watermarks, text overlays, and crop conventions can dominate the label.
- CNNs and ViTs can lock onto texture or acquisition artifacts instead of semantics.
- Check with masking, background swaps, crop perturbations, and site-balanced splits.

### NLP

- Prompt wrappers, answer formatting, benchmark template reuse, document source, and position cues can leak labels.
- Transformers can memorize template-specific cues and benchmark contamination from public corpora.
- Check with paraphrases, template swaps, source holdouts, and retrieval-contamination audits.

### Time series and tabular

- Future-derived features, leakage through normalization, entity overlap, and post-treatment variables are common.
- Sequence models exploit timestamp regularities, missingness patterns, and window-construction mistakes.
- Check with chronological splits, entity holdouts, feature lineage review, and per-feature ablations.

### Graphs and algorithm learning

- Node order, graph size, execution length, source-node conventions, padding patterns, and dataset generator reuse can become shortcuts.
- GNNs and recurrent graph executors can exploit structural regularities that are easier than the intended algorithmic reasoning.
- Check with graph-family holdouts, randomized node relabeling, size-controlled evaluation, generator-seed isolation, and counterfactual graph perturbations that preserve the intended algorithm target.

### Multimodal and retrieval-augmented systems

- Pairing metadata, filename alignment, retrieval index contamination, and cross-modal duplicates create hidden shortcuts.
- Fusion models can ignore one modality if another carries an easier proxy.
- Check modality dropout, swapped-pair negatives, retrieval-corpus decontamination, and source holdouts.

### Generative, diffusion, and reinforcement learning systems

- Reward-model leakage, simulator quirks, prompt contamination, teacher-forcing mismatch, and evaluation-by-example overlap are common.
- Generative models can memorize benchmarks or exploit evaluator weaknesses instead of learning the intended capability.
- RL systems can overfit to environment bugs, shaping terms, reset policies, or narrow simulators while failing the real task.
- Check held-out prompt families, decontaminated eval sets, rollout-level interventions, simulator variation, and reward-channel ablations.

## Claim-specific red flags

| Claim type | Red flag |
| --- | --- |
| "generalizes to unseen entities" | split is by sample instead of entity |
| "learned reasoning" | shallow proxies or templates solve the benchmark |
| "uses modality X" | performance barely changes when X is removed |
| "robust" | only in-distribution average metrics are reported |
| "interpretable" | explanation method is used as proof instead of intervention |

## Minimal decisive interventions

- Remove the suspected proxy.
- Hold out the suspected grouping factor.
- Randomize the proxy while preserving the target.
- Preserve the proxy while breaking the target.
- Compare against a trivial baseline that only sees the proxy.

If the headline result survives these tests, the claim becomes materially stronger.
