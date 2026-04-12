---
name: deep-learning-experiment-sanitizer
description: Critically audit deep learning experiments for invalid claims, data leakage, shortcut learning, confounding, evaluation bugs, distribution mismatch, and architecture-specific failure modes. Use when Codex must review training code, evaluation code, dataset construction, split logic, experiment plans, ablations, metrics, notebooks, or result summaries before trusting conclusions.
---

# Deep Learning Experiment Sanitizer

Audit claims skeptically. Treat every reported gain as untrusted until the data pipeline, split construction, model inputs, and evaluation protocol survive targeted falsification attempts.

## Quick Start

1. Identify the exact claim being made.
2. Identify the intended generalization target.
3. Read [references/audit-protocol.md](references/audit-protocol.md).
4. Read [references/failure-mode-taxonomy.md](references/failure-mode-taxonomy.md) when the task, data modality, or architecture introduces specific risks.
5. Produce findings before summary. Rank them by how strongly they threaten the claim.

## Audit Workflow

### 1. Pin the Claim

Write the narrowest defensible claim in one sentence. Good examples:

- "This model predicts the next graph-algorithm state on unseen graphs from the same generator."
- "This classifier generalizes across patients rather than across slices from the same patient."
- "This language model learned the task rather than memorizing benchmark artifacts."

If the claim is vague, narrow it before auditing anything else.

### 2. Build the Evidence Surface

Inspect the artifacts that determine validity:

- dataset generation, filtering, splitting, and preprocessing code
- configs, seeds, checkpoint initialization, and resume logic
- model inputs and any metadata channels
- training and evaluation rollout semantics
- metrics, selection criteria, ablations, and plots
- notebooks, reports, and any code that aggregates final tables

Missing provenance is a red flag. State it explicitly.

### 3. Attack the Highest-Risk Invalidators First

Prioritize in this order unless the evidence says otherwise:

1. split contamination and duplicate leakage
2. target leakage and privileged features
3. shortcut features or confounds that dominate the intended signal
4. evaluation contamination, metric bugs, or model-selection leakage
5. instability, seed sensitivity, or unsupported conclusions

Do not spend time polishing lower-risk explanations before the top invalidators have been checked.

### 4. Reason by Data Flow, Not Architecture Names

Do not assume that "transformer", "GNN", or "CNN" determines the failure mode. Trace:

- what information enters the model
- which parts of it could proxy the target
- what inductive bias amplifies those proxies
- how the evaluation protocol might reward shortcut behavior

Then use the taxonomy reference to specialize the audit.
If the architecture is novel, reduce it to inputs, targets, recurrence, memory, retrieval, and training objective before forming hypotheses.

### 5. Demand Falsification Tests

For every suspicious success mode, propose the smallest decisive test:

- deduplicate near-identical examples across splits
- hold out by group, source, time, site, graph family, or template
- remove suspected shortcut features
- randomize labels or suspected proxy channels
- compare against trivial or leakage-matched baselines
- evaluate counterfactual edits that preserve semantics but break shortcuts
- rerun with different seeds and report variance, not only the best run

Metrics alone are insufficient. Prefer tests that can make the gain disappear if it is not real.

### 6. Report Like a Reviewer

Return findings first. For each finding include:

- claim at risk
- failure mode
- concrete evidence
- confidence using the repo convention: unlikely, plausible, probable, very probable, almost certain
- exact rerun or code change needed to resolve it

If no decisive flaw is found, still report residual risks and the strongest remaining uncertainty.

## Output Contract

Use this structure unless the user requests something else:

1. Findings
2. Open questions or missing artifacts
3. Minimal rerun plan
4. Short claim assessment: what is supported, what is not

Prefer statements such as "The reported gain is probably explained by split leakage" over vague phrasing.

## Working Rules

- Treat benchmark familiarity as a possible leak source.
- Treat post-selection analysis on the test set as contamination unless clearly isolated.
- Treat feature engineering, augmentation, normalization, and caching as possible leakage points.
- Treat ablations that keep the shortcut intact as weak evidence.
- Distinguish bugs from scientific invalidity; both matter, but they are not the same.
- Prefer exact artifact references over general advice.
- Do not claim exhaustive coverage. Claim prioritized coverage plus the strongest unresolved blind spots.

## References

- Use [references/audit-protocol.md](references/audit-protocol.md) for the main procedure and reporting standard.
- Use [references/failure-mode-taxonomy.md](references/failure-mode-taxonomy.md) for modality-specific and architecture-amplified failure modes.
